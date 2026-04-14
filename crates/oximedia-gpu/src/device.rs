//! GPU device management and enumeration

use crate::{GpuError, Result};
use std::sync::Arc;
use wgpu::{
    Adapter, Device, DeviceDescriptor, Features, Instance, Limits, PowerPreference, Queue,
    RequestAdapterOptions,
};

/// Information about a GPU device
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device name
    pub name: String,
    /// Vendor ID
    pub vendor: u32,
    /// Device ID
    pub device: u32,
    /// Device type (discrete, integrated, virtual, cpu, unknown)
    pub device_type: String,
    /// Backend being used (Vulkan, Metal, DX12, etc.)
    pub backend: String,
}

/// GPU device wrapper
///
/// This structure manages the WGPU device and queue, providing a safe
/// interface for GPU operations.
pub struct GpuDevice {
    device: Arc<Device>,
    queue: Arc<Queue>,
    info: GpuDeviceInfo,
    #[allow(dead_code)]
    adapter: Adapter,
    /// When true, this device was created via the fallback (software) adapter.
    /// GPU-dependent operations will return `GpuError::NotSupported`.
    pub is_fallback: bool,
}

impl GpuDevice {
    /// Create a new GPU device
    ///
    /// # Arguments
    ///
    /// * `device_index` - Optional device index for multi-GPU selection
    ///
    /// # Errors
    ///
    /// Returns an error if no suitable adapter is found or device request fails.
    pub fn new(device_index: Option<usize>) -> Result<Self> {
        let instance = Self::create_instance();
        let adapter = pollster::block_on(Self::select_adapter(&instance, device_index))?;

        let info = Self::adapter_info(&adapter);

        let (device, queue) = pollster::block_on(Self::request_device(&adapter))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            info,
            adapter,
            is_fallback: false,
        })
    }

    /// Create a CPU-only fallback device using the wgpu software (Vulkan Portability /
    /// force_fallback_adapter) path.
    ///
    /// Returns `Err(GpuError::NoAdapter)` only when there is genuinely no wgpu
    /// adapter available on the system (e.g., a truly headless environment with
    /// no software renderer).  In all other cases, the returned device will be
    /// functional, albeit potentially CPU-backed.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::NoAdapter` if no adapter can be obtained through any
    /// backend on this system.
    pub fn new_fallback() -> Result<Self> {
        let instance = Self::create_instance();

        // Attempt software / CPU adapter.  The `force_fallback_adapter` flag tells
        // wgpu to prefer the software (Mesa lavapipe / wgpu-hal-null) renderer.
        let maybe_adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: true,
        }));

        if let Ok(adapter) = maybe_adapter {
            let info = Self::adapter_info(&adapter);
            if let Ok((device, queue)) = pollster::block_on(Self::request_device(&adapter)) {
                return Ok(Self {
                    device: Arc::new(device),
                    queue: Arc::new(queue),
                    info,
                    adapter,
                    is_fallback: true,
                });
            }
        }

        // Primary path failed — try every available strategy via make_null_device.
        Self::make_null_device().ok_or(GpuError::NoAdapter)
    }

    /// Build a minimal "null device" that holds dummy wgpu objects.
    ///
    /// This is used only when both the primary and fallback GPU paths fail.
    /// Returns `None` when no wgpu adapter can be obtained on the current
    /// system (e.g., headless CI without any GPU or software renderer).
    /// Callers that receive `None` must handle the absence of a real device.
    fn make_null_device() -> Option<Self> {
        let null_info = GpuDeviceInfo {
            name: "CPU Null Device".to_string(),
            vendor: 0,
            device: 0,
            device_type: "cpu".to_string(),
            backend: "Null".to_string(),
        };

        // Helper: try to build a GpuDevice from an adapter, returning None on
        // any failure rather than panicking.
        fn try_adapter(adapter: Adapter, info: GpuDeviceInfo) -> Option<GpuDevice> {
            match pollster::block_on(GpuDevice::request_device(&adapter)) {
                Ok((device, queue)) => Some(GpuDevice {
                    device: Arc::new(device),
                    queue: Arc::new(queue),
                    info,
                    adapter,
                    is_fallback: true,
                }),
                Err(_) => None,
            }
        }

        // Attempt 1: GL / software backend with force_fallback.
        let mut gl_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        gl_desc.backends = wgpu::Backends::GL;
        let gl_instance = wgpu::Instance::new(gl_desc);
        if let Ok(adapter) =
            pollster::block_on(gl_instance.request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::None,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        {
            let info = Self::adapter_info(&adapter);
            if let Some(dev) = try_adapter(adapter, info) {
                return Some(dev);
            }
        }

        // Attempt 2: enumerate all adapters from the GL instance and try the first.
        let adapters = pollster::block_on(gl_instance.enumerate_adapters(wgpu::Backends::all()));
        for adapter in adapters {
            let info = Self::adapter_info(&adapter);
            if let Some(dev) = try_adapter(adapter, info) {
                return Some(dev);
            }
        }

        // Attempt 3: default instance with force_fallback.
        let default_instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        if let Ok(adapter) =
            pollster::block_on(default_instance.request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::None,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        {
            let info = Self::adapter_info(&adapter);
            if let Some(dev) = try_adapter(adapter, info) {
                return Some(dev);
            }
        }

        // Attempt 4: enumerate all adapters from the default instance.
        let adapters =
            pollster::block_on(default_instance.enumerate_adapters(wgpu::Backends::all()));
        for adapter in adapters {
            let info = Self::adapter_info(&adapter);
            if let Some(dev) = try_adapter(adapter, info) {
                return Some(dev);
            }
        }

        // No adapter found at all on this system — caller handles None.
        let _ = null_info;
        None
    }

    /// List all available GPU devices
    pub fn list_devices() -> Result<Vec<GpuDeviceInfo>> {
        let instance = Self::create_instance();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
            Ok(adapters.iter().map(Self::adapter_info).collect())
        }

        #[cfg(target_arch = "wasm32")]
        {
            // On wasm, enumerate_adapters is not available; request a single adapter instead
            let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }));
            match adapter {
                Ok(a) => Ok(vec![Self::adapter_info(&a)]),
                Err(_) => Ok(Vec::new()),
            }
        }
    }

    /// Get device information
    #[must_use]
    pub fn info(&self) -> &GpuDeviceInfo {
        &self.info
    }

    /// Get the WGPU device
    #[must_use]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get the WGPU queue
    #[must_use]
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Wait for all GPU operations to complete
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    fn create_instance() -> Instance {
        Instance::new(wgpu::InstanceDescriptor::new_without_display_handle())
    }

    async fn select_adapter(instance: &Instance, device_index: Option<usize>) -> Result<Adapter> {
        if let Some(index) = device_index {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
                return adapters.into_iter().nth(index).ok_or(GpuError::NoAdapter);
            }

            #[cfg(target_arch = "wasm32")]
            {
                // On wasm, enumerate_adapters is not available; only index 0 is supported
                if index != 0 {
                    return Err(GpuError::NoAdapter);
                }
                return instance
                    .request_adapter(&RequestAdapterOptions {
                        power_preference: PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                    .map_err(|_| GpuError::NoAdapter);
            }
        } else {
            // Select high-performance adapter by default
            instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|_| GpuError::NoAdapter)
        }
    }

    async fn request_device(adapter: &Adapter) -> Result<(Device, Queue)> {
        adapter
            .request_device(&DeviceDescriptor {
                label: Some("OxiMedia GPU Device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))
    }

    fn adapter_info(adapter: &Adapter) -> GpuDeviceInfo {
        let info = adapter.get_info();

        let device_type = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => "discrete",
            wgpu::DeviceType::IntegratedGpu => "integrated",
            wgpu::DeviceType::VirtualGpu => "virtual",
            wgpu::DeviceType::Cpu => "cpu",
            wgpu::DeviceType::Other => "unknown",
        };

        let backend = match info.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx12 => "DirectX 12",
            wgpu::Backend::Gl => "OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            _ => "Unknown",
        };

        GpuDeviceInfo {
            name: info.name,
            vendor: info.vendor,
            device: info.device,
            device_type: device_type.to_string(),
            backend: backend.to_string(),
        }
    }
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("info", &self.info)
            .field("is_fallback", &self.is_fallback)
            .finish()
    }
}
