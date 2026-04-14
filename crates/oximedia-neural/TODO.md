# oximedia-neural TODO

## Current Status
- 6 source files providing lightweight pure Rust neural network inference
- tensor: Core n-dimensional f32 tensor with math operations (add, matmul, mul, sum_along)
- activations: relu, sigmoid, tanh, gelu, leaky_relu, softmax, swish with tensor apply helpers
- layers: LinearLayer, Conv2dLayer, DepthwiseConv2d, BatchNorm1d, MaxPool2d, GlobalAvgPool
- media_models: SceneClassifier, ThumbnailRanker, SrUpscaler, FeatureExtractor
- error: NeuralError type
- Zero external dependencies beyond thiserror (pure Rust)

## Enhancements
- [ ] Add batch dimension support to Tensor -- currently only single-sample inference; batch processing needed for throughput
- [ ] Implement in-place operations on Tensor (relu_inplace, add_inplace) to reduce allocation during inference
- [ ] Add Tensor broadcasting for element-wise ops between different-shaped tensors (e.g., [N,C,H,W] + [1,C,1,1])
- [ ] Implement transposed convolution (ConvTranspose2d) in layers for decoder/upsampling networks
- [ ] Add average pooling (AvgPool2d) alongside existing MaxPool2d
- [ ] Extend BatchNorm1d to BatchNorm2d for use with Conv2d outputs (spatial batch normalization)
- [ ] Add residual/skip connection helper to simplify ResNet-style architecture building
- [ ] Improve SrUpscaler with sub-pixel convolution (PixelShuffle) layer for efficient super-resolution

## New Features
- [ ] Implement ONNX model loading -- parse .onnx protobuf and map ops to existing layers
- [ ] Add model serialization/deserialization (save/load trained weights in a custom binary format)
- [ ] Implement GRU/LSTM recurrent layers for temporal media models (e.g., audio classification)
- [ ] Add attention mechanism (self-attention, cross-attention) for transformer-based architectures
- [ ] Implement Tensor quantization (INT8) for faster inference on resource-constrained devices
- [ ] Add model graph builder API for defining networks declaratively (Sequential, Functional)
- [x] Implement deformable convolution for object detection in video frames
- [x] Add face detection model using existing Conv2d + pooling layers (pre-defined architecture)
- [x] Implement optical flow estimation model for motion analysis in video

## Performance
- [ ] Add SIMD-accelerated matmul using portable_simd for f32 dot products (currently scalar loops)
- [ ] Implement tiled/blocked matrix multiplication to improve cache locality for large tensors
- [ ] Add multi-threaded Conv2d using rayon for parallel output channel computation
- [ ] Implement im2col optimization for Conv2d to convert convolution to matrix multiplication
- [ ] Add memory pool/arena allocator for Tensor data to reduce allocation overhead during inference
- [ ] Profile and optimize Tensor indexing -- consider flat indexing with stride computation

## Testing
- [ ] Add numerical accuracy test: compare LinearLayer output against known hand-computed values
- [ ] Test Conv2dLayer with 3x3 kernel on known input, verify output matches manual convolution
- [ ] Add gradient-free training test: verify forward pass produces deterministic output for fixed weights
- [ ] Test SceneClassifier with synthetic scene patterns (uniform color -> sky, gradient -> landscape)
- [ ] Add tensor shape mismatch error tests for all operations (matmul, add, conv)
- [ ] Benchmark inference latency for each media_model at common resolutions (720p, 1080p)

## Documentation
- [ ] Add architecture guide showing supported layer types and how to compose them into models
- [ ] Document the media_models with input/output specifications (tensor shapes, value ranges)
- [ ] Add performance comparison table: inference time per model at various resolutions
