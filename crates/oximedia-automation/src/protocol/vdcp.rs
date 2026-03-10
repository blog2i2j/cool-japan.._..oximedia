//! VDCP (Video Disk Control Protocol) implementation.

use crate::protocol::serial::SerialPort;
use crate::{AutomationError, Result};
use bytes::{BufMut, BytesMut};
use tracing::{debug, info};

/// VDCP command codes.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum VdcpCommand {
    /// Play command
    Play = 0x01,
    /// Stop command
    Stop = 0x02,
    /// Cue command
    Cue = 0x03,
    /// Status request
    Status = 0x10,
}

/// VDCP protocol handler.
pub struct VdcpProtocol {
    serial: SerialPort,
}

impl VdcpProtocol {
    /// Create a new VDCP protocol handler.
    pub async fn new(port: &str) -> Result<Self> {
        info!("Creating VDCP protocol on port: {}", port);

        let serial = SerialPort::new(port, 38400)?;

        Ok(Self { serial })
    }

    /// Close the connection.
    pub async fn close(&mut self) -> Result<()> {
        self.serial.close()
    }

    /// Send play command.
    pub async fn send_play(&mut self) -> Result<()> {
        debug!("Sending VDCP play command");
        self.send_command(VdcpCommand::Play, &[]).await
    }

    /// Send stop command.
    pub async fn send_stop(&mut self) -> Result<()> {
        debug!("Sending VDCP stop command");
        self.send_command(VdcpCommand::Stop, &[]).await
    }

    /// Send cue command.
    pub async fn send_cue(&mut self, timecode: &str) -> Result<()> {
        debug!("Sending VDCP cue command to: {}", timecode);

        // Parse timecode and encode as VDCP format
        let tc_bytes = self.encode_timecode(timecode)?;
        self.send_command(VdcpCommand::Cue, &tc_bytes).await
    }

    /// Get device status.
    pub async fn get_status(&mut self) -> Result<String> {
        debug!("Requesting VDCP status");
        self.send_command(VdcpCommand::Status, &[]).await?;

        // In a real implementation, this would read and parse the response
        Ok("OK".to_string())
    }

    /// Send a VDCP command.
    async fn send_command(&mut self, command: VdcpCommand, data: &[u8]) -> Result<()> {
        let mut buffer = BytesMut::with_capacity(256);

        // VDCP packet format: [STX][LEN][CMD][DATA...][CHK][ETX]
        buffer.put_u8(0x02); // STX
        buffer.put_u8((data.len() + 1) as u8); // Length
        buffer.put_u8(command as u8); // Command
        buffer.put_slice(data); // Data

        // Calculate checksum
        let checksum = self.calculate_checksum(&buffer[1..]);
        buffer.put_u8(checksum);
        buffer.put_u8(0x03); // ETX

        self.serial.write(&buffer)?;

        Ok(())
    }

    /// Encode timecode to VDCP format.
    fn encode_timecode(&self, timecode: &str) -> Result<Vec<u8>> {
        // Parse timecode format: HH:MM:SS:FF
        let parts: Vec<&str> = timecode.split(':').collect();
        if parts.len() != 4 {
            return Err(AutomationError::Protocol(format!(
                "Invalid timecode format: {timecode}"
            )));
        }

        let hours: u8 = parts[0]
            .parse()
            .map_err(|_| AutomationError::Protocol("Invalid hours".to_string()))?;
        let minutes: u8 = parts[1]
            .parse()
            .map_err(|_| AutomationError::Protocol("Invalid minutes".to_string()))?;
        let seconds: u8 = parts[2]
            .parse()
            .map_err(|_| AutomationError::Protocol("Invalid seconds".to_string()))?;
        let frames: u8 = parts[3]
            .parse()
            .map_err(|_| AutomationError::Protocol("Invalid frames".to_string()))?;

        Ok(vec![hours, minutes, seconds, frames])
    }

    /// Calculate VDCP checksum.
    fn calculate_checksum(&self, data: &[u8]) -> u8 {
        data.iter().fold(0u8, |acc, &byte| acc.wrapping_add(byte))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_timecode() {
        let protocol = VdcpProtocol {
            serial: SerialPort::mock(),
        };

        let result = protocol.encode_timecode("01:23:45:12");
        assert!(result.is_ok());

        let tc = result.expect("tc should be valid");
        assert_eq!(tc, vec![1, 23, 45, 12]);
    }

    #[test]
    fn test_invalid_timecode() {
        let protocol = VdcpProtocol {
            serial: SerialPort::mock(),
        };

        let result = protocol.encode_timecode("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_checksum() {
        let protocol = VdcpProtocol {
            serial: SerialPort::mock(),
        };

        let data = vec![0x01, 0x02, 0x03];
        let checksum = protocol.calculate_checksum(&data);
        assert_eq!(checksum, 0x06);
    }
}
