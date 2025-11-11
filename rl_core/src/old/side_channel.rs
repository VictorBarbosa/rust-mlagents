// Side Channel implementation for communicating with Unity
// Based on ml-agents Python implementation

use std::collections::HashMap;

/// UUID for EnvironmentParametersChannel
/// Python: uuid.UUID("534c891e-810f-11ea-a9d0-822485860400")
const ENV_PARAMS_CHANNEL_ID: [u8; 16] = [
    0x1e, 0x89, 0x4c, 0x53, // little-endian UUID bytes
    0x0f, 0x81,
    0xea, 0x11,
    0xa9, 0xd0,
    0x82, 0x24, 0x85, 0x86, 0x04, 0x00,
];

/// UUID for EngineConfigurationChannel
/// Python: uuid.UUID("e951342c-4f7e-11ea-b238-784f4387d1f7")
const ENGINE_CONFIG_CHANNEL_ID: [u8; 16] = [
    0x2c, 0x34, 0x51, 0xe9, // little-endian UUID bytes
    0x7e, 0x4f,
    0xea, 0x11,
    0xb2, 0x38,
    0x78, 0x4f, 0x43, 0x87, 0xd1, 0xf7,
];

/// Environment parameter data types
#[repr(i32)]
enum EnvironmentDataType {
    Float = 0,
    #[allow(dead_code)]
    Sampler = 1,
}

/// OutgoingMessage builder (equivalent to Python's OutgoingMessage)
struct OutgoingMessage {
    buffer: Vec<u8>,
}

impl OutgoingMessage {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
        }
    }

    fn write_int32(&mut self, value: i32) {
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    fn write_float32(&mut self, value: f32) {
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    fn write_bool(&mut self, value: bool) {
        self.write_int32(if value { 1 } else { 0 });
    }

    fn write_string(&mut self, s: &str) {
        let encoded = s.as_bytes();
        self.write_int32(encoded.len() as i32);
        self.buffer.extend_from_slice(encoded);
    }

    fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }
}

/// Serialize environment parameters into side channel format
/// 
/// Format:
/// - 16 bytes: Channel UUID (little-endian)
/// - 4 bytes: Message length (i32, little-endian)
/// - N bytes: Message data
///   For each parameter:
///     - String length (i32) + String bytes (key)
///     - Data type (i32) = 0 for FLOAT
///     - Float value (f32)
pub fn serialize_environment_parameters(params: &HashMap<String, serde_yaml::Value>) -> Vec<u8> {
    if params.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    
    // For each parameter, create a separate message
    for (key, value) in params {
        let mut msg = OutgoingMessage::new();
        
        // Write key
        msg.write_string(key);
        
        // Write data type (FLOAT)
        msg.write_int32(EnvironmentDataType::Float as i32);
        
        // Convert value to float
        let float_value = match value {
            serde_yaml::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    i as f32
                } else if let Some(f) = n.as_f64() {
                    f as f32
                } else {
                    0.0
                }
            }
            serde_yaml::Value::Bool(b) => if *b { 1.0 } else { 0.0 },
            serde_yaml::Value::String(s) => s.parse::<f32>().unwrap_or(0.0),
            _ => 0.0,
        };
        
        // Write float value
        msg.write_float32(float_value);
        
        let message_bytes = msg.into_bytes();
        
        // Append to result:
        // 1. Channel UUID (16 bytes)
        result.extend_from_slice(&ENV_PARAMS_CHANNEL_ID);
        
        // 2. Message length (4 bytes, little-endian)
        result.extend_from_slice(&(message_bytes.len() as i32).to_le_bytes());
        
        // 3. Message data
        result.extend_from_slice(&message_bytes);
    }
    
    result
}

/// Engine configuration settings
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub width: u32,
    pub height: u32,
    pub quality_level: i32,
    pub time_scale: f32,
    pub target_frame_rate: i32,
    pub capture_frame_rate: i32,
    pub no_graphics: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            width: 84,
            height: 84,
            quality_level: 5,
            time_scale: 20.0,
            target_frame_rate: -1,
            capture_frame_rate: 60,
            no_graphics: false,
        }
    }
}

/// Serialize engine configuration into side channel format
///
/// Format matches Python's EngineConfigurationChannel:
/// - 16 bytes: Channel UUID
/// - 4 bytes: Message length
/// - Message data:
///   - width (i32)
///   - height (i32)
///   - quality_level (i32)
///   - time_scale (f32)
///   - target_frame_rate (i32)
///   - capture_frame_rate (i32)
///   - no_graphics (i32: 1 for true, 0 for false)
pub fn serialize_engine_config(config: &EngineConfig) -> Vec<u8> {
    let mut msg = OutgoingMessage::new();
    
    // Write all engine settings
    msg.write_int32(config.width as i32);
    msg.write_int32(config.height as i32);
    msg.write_int32(config.quality_level);
    msg.write_float32(config.time_scale);
    msg.write_int32(config.target_frame_rate);
    msg.write_int32(config.capture_frame_rate);
    msg.write_bool(config.no_graphics);
    
    let message_bytes = msg.into_bytes();
    
    let mut result = Vec::new();
    
    // 1. Channel UUID (16 bytes)
    result.extend_from_slice(&ENGINE_CONFIG_CHANNEL_ID);
    
    // 2. Message length (4 bytes)
    result.extend_from_slice(&(message_bytes.len() as i32).to_le_bytes());
    
    // 3. Message data
    result.extend_from_slice(&message_bytes);
    
    result
}

/// Combine multiple side channel messages into a single byte array
pub fn combine_side_channels(channels: &[Vec<u8>]) -> Vec<u8> {
    let mut result = Vec::new();
    for channel in channels {
        result.extend_from_slice(channel);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_empty_params() {
        let params = HashMap::new();
        let result = serialize_environment_parameters(&params);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_serialize_single_float() {
        let mut params = HashMap::new();
        params.insert("test".to_string(), serde_yaml::Value::Number(42.0.into()));
        
        let result = serialize_environment_parameters(&params);
        
        // Should contain:
        // - 16 bytes UUID
        // - 4 bytes message length
        // - Message data (string length + "test" + type + value)
        assert!(result.len() > 16 + 4);
        
        // Check UUID
        assert_eq!(&result[0..16], &ENV_PARAMS_CHANNEL_ID);
    }

    #[test]
    fn test_serialize_multiple_params() {
        let mut params = HashMap::new();
        params.insert("param1".to_string(), serde_yaml::Value::Number(1.0.into()));
        params.insert("param2".to_string(), serde_yaml::Value::Number(2.0.into()));
        
        let result = serialize_environment_parameters(&params);
        
        // Should have 2 messages
        // Each message: 16 (UUID) + 4 (length) + data
        assert!(result.len() > 40); // At least 2 * (16 + 4)
    }
}
