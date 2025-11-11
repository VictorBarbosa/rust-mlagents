// SAC Configuration
use serde::{Deserialize, Serialize};
use tch::Kind;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SACConfig {
    // Network architecture
    pub hidden_layers: Vec<i64>,
    
    // Precision (use float32 for MPS compatibility, float64 for CPU/CUDA if desired)
    #[serde(skip, default = "default_dtype")]
    pub dtype: Kind,
    
    // Hyperparameters
    pub gamma: f64,           // Discount factor
    pub tau: f64,             // Soft update coefficient
    pub lr_actor: f64,        // Actor learning rate
    pub lr_critic: f64,       // Critic learning rate
    pub lr_alpha: f64,        // Alpha learning rate
    
    // Replay buffer
    pub buffer_size: usize,
    pub batch_size: usize,
    pub warmup_steps: usize,
    
    // Entropy
    pub auto_alpha: bool,
    pub init_alpha: f64,
    pub target_entropy: Option<f64>,
    
    // Training
    pub gradient_steps: usize,
    pub target_update_interval: usize,
    
    // Checkpoints
    pub checkpoint_interval: usize,
    pub save_onnx: bool,
}

fn default_dtype() -> Kind {
    Kind::Float  // float32 for cross-platform compatibility
}

impl Default for SACConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![256, 256],
            dtype: default_dtype(),
            gamma: 0.99,
            tau: 0.005,
            lr_actor: 3e-4,
            lr_critic: 3e-4,
            lr_alpha: 3e-4,
            buffer_size: 1_000_000,
            batch_size: 256,
            warmup_steps: 1000,
            auto_alpha: true,
            init_alpha: 0.2,
            target_entropy: None,
            gradient_steps: 1,
            target_update_interval: 1,
            checkpoint_interval: 10000,
            save_onnx: true,
        }
    }
}

impl SACConfig {
    /// Auto-configure dtype based on device
    /// MPS requires float32, CPU/CUDA can use float64 but float32 is safer
    pub fn with_device(mut self, device: tch::Device) -> Self {
        self.dtype = match device {
            tch::Device::Mps => Kind::Float,      // MPS requires float32
            tch::Device::Cuda(_) => Kind::Float,  // Use float32 for consistency
            tch::Device::Cpu => Kind::Float,      // Use float32 for consistency
            _ => Kind::Float,
        };
        self
    }
}
