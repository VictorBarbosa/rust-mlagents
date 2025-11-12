// SAC (Soft Actor-Critic) Implementation
pub mod config;
pub mod networks;
pub mod replay_buffer;
pub mod trainer;
pub mod onnx_export;
// pub mod unity_env;  // TODO: Fix Unity integration

#[cfg(test)]
#[path = "../test/sac/test_export.rs"]
mod test_export;

pub use config::SACConfig;
pub use networks::{ActorNetwork, CriticNetwork};
pub use replay_buffer::{ReplayBuffer, Transition, Batch};
pub use trainer::{SACTrainer, SACMetrics};
pub use onnx_export::ONNXExporter;
// pub use unity_env::{UnityEnvironment, UnityTrainer};  // TODO: Fix Unity integration
