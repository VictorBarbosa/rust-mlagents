// ML-Agents Core Library - Rust implementation
// Equivalent to mlagents Python package

pub mod trainers;
pub mod utils;
pub mod plugins;
pub mod cli;
pub mod old;  // Keep old implementation for gRPC and environment communication
pub mod env;  // Unity environment wrapper

// Re-export main types
pub use trainers::{
    run_training,
    run_cli,
    parse_command_line,
    TrainerController,
    RunOptions,
};

// Include generated gRPC code from build.rs
pub mod communicator_objects {
    tonic::include_proto!("communicator_objects");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_string() {
        let version = trainers::learn::get_version_string();
        assert!(version.contains("ml-agents-rust"));
    }
}
