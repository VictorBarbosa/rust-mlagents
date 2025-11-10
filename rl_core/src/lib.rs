pub mod settings;
pub mod networks;
pub mod algorithms;
pub mod env_manager;
pub mod grpc;
pub mod grpc_server;
pub mod multi_env_manager;
pub mod side_channel;
pub mod logging;

// Legacy module exports for backward compatibility
pub mod ppo {
    pub use crate::algorithms::ppo::trainer::*;
}
pub mod ppo_buffer {
    pub use crate::algorithms::ppo::buffer::*;
}
pub mod ppo_trainer {
    pub use crate::algorithms::ppo::algorithm::*;
}

pub mod communicator_objects {
    tonic::include_proto!("communicator_objects");
}