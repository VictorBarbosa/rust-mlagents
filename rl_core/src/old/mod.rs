pub mod settings;
#[allow(dead_code)]
mod networks;  // Kept for historical reference only
pub mod env_manager;
pub mod grpc;
pub mod grpc_server;
pub mod multi_env_manager;
pub mod side_channel;
pub mod logging;

// Legacy module exports for backward compatibility
// TODO: Implement these modules
// pub mod ppo {
//     pub use crate::algorithms::ppo::trainer::*;
// }
// pub mod ppo_buffer {
//     pub use crate::algorithms::ppo::buffer::*;
// }
// pub mod ppo_trainer {
//     pub use crate::algorithms::ppo::algorithm::*;
// }

// Temporary stub for communicator_objects until proto files are generated
pub mod communicator_objects {
    // TODO: Generate from proto files using tonic::include_proto!("communicator_objects");
    // For now, we'll use the ones from the parent crate
}