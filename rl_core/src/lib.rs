pub mod settings;
pub mod networks;
pub mod ppo;
pub mod ppo_buffer;
pub mod ppo_trainer;
pub mod env_manager;
pub mod grpc;
pub mod grpc_server;
pub mod multi_env_manager;
pub mod side_channel;

pub mod communicator_objects {
    tonic::include_proto!("communicator_objects");
}