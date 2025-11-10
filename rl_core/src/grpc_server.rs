// gRPC Server implementation following Python's RpcCommunicator pattern
// Python is SERVER, Unity is CLIENT that connects to it

use tonic::{transport::Server, Request, Response, Status};
use tokio::sync::mpsc;
use std::net::SocketAddr;

use crate::communicator_objects::{
    UnityMessageProto, UnityInputProto, UnityOutputProto,
    UnityRlCapabilitiesProto,
    unity_to_external_proto_server::{UnityToExternalProto, UnityToExternalProtoServer},
};

// Channel-based communication between gRPC service and main trainer loop
// Similar to Python's Pipe() mechanism
#[derive(Clone)]
pub struct UnityServiceImpl {
    // Sender to main loop: Unity -> Trainer
    unity_to_trainer_tx: mpsc::UnboundedSender<UnityMessageProto>,
    // Receiver from main loop: Trainer -> Unity
    trainer_to_unity_rx: std::sync::Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<UnityMessageProto>>>,
}

impl UnityServiceImpl {
    pub fn new(
        unity_to_trainer_tx: mpsc::UnboundedSender<UnityMessageProto>,
        trainer_to_unity_rx: mpsc::UnboundedReceiver<UnityMessageProto>,
    ) -> Self {
        Self {
            unity_to_trainer_tx,
            trainer_to_unity_rx: std::sync::Arc::new(tokio::sync::Mutex::new(trainer_to_unity_rx)),
        }
    }
}

#[tonic::async_trait]
impl UnityToExternalProto for UnityServiceImpl {
    async fn exchange(
        &self,
        request: Request<UnityMessageProto>,
    ) -> Result<Response<UnityMessageProto>, Status> {
        let msg_from_unity = request.into_inner();
        
        self.unity_to_trainer_tx
            .send(msg_from_unity)
            .map_err(|e| Status::internal(format!("Failed to send to trainer: {}", e)))?;
        
        let mut rx = self.trainer_to_unity_rx.lock().await;
        let response_msg = rx.recv().await
            .ok_or_else(|| Status::internal("Trainer channel closed"))?;
        
        Ok(Response::new(response_msg))
    }
}

pub struct GrpcServer {
    port: u16,
    unity_to_trainer_rx: mpsc::UnboundedReceiver<UnityMessageProto>,
    trainer_to_unity_tx: mpsc::UnboundedSender<UnityMessageProto>,
    pub verbose: bool,
}

impl GrpcServer {
    pub fn new(port: u16) -> Self {
        Self::new_with_verbosity(port, false)
    }
    
    pub fn new_with_verbosity(port: u16, verbose: bool) -> Self {
        let (unity_to_trainer_tx, unity_to_trainer_rx) = mpsc::unbounded_channel();
        let (trainer_to_unity_tx, trainer_to_unity_rx) = mpsc::unbounded_channel();
        
        let service = UnityServiceImpl::new(unity_to_trainer_tx, trainer_to_unity_rx);
        let addr: SocketAddr = format!("0.0.0.0:{}", port).parse().unwrap();
        
        // Spawn server in background (quiet mode)
        tokio::spawn(async move {
            let _ = Server::builder()
                .add_service(UnityToExternalProtoServer::new(service))
                .serve(addr)
                .await;
        });
        
        Self {
            port,
            unity_to_trainer_rx,
            trainer_to_unity_tx,
            verbose,
        }
    }
    
    pub fn port(&self) -> u16 {
        self.port
    }
    
    // Initialize: wait for Unity to connect and send init message
    pub async fn initialize(&mut self, init_config: InitConfig) -> Result<UnityOutputProto, String> {
        // Wait for Unity's initialization message
        let unity_init_msg = self.unity_to_trainer_rx.recv().await
            .ok_or_else(|| "Unity didn't connect".to_string())?;
        
        // ML-Agents protocol: Unity sends OUTPUT first (contains initialization request)
        // We respond with INPUT (our capabilities and config)
        if unity_init_msg.unity_output.is_none() {
            return Err("No initialization output from Unity".to_string());
        }
        

        
        // Prepare our initialization INPUT to Unity (our capabilities)
        let caps = UnityRlCapabilitiesProto {
            base_rl_capabilities: true,
            concatenated_png_observations: true,
            compressed_channel_mapping: true,
            hybrid_actions: true,
            training_analytics: false,
            variable_length_observation: true,
            multi_agent_groups: true,
        };
        
        let init_input = crate::communicator_objects::UnityRlInitializationInputProto {
            seed: init_config.seed,
            communication_version: init_config.communication_version.clone(),
            package_version: init_config.package_version.clone(),
            capabilities: Some(caps),
            num_areas: init_config.num_areas,
            environment_parameters: std::collections::HashMap::new(), // Not used - sent via side_channel before first reset
        };
        
        let input = UnityInputProto {
            rl_initialization_input: Some(init_input),
            rl_input: None,
        };
        
        // Send response back to Unity (we send INPUT, Unity sends OUTPUT)
        let response_msg = UnityMessageProto {
            header: Some(crate::communicator_objects::HeaderProto { status: 200, message: String::new() }),
            unity_input: Some(input),
            unity_output: None,
        };
        
        self.trainer_to_unity_tx.send(response_msg)
            .map_err(|e| format!("Failed to send init response: {}", e))?;
        
        // Wait for Unity's empty ack
        let _ack_msg = self.unity_to_trainer_rx.recv().await
            .ok_or_else(|| "Unity didn't send ack".to_string())?;
        
        // Now we need to TRIGGER Unity to send specs by sending an empty input requesting reset
        use crate::communicator_objects::{UnityRlInputProto, CommandProto};
        
        let reset_input = UnityInputProto {
            rl_input: Some(UnityRlInputProto {
                agent_actions: std::collections::HashMap::new(),
                command: CommandProto::Reset as i32,
                side_channel: vec![],
            }),
            rl_initialization_input: None,
        };
        
        let reset_msg = UnityMessageProto {
            header: Some(crate::communicator_objects::HeaderProto { status: 200, message: String::new() }),
            unity_input: Some(reset_input),
            unity_output: None,
        };
        
        self.trainer_to_unity_tx.send(reset_msg)
            .map_err(|e| format!("Failed to send reset: {}", e))?;
        
        // Now Unity should send the actual output with brain_parameters
        let specs_msg = self.unity_to_trainer_rx.recv().await
            .ok_or_else(|| "Unity didn't send specs after reset".to_string())?;
        
        specs_msg.unity_output
            .ok_or_else(|| "No output in specs message".to_string())
    }
    
    // Step: send actions, receive observations
    pub async fn step(&mut self, actions: UnityInputProto) -> Result<UnityOutputProto, String> {
        self.step_with_side_channel(actions, vec![]).await
    }
    
    pub async fn step_with_side_channel(&mut self, actions: UnityInputProto, side_channel: Vec<u8>) -> Result<UnityOutputProto, String> {
        // Merge side_channel into actions if needed
        let mut actions_with_sc = actions;
        if !side_channel.is_empty() {
            if let Some(ref mut rl_input) = actions_with_sc.rl_input {
                rl_input.side_channel = side_channel;
            }
        }
        
        let step_msg = UnityMessageProto {
            header: Some(crate::communicator_objects::HeaderProto { status: 200, message: String::new() }),
            unity_input: Some(actions_with_sc),
            unity_output: None,
        };
        
        self.trainer_to_unity_tx.send(step_msg)
            .map_err(|e| format!("Failed to send step: {}", e))?;
        
        let response = self.unity_to_trainer_rx.recv().await
            .ok_or_else(|| "Unity disconnected".to_string())?;
        
        response.unity_output.ok_or_else(|| "No output from Unity".to_string())
    }
    
    pub async fn reset(&mut self) -> Result<UnityOutputProto, String> {
        self.reset_with_side_channel(vec![]).await
    }
    
    pub async fn reset_with_side_channel(&mut self, side_channel: Vec<u8>) -> Result<UnityOutputProto, String> {
        use crate::communicator_objects::{UnityRlInputProto, CommandProto};
        
        let reset_input = UnityInputProto {
            rl_input: Some(UnityRlInputProto {
                agent_actions: std::collections::HashMap::new(),
                command: CommandProto::Reset as i32,
                side_channel,
            }),
            rl_initialization_input: None,
        };
        
        self.step(reset_input).await
    }
}

#[derive(Clone, Debug)]
pub struct InitConfig {
    pub seed: i32,
    pub num_areas: i32,
    pub communication_version: String,
    pub package_version: String,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self {
            seed: -1,
            num_areas: 1,
            communication_version: "1.5.0".to_string(),
            package_version: "0.1.0".to_string(),
        }
    }
}
