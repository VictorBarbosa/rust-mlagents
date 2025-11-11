// Unity Environment wrapper for training
use std::path::PathBuf;
use tokio::time::{sleep, Duration};

use crate::old::grpc_server::{GrpcServer, InitConfig};
use crate::communicator_objects::{UnityInputProto, UnityRlInputProto, AgentInfoProto};

pub struct UnityEnvironment {
    port: u16,
    _process: Option<std::process::Child>,
    grpc_server: GrpcServer,
    // Store agent IDs and behavior name from last observation
    last_behavior_name: Option<String>,
    last_agent_ids: Vec<i32>,
}

impl UnityEnvironment {
    pub async fn connect(
        unity_path: &PathBuf,
        base_port: u16,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use std::process::{Command, Stdio};
        
        println!("📂 Unity path: {}", unity_path.display());
        
        if !unity_path.exists() {
            return Err(format!("Unity executable not found: {}", unity_path.display()).into());
        }
        
        // Handle macOS .app bundles
        let exec_path = if unity_path.extension().and_then(|s| s.to_str()) == Some("app") {
            let macos_dir = unity_path.join("Contents/MacOS");
            println!("📱 Detected macOS app bundle");
            
            let mut found_exec: Option<PathBuf> = None;
            if macos_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(&macos_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_file() {
                            #[cfg(unix)]
                            {
                                use std::os::unix::fs::PermissionsExt;
                                if let Ok(metadata) = std::fs::metadata(&path) {
                                    let permissions = metadata.permissions();
                                    if permissions.mode() & 0o111 != 0 {
                                        println!("   Found executable: {}", path.display());
                                        found_exec = Some(path);
                                        break;
                                    }
                                }
                            }
                            #[cfg(not(unix))]
                            {
                                found_exec = Some(path);
                                break;
                            }
                        }
                    }
                }
            }
            
            found_exec.ok_or_else(|| {
                format!("No executable found in app bundle: {}", unity_path.display())
            })?
        } else {
            unity_path.clone()
        };
        
        if !exec_path.exists() {
            return Err(format!("Unity executable not found: {}", exec_path.display()).into());
        }
        
        // Start gRPC server FIRST (before launching Unity)
        println!("🔌 Starting gRPC server on port {}...", base_port);
        let mut grpc_server = GrpcServer::new(base_port);
        
        // Give server a moment to start
        sleep(Duration::from_millis(500)).await;
        println!("✅ gRPC server ready!");
        
        // Launch Unity process
        println!("🚀 Launching Unity environment...");
        println!("   Command: {}", exec_path.display());
        println!("   Port: {}", base_port);
        
        let child = Command::new(&exec_path)
            .arg("--mlagents-port")
            .arg(base_port.to_string())
            .arg("-logFile")
            .arg("-")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        
        println!("⏳ Waiting for Unity to connect...");
        
        // Initialize connection (this waits for Unity to connect)
        match grpc_server.initialize(Default::default()).await {
            Ok(output) => {
                println!("✅ Unity connected successfully!");
                println!("   Received initialization from Unity");
                
                Ok(Self {
                    port: base_port,
                    _process: Some(child),
                    grpc_server,
                    last_behavior_name: None,
                    last_agent_ids: Vec::new(),
                })
            }
            Err(e) => {
                Err(format!("Failed to initialize Unity connection: {}", e).into())
            }
        }
    }
    
    pub fn port(&self) -> u16 {
        self.port
    }
    
    pub async fn reset(&mut self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        println!("🔄 Resetting environment...");
        
        let output = self.grpc_server.reset().await
            .map_err(|e| format!("Reset failed: {}", e))?;
        
        // Extract observations from first agent and store agent IDs
        if let Some(rl_output) = output.rl_output {
            for (behavior_name, agent_info_list) in rl_output.agent_infos.iter() {
                // Store behavior name and agent IDs
                self.last_behavior_name = Some(behavior_name.clone());
                self.last_agent_ids = agent_info_list.value.iter().map(|a| a.id).collect();
                
                if let Some(first_agent) = agent_info_list.value.first() {
                    if let Some(obs) = first_agent.observations.first() {
                        // Extract float data based on the observation_data oneof
                        let obs_data: Vec<f32> = if let Some(obs_data) = &obs.observation_data {
                            use crate::communicator_objects::observation_proto::ObservationData;
                            match obs_data {
                                ObservationData::FloatData(float_data) => {
                                    float_data.data.clone()
                                }
                                ObservationData::CompressedData(_) => {
                                    // TODO: decompress if needed
                                    vec![]
                                }
                            }
                        } else {
                            vec![]
                        };
                        
                        println!("✅ Reset complete, got {} observations from {} agents", 
                                 obs_data.len(), self.last_agent_ids.len());
                        return Ok(obs_data);
                    }
                }
            }
        }
        
        Err("No observations received from Unity".into())
    }
    
    pub async fn step(
        &mut self,
        action: Vec<f32>,
    ) -> Result<(Vec<f32>, f32, bool), Box<dyn std::error::Error>> {
        use crate::communicator_objects::{AgentActionProto, unity_rl_input_proto::ListAgentActionProto};
        use std::collections::HashMap;
        
        // Create action message
        let mut rl_input = UnityRlInputProto::default();
        rl_input.command = 0; // STEP command
        
        // Populate with actual agent actions
        if let Some(behavior_name) = &self.last_behavior_name {
            let mut agent_actions_map: HashMap<String, ListAgentActionProto> = HashMap::new();
            
            // For now, send same action to all agents
            let mut agent_actions_list = Vec::new();
            for _agent_id in &self.last_agent_ids {
                let mut agent_action = AgentActionProto::default();
                agent_action.continuous_actions = action.clone();
                agent_actions_list.push(agent_action);
            }
            
            let list_agent_action = ListAgentActionProto {
                value: agent_actions_list,
            };
            
            agent_actions_map.insert(behavior_name.clone(), list_agent_action);
            rl_input.agent_actions = agent_actions_map;
        }
        
        let mut unity_input = UnityInputProto::default();
        unity_input.rl_input = Some(rl_input);
        
        // Send action and get result
        let output = self.grpc_server.step(unity_input).await
            .map_err(|e| format!("Step failed: {}", e))?;
        
        // Extract observations, reward, done from first agent and update agent IDs
        if let Some(rl_output) = output.rl_output {
            for (behavior_name, agent_info_list) in rl_output.agent_infos.iter() {
                // Update stored agent IDs
                self.last_behavior_name = Some(behavior_name.clone());
                self.last_agent_ids = agent_info_list.value.iter().map(|a| a.id).collect();
                
                if let Some(first_agent) = agent_info_list.value.first() {
                    // Get observations
                    let obs_data: Vec<f32> = if let Some(obs) = first_agent.observations.first() {
                        if let Some(obs_data) = &obs.observation_data {
                            use crate::communicator_objects::observation_proto::ObservationData;
                            match obs_data {
                                ObservationData::FloatData(float_data) => float_data.data.clone(),
                                ObservationData::CompressedData(_) => vec![],
                            }
                        } else {
                            vec![]
                        }
                    } else {
                        vec![]
                    };
                    
                    // Get reward
                    let reward = first_agent.reward;
                    
                    // Get done
                    let done = first_agent.done;
                    
                    return Ok((obs_data, reward, done));
                }
            }
        }
        
        Err("No agent info received from Unity".into())
    }
    
    pub async fn close(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🛑 Closing Unity environment...");
        Ok(())
    }
}

impl Drop for UnityEnvironment {
    fn drop(&mut self) {
        println!("🧹 Cleaning up Unity environment");
    }
}
