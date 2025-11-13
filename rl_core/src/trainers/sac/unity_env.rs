// Unity Environment Integration for SAC
use super::{SACTrainer, Transition, ObservationSpec};
use crate::old::grpc_server::GrpcServer;
use tch::{Tensor, Device};

// Temporary: Define stub types until proto files are properly generated
pub struct UnityRlInputProto {
    pub command: i32,
    pub agent_actions: std::collections::HashMap<i32, AgentActionProto>,
    pub side_channel: Vec<u8>,
}

pub struct UnityOutputProto {
    pub rl_output: Option<RlOutputProto>,
    pub rl_initialization_output: Option<RlInitializationOutputProto>,
}

pub struct RlOutputProto {
    pub agent_infos: std::collections::HashMap<String, AgentInfoListProto>,
}

pub struct RlInitializationOutputProto {
    pub brain_parameters: Vec<BrainParametersProto>,
}

pub struct BrainParametersProto {
    pub brain_name: String,
    pub action_spec: Option<ActionSpecProto>,
}

pub struct ActionSpecProto {
    pub num_continuous_actions: i32,
}

pub struct AgentInfoListProto {
    pub value: Vec<AgentInfoProto>,
}

pub struct AgentInfoProto {
    pub observations: Vec<ObservationProto>,
    pub reward: f32,
    pub done: bool,
}

pub struct ObservationProto {
    pub float_data: FloatDataProto,
}

pub struct FloatDataProto {
    pub compressed_data: Vec<f32>,
}

#[derive(Clone)]
pub struct AgentActionProto {
    pub continuous_actions: Option<ContinuousActionsProto>,
    pub discrete_actions: Option<DiscreteActionsProto>,
}

#[derive(Clone)]
pub struct ContinuousActionsProto {
    pub vector: Vec<f32>,
}

#[derive(Clone)]
pub struct DiscreteActionsProto {
    pub actions: Vec<i32>,
}

pub struct CommandProto;
impl CommandProto {
    pub const Reset: i32 = 0;
    pub const Step: i32 = 1;
}

pub struct InitConfig {
    pub seed: i32,
    pub communication_version: String,
    pub package_version: String,
    pub num_areas: i32,
}

pub struct UnityEnvironment {
    server: GrpcServer,
    behavior_name: String,
    obs_spec: Option<ObservationSpec>,  // Detectado na primeira observação
    action_dim: usize,
    device: Device,
}

impl UnityEnvironment {
    pub async fn new(port: u16, device: Device) -> Result<Self, String> {
        let mut server = GrpcServer::new_with_verbosity(port, true);
        
        // Initialize connection with Unity
        let init_config = InitConfig {
            seed: 42,
            communication_version: "1.5.0".to_string(),
            package_version: "0.30.0".to_string(),
            num_areas: 1,
        };
        
        let init_output = server.initialize(init_config).await?;
        
        // Extract behavior specs from initialization
        let (behavior_name, action_dim) = Self::extract_specs(&init_output)?;
        
        println!("🔗 Unity connected: behavior '{}', {} continuous actions", behavior_name, action_dim);
        println!("🔍 Waiting for first observation to detect sensor configuration...");
        
        Ok(Self {
            server,
            behavior_name,
            obs_spec: None,  // Será detectado na primeira observação
            action_dim,
            device,
        })
    }
    
    fn extract_specs(output: &UnityRlOutputProto) -> Result<(String, usize), String> {
        if let Some(init) = &output.rl_initialization_output {
            if let Some(brain_params) = init.brain_parameters.first() {
                let behavior_name = brain_params.brain_name.clone();
                
                // Get action dimensions
                let action_dim = if let Some(action_spec) = &brain_params.action_spec {
                    action_spec.num_continuous_actions as usize
                } else {
                    return Err("No action spec found".to_string());
                };
                
                Ok((behavior_name, action_dim))
            } else {
                Err("No brain parameters found".to_string())
            }
        } else {
            Err("No initialization output".to_string())
        }
    }
    
    pub async fn reset(&mut self) -> Result<Vec<f32>, String> {
        // Send reset command to Unity
        let reset_input = UnityRlInputProto {
            command: CommandProto::Reset as i32,
            agent_actions: std::collections::HashMap::new(),
            side_channel: vec![],
        };
        
        self.server.send_input(reset_input).await?;
        
        // Get first observation
        let output = self.server.receive_output().await?;
        self.extract_observation(&output)
    }
    
    pub async fn step(&mut self, action: &Tensor) -> Result<(Vec<f32>, f32, bool), String> {
        // Convert tensor action to vec
        let action_vec = Vec::<f32>::try_from(action.shallow_clone())
            .map_err(|e| format!("Failed to convert action: {:?}", e))?;
        
        // Prepare action for Unity
        let agent_action = AgentActionProto {
            continuous_actions: Some(ContinuousActionsProto {
                vector: action_vec,
            }),
            discrete_actions: None,
        };
        
        // Get agent IDs from last output
        let agent_ids = self.get_agent_ids().await?;
        
        let mut agent_actions = std::collections::HashMap::new();
        for agent_id in agent_ids {
            agent_actions.insert(agent_id, agent_action.clone());
        }
        
        // Send step command
        let step_input = UnityRlInputProto {
            command: CommandProto::Step as i32,
            agent_actions,
            side_channel: vec![],
        };
        
        self.server.send_input(step_input).await?;
        
        // Get response
        let output = self.server.receive_output().await?;
        
        let obs = self.extract_observation(&output)?;
        let (reward, done) = self.extract_reward_done(&output)?;
        
        Ok((obs, reward, done))
    }
    
    fn extract_observation(&mut self, output: &UnityOutputProto) -> Result<Vec<f32>, String> {
        if let Some(rl_output) = &output.rl_output {
            if let Some(agent_list) = rl_output.agent_infos.get(&self.behavior_name) {
                if let Some(agent_info) = agent_list.value.first() {
                    // Extrair todas as observações (vector, ray perception, etc)
                    let observations: Vec<Vec<f32>> = agent_info.observations
                        .iter()
                        .map(|obs| obs.float_data.compressed_data.clone())
                        .collect();
                    
                    // Detectar spec na primeira observação
                    if self.obs_spec.is_none() {
                        let spec = ObservationSpec::detect_from_observations(&observations);
                        spec.print_info();
                        self.obs_spec = Some(spec);
                    }
                    
                    // Validar se configuração mudou
                    if let Some(spec) = &self.obs_spec {
                        if !spec.matches(&observations) {
                            println!("⚠️  WARNING: Observation configuration changed!");
                            println!("   Expected: {} dimensions", spec.total_obs_size);
                            println!("   Received: {} dimensions", observations.iter().map(|o| o.len()).sum::<usize>());
                        }
                        
                        // Flatten todas as observações em um único vetor
                        return Ok(spec.flatten_observations(&observations));
                    }
                    
                    // Fallback: se spec ainda não existe, retorna primeira obs
                    if let Some(first_obs) = observations.first() {
                        return Ok(first_obs.clone());
                    }
                }
            }
        }
        Err("No observation found in output".to_string())
    }
    
    fn extract_reward_done(&self, output: &UnityOutputProto) -> Result<(f32, bool), String> {
        if let Some(rl_output) = &output.rl_output {
            if let Some(agent_list) = rl_output.agent_infos.get(&self.behavior_name) {
                if let Some(agent_info) = agent_list.value.first() {
                    let reward = agent_info.reward;
                    let done = agent_info.done;
                    return Ok((reward, done));
                }
            }
        }
        Err("No reward/done found in output".to_string())
    }
    
    async fn get_agent_ids(&self) -> Result<Vec<i32>, String> {
        // This should get agent IDs from the last output
        // For now, return a placeholder
        Ok(vec![0])
    }
    
    pub fn get_obs_dim(&self) -> usize {
        self.obs_spec.as_ref().map(|s| s.total_obs_size).unwrap_or(0)
    }
    
    pub fn get_action_dim(&self) -> usize {
        self.action_dim
    }
    
    pub fn get_obs_spec(&self) -> Option<&ObservationSpec> {
        self.obs_spec.as_ref()
    }
    
    pub fn has_ray_perception(&self) -> bool {
        self.obs_spec.as_ref().map(|s| s.has_ray_perception).unwrap_or(false)
    }
}

// Training loop that integrates SAC with Unity
pub struct UnityTrainer {
    env: UnityEnvironment,
    sac: SACTrainer,
    max_steps: usize,
    log_interval: usize,
}

impl UnityTrainer {
    pub async fn new(
        port: u16,
        sac_trainer: SACTrainer,
        max_steps: usize,
    ) -> Result<Self, String> {
        let device = Device::cuda_if_available();
        let env = UnityEnvironment::new(port, device).await?;
        
        Ok(Self {
            env,
            sac: sac_trainer,
            max_steps,
            log_interval: 100,
        })
    }
    
    pub async fn train(&mut self) -> Result<(), String> {
        println!("🚀 Starting SAC training with Unity environment");
        
        // Reset to detect observation spec
        let _ = self.env.reset().await?;
        
        let obs_dim = self.env.get_obs_dim();
        let action_dim = self.env.get_action_dim();
        
        println!("\n🤖 Training Configuration:");
        println!("   └─ Total observation size: {} dimensions", obs_dim);
        println!("   └─ Action size: {} dimensions", action_dim);
        
        if let Some(spec) = self.env.get_obs_spec() {
            if spec.has_ray_perception {
                println!("   └─ ✅ RayPerception sensors detected: {} sensor(s)", spec.ray_perception_specs.len());
            } else {
                println!("   └─ Vector observations only (no RayPerception)");
            }
        }
        
        // Validar dimensões do modelo
        if self.sac.obs_dim as usize != obs_dim {
            return Err(format!(
                "Model observation size mismatch! Model: {}, Unity: {}",
                self.sac.obs_dim, obs_dim
            ));
        }
        
        println!();
        
        let mut episode = 0;
        let mut step_count = 0;
        
        while step_count < self.max_steps {
            // Reset environment
            let obs = self.env.reset().await?;
            let mut episode_reward = 0.0;
            let mut episode_steps = 0;
            
            let mut current_obs = obs;
            
            loop {
                // Get observation tensor
                let obs_tensor = Tensor::from_slice(&current_obs)
                    .to_device(self.sac.device)
                    .unsqueeze(0); // Add batch dimension
                
                // Select action
                let action = self.sac.select_action(&obs_tensor, false);
                
                // Step environment
                let (next_obs, reward, done) = self.env.step(&action).await?;
                
                // Store transition
                let transition = Transition {
                    obs: current_obs.clone(),
                    action: Vec::<f32>::try_from(action.squeeze_dim(0))
                        .map_err(|e| format!("Action conversion failed: {:?}", e))?,
                    reward,
                    next_obs: next_obs.clone(),
                    done,
                };
                self.sac.store_transition(transition);
                
                // Update SAC
                if let Some(metrics) = self.sac.update() {
                    if step_count % self.log_interval == 0 {
                        println!(
                            "Step {}: Actor Loss: {:.4}, Critic Loss: {:.4}, Alpha: {:.4}, Reward: {:.2}",
                            step_count,
                            metrics.actor_loss,
                            metrics.critic_loss,
                            metrics.alpha,
                            episode_reward
                        );
                    }
                }
                
                episode_reward += reward;
                episode_steps += 1;
                step_count += 1;
                
                if done || step_count >= self.max_steps {
                    break;
                }
                
                current_obs = next_obs;
            }
            
            episode += 1;
            println!(
                "✓ Episode {} finished: {} steps, reward: {:.2}",
                episode, episode_steps, episode_reward
            );
            
            // Save checkpoint periodically based on config.checkpoint_interval
            if self.sac.should_checkpoint() {
                let checkpoint_path = format!("checkpoints/sac_step_{}", self.sac.step);
                self.sac.save_checkpoint(&checkpoint_path)
                    .map_err(|e| format!("Failed to save checkpoint: {}", e))?;
                println!("✓ Checkpoint saved at step {}", self.sac.step);
                
                // Export ONNX if enabled in config
                if self.sac.config.save_onnx {
                    self.sac.export_onnx(&checkpoint_path)
                        .map_err(|e| format!("Failed to export ONNX: {}", e))?;
                    println!("✓ ONNX exported at step {}", self.sac.step);
                }
            }
        }
        
        println!("🎉 Training completed! Total steps: {}", step_count);
        
        // Save final model
        self.sac.save_checkpoint("checkpoints/sac_final")
            .map_err(|e| format!("Failed to save final checkpoint: {}", e))?;
        println!("✓ Final checkpoint saved: checkpoints/sac_final.pt");
        
        // Export ONNX if enabled in config
        if self.sac.config.save_onnx {
            self.sac.export_onnx("checkpoints/sac_final")
                .map_err(|e| format!("Failed to export final ONNX: {}", e))?;
            println!("✓ Final ONNX exported: checkpoints/sac_final.onnx");
        }
        
        Ok(())
    }
}
