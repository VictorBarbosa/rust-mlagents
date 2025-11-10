use crate::networks::{Actor, Critic};
use crate::settings::RootConfig;
use crate::communicator_objects::UnityMessageProto;
use crate::logging::MetricsLogger;
use burn::tensor::{backend::Backend, Tensor};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPOTrainerConfig {
    pub input_size: usize,
    pub action_size: usize,
    pub hidden_units: usize,
    pub num_layers: usize,
    pub max_steps: u64,
    pub checkpoint_interval: u64,
    pub num_envs: usize,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub export_onnx_every_checkpoint: bool,
}

use burn::record::Recorder;
use crate::communicator_objects::unity_to_external_proto_client;

pub struct PPOTrainer<B: Backend> {
    actor: Actor<B>,
    critic: Critic<B>,
    device: B::Device,
    cfg: PPOTrainerConfig,
    clients: Vec<unity_to_external_proto_client::UnityToExternalProtoClient<tonic::transport::Channel>>,
    addresses: Vec<std::net::SocketAddr>,
    logger: MetricsLogger,
}

impl<B: Backend> PPOTrainer<B> {
    pub fn new(device: &B::Device, cfg: PPOTrainerConfig, addresses: Vec<std::net::SocketAddr>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let actor = Actor::new(
            cfg.input_size,
            cfg.hidden_units,
            cfg.num_layers,
            cfg.action_size,
            device,
        );
        let critic = Critic::new(cfg.input_size, cfg.hidden_units, cfg.num_layers, device);

        let rt = tokio::runtime::Runtime::new()?;
        let clients: Vec<_> = rt.block_on(async {
            let mut client_list = Vec::new();
            for addr in &addresses {
                let client = crate::grpc::connect_to_unity(*addr).await?;
                client_list.push(client);
            }
            Ok::<Vec<_>, Box<dyn std::error::Error + Send + Sync>>(client_list)
        })?;

        // Create metrics logger
        let logger = MetricsLogger::new("logs");

        Ok(Self {
            actor,
            critic,
            device: device.clone(),
            cfg,
            clients,
            addresses,
            logger,
        })
    }

    pub fn train(&mut self) {
        let initial_actions: Vec<Vec<f32>> = vec![vec![0.0; self.cfg.action_size]; self.cfg.num_envs];
        let _current_actions = initial_actions;

        let mut buf = super::buffer::RolloutBuffer::new();
        // Estado para cada ambiente
        let mut current_obs_per_env = vec![Vec::new(); self.cfg.num_envs];
        let mut current_reward_per_env = vec![0.0f32; self.cfg.num_envs];
        let mut current_done_per_env = vec![false; self.cfg.num_envs];

        let rt = tokio::runtime::Runtime::new().unwrap(); // Runtime para as chamadas assíncronas

        for step in 1..=self.cfg.max_steps {
            let next_actions: Vec<Vec<f32>> = (0..self.cfg.num_envs).map(|i| {
                let obs = &current_obs_per_env[i];
                let obs_len = obs.len().max(self.cfg.input_size.max(1));
                let mut obs_vec = obs.clone();
                if obs_vec.len() < obs_len { obs_vec.resize(obs_len, 0.0); }
                let obs_t = Tensor::<B,2>::from_floats(obs_vec.as_slice(), &self.device).reshape([1, obs_len]);

                let act_t = self.actor.forward(obs_t.clone());
                let data = act_t.into_data();
                let bytes = data.bytes;
                let floats: &[f32] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / std::mem::size_of::<f32>()) };
                let mut action = Vec::new();
                action.extend_from_slice(&floats[0..self.cfg.action_size.min(floats.len())]);
                action
            }).collect();

            let mut new_obs_per_env = vec![None; self.cfg.num_envs];
            let mut new_rewards = vec![0.0f32; self.cfg.num_envs];
            let mut new_done = vec![false; self.cfg.num_envs];
            let mut transitions_to_store = vec![];

            for (i, client) in self.clients.iter_mut().enumerate() {
                let action_to_send = next_actions[i].clone();
                use crate::communicator_objects::{UnityInputProto, UnityRlInputProto, unity_rl_input_proto::ListAgentActionProto, AgentActionProto, CommandProto};
                let agent_action_proto = AgentActionProto {
                    vector_actions_deprecated: vec![],
                    value: 0.0,
                    continuous_actions: action_to_send,
                    discrete_actions: vec![],
                };
                let mut action_map = HashMap::new();
                action_map.insert("Behavior".to_string(), ListAgentActionProto { value: vec![agent_action_proto] });

                let rl_input = UnityRlInputProto {
                    agent_actions: action_map,
                    command: CommandProto::Step as i32,
                    side_channel: vec![],
                };
                let input_msg = UnityMessageProto {
                    header: None,
                    unity_output: None,
                    unity_input: Some(UnityInputProto { rl_input: Some(rl_input), rl_initialization_input: None }),
                };

                let response_msg = rt.block_on(async {
                    crate::grpc::exchange_with_unity(client, input_msg).await
                }).unwrap_or_else(|e| {
                    eprintln!("[error] gRPC exchange failed with environment {}: {}", i, e);
                    UnityMessageProto { header: None, unity_output: None, unity_input: None }
                });

                if let (Some((rew, done)), Some(obs)) = (crate::grpc::parse_step_reward_done(&response_msg), crate::grpc::parse_step_observation_flat(&response_msg)) {
                    new_obs_per_env[i] = Some(obs.clone());
                    new_rewards[i] = rew;
                    new_done[i] = done;
                    transitions_to_store.push((rew, 0.0, done));
                }
            }

            for (i, new_obs) in new_obs_per_env.into_iter().enumerate() {
                if let Some(obs) = new_obs {
                    current_obs_per_env[i] = obs;
                    current_reward_per_env[i] = new_rewards[i];
                    current_done_per_env[i] = new_done[i];
                }
            }

            for (rew, val, done) in transitions_to_store {
                buf.push(vec![], vec![], rew, val, done, 0.0);
            }

            if step % self.cfg.checkpoint_interval == 0 {
                buf.finish_path(self.cfg.gamma, self.cfg.gae_lambda, 0.0);
                buf.clear();
                
                // Save the full model checkpoint with weights
                if let Err(e) = self.save_model_checkpoint(step) {
                    eprintln!("[warn] Failed to save model checkpoint at step {}: {}", step, e);
                }
                
                if self.cfg.export_onnx_every_checkpoint { let _ = self.export_onnx(step); }
            }
        }
    }

    fn checkpoints_dir() -> PathBuf { PathBuf::from("checkpoints") }

    fn save_checkpoint(&self, step: u64) -> std::io::Result<PathBuf> {
        // This is a legacy function, using the new model checkpoint function instead
        // The full model checkpoint is now handled separately in save_model_checkpoint
        let dir = Self::checkpoints_dir();
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("metadata_step_{step}.json"));
        let meta = serde_json::json!({
            "step": step,
            "input_size": self.cfg.input_size,
            "hidden_units": self.cfg.hidden_units,
            "num_layers": self.cfg.num_layers,
            "action_size": self.cfg.action_size,
            "max_steps": self.cfg.max_steps,
            "checkpoint_interval": self.cfg.checkpoint_interval,
            "num_envs": self.cfg.num_envs,
            "gamma": self.cfg.gamma,
            "gae_lambda": self.cfg.gae_lambda,
        });
        fs::write(&path, serde_json::to_string_pretty(&meta).unwrap())?;
        Ok(path)
    }
    
    /// Save complete model checkpoint with network weights
    pub fn save_model_checkpoint(&self, step: u64) -> Result<(), Box<dyn std::error::Error>> {
        let checkpoint_dir = format!("checkpoints/model_step_{}", step);
        std::fs::create_dir_all(&checkpoint_dir)?;
        
        // Save actor model
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, PrettyJsonFileRecorder, Recorder};
        
        let actor_path = std::path::Path::new(&checkpoint_dir).join("actor");
        let actor_record = self.actor.clone().into_record();
        PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
            .record(actor_record, actor_path)?;
        
        // Save critic model  
        let critic_path = std::path::Path::new(&checkpoint_dir).join("critic");
        let critic_record = self.critic.clone().into_record();
        PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
            .record(critic_record, critic_path)?;
        
        // Save trainer configuration
        let config_path = std::path::Path::new(&checkpoint_dir).join("config.json");
        let config = serde_json::json!({
            "cfg": self.cfg,
            "step": step,
        });
        
        std::fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;
        
        Ok(())
    }
    
    /// Load the latest checkpoint
    pub fn load_latest_checkpoint(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        // Look for the latest checkpoint directory
        let checkpoints_dir = std::path::Path::new("checkpoints");
        if !checkpoints_dir.exists() {
            return Err("No checkpoints directory found".into());
        }
        
        let mut checkpoints = Vec::new();
        for entry in std::fs::read_dir(checkpoints_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name.starts_with("model_step_") {
                    let step_str = name.trim_start_matches("model_step_");
                    if let Ok(step) = step_str.parse::<u64>() {
                        checkpoints.push((step, path));
                    }
                }
            }
        }
        
        if checkpoints.is_empty() {
            return Err("No checkpoint directories found".into());
        }
        
        // Get the checkpoint with the highest step number
        checkpoints.sort_by_key(|(step, _)| *step);
        let (_, latest_path) = checkpoints.last().unwrap();
        
        // Load actor model
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, PrettyJsonFileRecorder, Recorder};
        
        let actor_path = latest_path.join("actor");
        if actor_path.exists() {
            let record = PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
                .load(actor_path, &self.device)?;
            // Create a fresh actor and load the record into it
            let fresh_actor = Actor::new(
                self.cfg.input_size,
                self.cfg.hidden_units,
                self.cfg.num_layers,
                self.cfg.action_size,
                &self.device
            );
            self.actor = fresh_actor.load_record(record);
        }
        
        // Load critic model
        let critic_path = latest_path.join("critic");
        if critic_path.exists() {
            let record = PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
                .load(critic_path, &self.device)?;
            // Create a fresh critic and load the record into it
            let fresh_critic = Critic::new(
                self.cfg.input_size,
                self.cfg.hidden_units,
                self.cfg.num_layers,
                &self.device
            );
            self.critic = fresh_critic.load_record(record);
        }
        
        // Load configuration and extract step
        let config_path = latest_path.join("config.json");
        let mut step = 0u64;
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(s) = config["step"].as_u64() {
                step = s;
            }
        }
        
        Ok(step)
    }
    
    fn export_onnx(&self, step: u64) -> std::io::Result<()> {
        let script_path = PathBuf::from("rl_core").join("export_onnx.py");
        if !script_path.exists() { return Ok(()); }
        let out_dir = Self::checkpoints_dir();
        fs::create_dir_all(&out_dir)?;
        let onnx_out = out_dir.join(format!("step_{step}.onnx"));
        let status = Command::new("python3").arg(script_path).arg("--out").arg(&onnx_out).status();
        match status {
            Ok(s) if s.success() => Ok(()),
            _ => Ok(()),
        }
    }
}

pub fn run_from_config<B: Backend>(device: &B::Device, root: &RootConfig, behavior: &str) {
    if let Some(bcfg) = root.behaviors.get(behavior) {
        let hidden = bcfg.network_settings.as_ref().and_then(|n| n.hidden_units).unwrap_or(128);
        let layers = bcfg.network_settings.as_ref().and_then(|n| n.num_layers).unwrap_or(2);
        let num_envs = root.env_settings.as_ref().and_then(|e| e.num_envs).unwrap_or(1);
        let max_steps = bcfg.max_steps;
        let ckpt = bcfg.checkpoint_interval as u64;

        if let Some(es) = &root.env_settings {
            let cfg = crate::grpc::InitConfig {
                seed: es.seed.unwrap_or(-1),
                num_areas: es.num_areas.unwrap_or(1),
                communication_version: "1.5.0".to_string(),
                package_version: "0.1.0".to_string(),
            };
            crate::grpc::set_init_config(cfg);
        }

        let env_params = root.env_settings.as_ref().and_then(|es| es.environment_parameters.clone()).unwrap_or_default();

        let rt_init = tokio::runtime::Runtime::new().unwrap();
        let mut specs_per_env = HashMap::new();
        let base_port = root.env_settings.as_ref().and_then(|e| e.base_port).unwrap_or(5005);
        let mut addrs = Vec::new();
        for i in 0..num_envs {
            addrs.push(std::net::SocketAddr::from(([127,0,0,1], base_port + i as u16)));
        }

        let timeout_secs = 30;
        let start_time = std::time::Instant::now();
        loop {
            let mut all_connected = true;
            for addr in &addrs {
                if !specs_per_env.contains_key(addr) {
                    match rt_init.block_on(async { crate::grpc::connect_to_unity(*addr).await }) {
                        Ok(mut client) => {
                            match rt_init.block_on(async { crate::grpc::initialize_unity_with_params(&mut client, env_params.clone()).await }) {
                                Ok(specs) => { specs_per_env.insert(*addr, specs); },
                                Err(e) => { eprintln!("[debug] Initialization with {} failed: {}, retrying...", addr, e); all_connected = false; }
                            }
                        },
                        Err(e) => { eprintln!("[debug] Connection to {} failed: {}, retrying...", addr, e); all_connected = false; }
                    }
                }
            }
            if all_connected { break; }
            if start_time.elapsed().as_secs() > timeout_secs {
                eprintln!("[error] Timeout waiting for Unity initialization on all ports: {:?}", addrs);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }

        if specs_per_env.is_empty() {
            eprintln!("[error] No successful Unity connections after initialization.");
            return;
        }

        let first_specs = specs_per_env.values().next().unwrap();
        let obs_sum: usize = first_specs.observation_sizes.iter().copied().sum();
        let input_size = obs_sum.max(1);
        let action_size = first_specs.action_size.max(1);

        let gamma = bcfg.reward_signals.as_ref().and_then(|m| m.get("extrinsic")).and_then(|e| e.gamma).unwrap_or(0.99);
        let gae_lambda = bcfg.hyperparameters.lambd;

        let mut trainer: PPOTrainer<B> = match PPOTrainer::new(
            device,
            PPOTrainerConfig {
                input_size,
                action_size,
                hidden_units: hidden,
                num_layers: layers,
                max_steps,
                checkpoint_interval: ckpt,
                num_envs,
                gamma,
                gae_lambda,
                export_onnx_every_checkpoint: true,
            },
            addrs,
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[error] Failed to create PPOTrainer: {}", e);
                return;
            }
        };


        // Try to resume from latest checkpoint if resume flag was specified globally
        // Note: We'd need access to global args to check if resume is enabled
        // For now, we'll add a basic checkpoint loading before training
        match trainer.load_latest_checkpoint() {
            Ok(step) => {
                println!("🔄 Successfully resumed from checkpoint at step {}", step);
                // TODO: In a complete implementation, we would adjust training steps to continue from where we left off
            },
            Err(e) => {
                println!("⚠️  Failed to load checkpoint (starting fresh): {}", e);
            }
        }

        trainer.train();
    }
    
}