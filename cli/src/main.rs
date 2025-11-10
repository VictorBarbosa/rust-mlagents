use clap::Parser;
use rl_core::communicator_objects::{UnityRlInitializationOutputProto, UnityInputProto, UnityRlInputProto, CommandProto};
use std::collections::HashMap;

// Single-level CLI matching original mlagents-learn style
#[derive(Parser, Debug)]
#[command(name = "rust-mlagents-learn", version, about = "Rust reimplementation of mlagents-learn", disable_help_subcommand = true)]
struct Args {
    /// Caminho do arquivo YAML de configuração dos treinadores (trainer_config_path)
    #[arg(value_name = "trainer_config_path", num_args=0..)]
    trainer_config_path: Vec<String>,
    /// Caminho para o executável Unity para treinamento
    #[arg(long = "env", help = "Path to the Unity executable to train", alias = "env-path")]
    env_path: Option<String>,
    /// Retomar treinamento de um checkpoint existente (requer run-id)
    #[arg(long, help = "Resume training from an existing checkpoint for this run-id")]
    resume: bool,
    /// Seleciona ações determinísticas (mean / argmax)
    #[arg(long, help = "Deterministic action selection instead of sampling")]
    deterministic: bool,
    /// Força sobrescrita de artefatos existentes do run-id
    #[arg(long, help = "Force overwrite existing run artifacts for the run-id")]
    force: bool,
    /// Identificador único do treinamento (subpastas de resultados e modelos)
    #[arg(long = "run-id", default_value = "ppo", help = "Unique identifier for the training run")]
    run_id: String,
    /// Inicializa pesos a partir de outro run-id salvo
    #[arg(long = "initialize-from", help = "Initialize model weights from previous run-id")]
    initialize_from: Option<String>,
    /// Semente para gerador de números aleatórios (-1 para aleatório)
    #[arg(long, default_value_t = -1, help = "Random seed for reproducibility (-1 for none)")]
    seed: i64,
    /// Executa apenas inferência (sem atualizar pesos)
    #[arg(long, help = "Run in inference-only mode (no training updates)")]
    inference: bool,
    /// Porta base para comunicação gRPC com instâncias Unity
    #[arg(long = "base-port", default_value_t = 5005, help = "Starting port for environment instances (Unity default 5005)")]
    base_port: u16,
    /// Número de instâncias Unity paralelas
    #[arg(long = "num-envs", default_value_t = 1, help = "Number of concurrent Unity environment processes")]
    num_envs: usize,
    /// Número de áreas paralelas internas por instância
    #[arg(long = "num-areas", default_value_t = 1, help = "Parallel training areas per Unity instance")]
    num_areas: usize,
    /// Ativa logs de debug adicionais
    #[arg(long, help = "Enable debug-level logging")]
    debug: bool,
    /// Argumentos extras repassados ao executável Unity
    #[arg(long = "env-args", num_args = 1.., help = "Extra arguments forwarded to Unity executable")]
    env_args: Vec<String>,
    /// Máximo de reinícios permitidos por processo durante a vida
    #[arg(long = "max-lifetime-restarts", default_value_t = 10, help = "Max crashes per executable lifetime (-1 unlimited)")]
    max_lifetime_restarts: i32,
    /// Limite de reinícios dentro da janela de tempo
    #[arg(long = "restarts-rate-limit-n", default_value_t = 1, help = "Max restarts allowed in period (-1 disable)")]
    restarts_rate_limit_n: i32,
    /// Janela (segundos) para aplicar limite de reinícios
    #[arg(long = "restarts-rate-limit-period-s", default_value_t = 60, help = "Time period (s) for restart rate limiting")]
    restarts_rate_limit_period_s: i32,
    /// Carrega modelo existente (legado / deprecated)
    #[arg(long = "load", help = "Load existing model (deprecated)")]
    load: bool,
    /// Diretório base de resultados
    #[arg(long = "results-dir", default_value = "results", help = "Base results directory")]
    results_dir: String,
    /// Tempo máximo (s) de espera para cada instância iniciar
    #[arg(long = "timeout-wait", default_value_t = 60, help = "Timeout (s) waiting Unity to start (default matches original 60s)")]
    timeout_wait: u64,
    /// Largura da janela do executável (px)
    #[arg(long, default_value_t = 84, help = "Window width in pixels")]
    width: u32,
    /// Altura da janela do executável (px)
    #[arg(long, default_value_t = 84, help = "Window height in pixels")]
    height: u32,
    /// Nível de qualidade gráfico Unity
    #[arg(long = "quality-level", default_value_t = 5, help = "Unity QualitySettings level")]
    quality_level: i32,
    /// Time scale do motor Unity
    #[arg(long = "time-scale", default_value_t = 20.0, help = "Unity Time.timeScale value")]
    time_scale: f32,
    /// Frame rate alvo (-1 ilimitado)
    #[arg(long = "target-frame-rate", default_value_t = -1, help = "Application.targetFrameRate (-1 unlimited)")]
    target_frame_rate: i32,
    /// Frame rate de captura (Time.captureFramerate)
    #[arg(long = "capture-frame-rate", default_value_t = 60, help = "Time.captureFramerate value")]
    capture_frame_rate: i32,
    /// Executa sem inicializar driver gráfico
    #[arg(long = "no-graphics", help = "Run Unity with graphics disabled")]
    no_graphics: bool,
    /// Apenas processo principal com gráficos, workers sem
    #[arg(long = "no-graphics-monitor", help = "Main worker graphics on, others off")]
    no_graphics_monitor: bool,
    /// Dispositivo Torch (cpu, cuda, cuda:0)
    #[arg(long = "torch-device", help = "Torch device string (cpu / cuda / cuda:0 / MPS(only macOS apple silicon))")]
    device: Option<String>,
    /// Flag placeholder Torch removido no original
    #[arg(long, help = "(Removed) Torch framework flag placeholder")]
    torch: bool,
    /// Flag placeholder TensorFlow removido no original
    #[arg(long, help = "(Removed) TensorFlow framework flag placeholder")]
    tensorflow: bool,
    /// Abre a interface gráfica (ações em breve)
    #[arg(long, help = "Launch GUI (actions TBD)")]
    gui: bool,
}

fn pick_behavior(root: &rl_core::settings::RootConfig) -> Option<String> {
    let mut keys: Vec<_> = root.behaviors.keys().cloned().collect();
    keys.sort();
    keys.into_iter().next()
}

async fn run_training_loop(
    mut server: rl_core::grpc_server::GrpcServer,
    specs: UnityRlInitializationOutputProto,
    config: &rl_core::settings::RootConfig,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    use burn::backend::NdArray;
    use burn::tensor::Tensor;
    use rl_core::ppo_trainer::PPOTrainer;
    use rl_core::ppo_buffer::RolloutBuffer;
    
    // Extract specs from first brain
    let brain = specs.brain_parameters.first()
        .ok_or("No brain parameters")?;
    
    let action_spec = brain.action_spec.as_ref()
        .ok_or("No action spec")?;
    
    let num_continuous = action_spec.num_continuous_actions as usize;
    let num_discrete_branches = action_spec.discrete_branch_sizes.len();
    
    if args.debug {
        println!("\n📊 Configuração do agente:");
        println!("  - Behavior: {}", brain.brain_name);
        println!("  - Ações contínuas: {}", num_continuous);
        println!("  - Branches discretas: {}", num_discrete_branches);
        println!("  - Training areas: {}", args.num_areas);
    }
    
    // Get behavior config
    let behavior_name = brain.brain_name.split('?').next().unwrap_or(&brain.brain_name);
    let behavior_cfg = config.behaviors.get(behavior_name)
        .ok_or(format!("Behavior '{}' não encontrado no config", behavior_name))?;
    
    // Prepare environment parameters and engine config to send via side channel
    let env_params = config.environment_parameters.clone()
        .or_else(|| config.env_settings.as_ref()
            .and_then(|es| es.environment_parameters.clone()))
        .unwrap_or_default();
    
    // Build engine configuration from YAML + CLI args
    let engine_config = rl_core::side_channel::EngineConfig {
        width: config.engine_settings.as_ref().map(|e| e.width).unwrap_or(args.width),
        height: config.engine_settings.as_ref().map(|e| e.height).unwrap_or(args.height),
        quality_level: config.engine_settings.as_ref().map(|e| e.quality_level).unwrap_or(args.quality_level),
        time_scale: config.engine_settings.as_ref().map(|e| e.time_scale).unwrap_or(args.time_scale),
        target_frame_rate: config.engine_settings.as_ref().map(|e| e.target_frame_rate).unwrap_or(args.target_frame_rate),
        capture_frame_rate: config.engine_settings.as_ref().map(|e| e.capture_frame_rate).unwrap_or(args.capture_frame_rate),
    };
    
    // Combine side channels (always send engine config, optionally send env params)
    let mut side_channels = vec![rl_core::side_channel::serialize_engine_config(&engine_config)];
    
    if !env_params.is_empty() {
        if args.debug {
            println!("\n📤 Environment Parameters:");
            for (key, value) in &env_params {
                println!("  - {}: {:?}", key, value);
            }
        }
        side_channels.push(rl_core::side_channel::serialize_environment_parameters(&env_params));
    }
    
    if args.debug {
        println!("\n⚙️  Engine Settings:");
        println!("  - Resolution: {}x{}", engine_config.width, engine_config.height);
        println!("  - Quality Level: {}", engine_config.quality_level);
        println!("  - Time Scale: {}x", engine_config.time_scale);
        println!("  - Target FPS: {}", if engine_config.target_frame_rate < 0 { "Unlimited".to_string() } else { engine_config.target_frame_rate.to_string() });
        println!("  - Capture FPS: {}", engine_config.capture_frame_rate);
    }
    
    // Reset environment with all side channel data
    let combined_side_channel = rl_core::side_channel::combine_side_channels(&side_channels);
    let reset_output = server.reset_with_side_channel(combined_side_channel).await?;
    
    // Parse observations from reset
    let rl_output = reset_output.rl_output.as_ref()
        .ok_or("No RL output from reset")?;
    
    if args.debug {
        println!("✓ Ambiente resetado!");
        println!("  - Behaviors ativos: {}", rl_output.agent_infos.len());
    }
    
    // Determine observation size from first agent
    let obs_size = if let Some((_, agent_list)) = rl_output.agent_infos.iter().next() {
        if let Some(agent_info) = agent_list.value.first() {
            let obs_len: usize = agent_info.observations.iter()
                .map(|obs| obs.shape.iter().product::<i32>() as usize)
                .sum();
            if args.debug {
                println!("  - Observação size: {}", obs_len);
            }
            obs_len
        } else {
            return Err("Agent list is empty".into());
        }
    } else {
        return Err("No agent info in reset output".into());
    };
    
    // Initialize networks with correct sizes
    let device = Default::default();
    let hidden_units = behavior_cfg.network_settings.as_ref()
        .and_then(|n| n.hidden_units)
        .unwrap_or(128);
    let num_layers = behavior_cfg.network_settings.as_ref()
        .and_then(|n| n.num_layers)
        .unwrap_or(2);
    
    let learning_rate = if behavior_cfg.hyperparameters.learning_rate > 0.0 {
        behavior_cfg.hyperparameters.learning_rate as f64
    } else {
        0.0003
    };
    
    let mut trainer = PPOTrainer::<NdArray>::new(
        obs_size,
        num_continuous,
        hidden_units,
        num_layers,
        learning_rate,
        &device,
    );
    
    // Configure PPO hyperparameters from config
    let hp = &behavior_cfg.hyperparameters;
    
    // Set all hyperparameters from YAML
    if hp.epsilon > 0.0 {
        trainer.clip_epsilon = hp.epsilon;
    }
    if hp.num_epoch > 0 {
        trainer.num_epochs = hp.num_epoch as usize;
    }
    if hp.lambd > 0.0 {
        trainer.gae_lambda = hp.lambd;
    }
    if hp.beta > 0.0 {
        trainer.entropy_coef = hp.beta;
    }
    
    trainer.batch_size = hp.batch_size;
    trainer.buffer_size = hp.buffer_size;
    
    if let Some(rs) = &behavior_cfg.reward_signals {
        if let Some(extrinsic) = rs.get("extrinsic") {
            trainer.gamma = extrinsic.gamma.unwrap_or(0.99);
        }
    }
    
    println!("🧠 PPO Trainer configurado (lr={}, hidden={}, layers={})", 
             learning_rate, hidden_units, num_layers);
    
    if args.debug {
        println!("  📋 Hyperparameters:");
        println!("     • Clip epsilon: {}", trainer.clip_epsilon);
        println!("     • Num epochs: {}", trainer.num_epochs);
        println!("     • Gamma: {}", trainer.gamma);
        println!("     • Lambda (GAE): {}", trainer.gae_lambda);
        println!("     • Beta (entropy): {}", trainer.entropy_coef);
        println!("     • Batch size: {}", trainer.batch_size);
        println!("     • Buffer size: {}", trainer.buffer_size);
    }
    
    // Training loop
    let max_steps = if behavior_cfg.max_steps > 0 {
        behavior_cfg.max_steps as usize
    } else {
        50000
    };
    
    let time_horizon = if behavior_cfg.time_horizon > 0 {
        behavior_cfg.time_horizon as usize
    } else {
        64
    };
    
    let batch_size = hp.batch_size;
    let buffer_size = hp.buffer_size;
    let summary_freq = behavior_cfg.summary_freq as usize;
    let checkpoint_interval = behavior_cfg.checkpoint_interval as usize;
    
    println!("🎯 Treinamento: max_steps={}, horizon={}, summary_freq={}", 
             max_steps, time_horizon, summary_freq);
    
    if args.debug {
        println!("  - Batch size: {}", batch_size);
        println!("  - Buffer size: {}", buffer_size);
        println!("  - Checkpoint interval: {}", checkpoint_interval);
    }
    
    // Check for init_path to load existing model
    if let Some(ref init_path) = behavior_cfg.init_path {
        println!("  - Init from: {}", init_path);
        // TODO: Load model weights from checkpoint
    }
    
    println!();
    
    let mut buffer = RolloutBuffer::new();
    let mut current_output = reset_output;
    let mut episode_rewards = Vec::new();
    let mut current_episode_reward = 0.0;
    let mut num_updates = 0;
    let mut total_reward_sum = 0.0;
    let mut total_episodes = 0;
    
    for step in 1..=max_steps {
        // Parse observations from all agents
        let rl_output = current_output.rl_output.as_ref().unwrap();
        
        // Generate actions for each agent
        let mut agent_actions = HashMap::new();
        for (behavior_name, agent_list) in &rl_output.agent_infos {
            let mut actions_for_behavior = Vec::new();
            
            for agent_info in &agent_list.value {
                // Flatten observations
                let obs: Vec<f32> = agent_info.observations.iter()
                    .flat_map(|obs| {
                        if let Some(ref data) = obs.observation_data {
                            use rl_core::communicator_objects::observation_proto::ObservationData;
                            match data {
                                ObservationData::FloatData(fd) => fd.data.clone(),
                                ObservationData::CompressedData(_) => vec![], // TODO: decompress
                            }
                        } else {
                            vec![]
                        }
                    })
                    .collect();
                
                // Generate action with policy and get value estimate
                let obs_tensor = Tensor::<NdArray, 1>::from_floats(obs.as_slice(), &device)
                    .reshape([1, obs.len()]);
                
                let (action_tensor, value_tensor, _log_prob) = trainer.get_action_and_value(obs_tensor);
                let action_data = action_tensor.to_data();
                let actions: Vec<f32> = action_data.to_vec().unwrap();
                
                let value_data = value_tensor.to_data();
                let _value: f32 = value_data.to_vec().unwrap()[0];
                
                // Create action proto (new format with continuous_actions)
                let action_proto = rl_core::communicator_objects::AgentActionProto {
                    vector_actions_deprecated: vec![],
                    value: 0.0,
                    continuous_actions: actions,
                    discrete_actions: vec![],
                };
                
                actions_for_behavior.push(action_proto);
            }
            
            agent_actions.insert(
                behavior_name.clone(),
                rl_core::communicator_objects::unity_rl_input_proto::ListAgentActionProto {
                    value: actions_for_behavior,
                }
            );
        }
        
        // Send actions to Unity
        let step_input = UnityInputProto {
            rl_input: Some(UnityRlInputProto {
                agent_actions,
                command: CommandProto::Step as i32,
                side_channel: vec![], // Parameters sent during initialization
            }),
            rl_initialization_input: None,
        };
        
        current_output = server.step(step_input).await?;
        let step_rl_output = current_output.rl_output.as_ref()
            .ok_or("No RL output from step")?;
        
        // Collect rewards and store transitions
        let mut step_reward = 0.0;
        let mut step_done = false;
        
        // We need to extract the observations and actions we just computed
        // For now, extract from the first agent in the first behavior
        if let Some((_behavior_name, agent_list)) = step_rl_output.agent_infos.iter().next() {
            if let Some(agent_info) = agent_list.value.first() {
                step_reward = agent_info.reward;
                step_done = agent_info.done;
                
                // Extract observations from the previous step (we need to track these)
                // For now, skip buffer storage to avoid dimension mismatch
                // The PPO update is disabled anyway until we properly track obs/actions per step
                
                // Note: To properly implement this, we'd need to:
                // 1. Store (obs, action, value, log_prob) when generating actions
                // 2. Then add (reward, done) when receiving the response
                // 3. This requires refactoring the loop structure
            }
        }
        
        current_episode_reward += step_reward;
        
        // Log progress (respecting summary_freq from config)
        if step % summary_freq == 0 || step == max_steps {
            let avg_reward = if total_episodes > 0 {
                total_reward_sum / total_episodes as f32
            } else {
                current_episode_reward
            };
            println!("[Step {}/{: >8}] Reward: {:.3} | Avg: {:.2} | Episodes: {}", 
                     step, max_steps, step_reward, avg_reward, total_episodes);
        }
        
        // Update policy when buffer is full (using time_horizon from config)
        if buffer.len() >= time_horizon {
            let (policy_loss, value_loss, entropy) = trainer.update(&mut buffer);
            num_updates += 1;
            
            if args.debug || step % summary_freq == 0 {
                println!("  🔄 Update #{}: policy={:.4}, value={:.4}, entropy={:.4}", 
                         num_updates, policy_loss, value_loss, entropy);
            }
            
            buffer.clear();
        }
        
        // Save checkpoint (using checkpoint_interval from config)
        if step % checkpoint_interval == 0 && step > 0 {
            if args.debug {
                println!("  💾 Checkpoint at step {} (saving disabled for now)", step);
            }
            // TODO: Save model checkpoint
        }
        
        // Check if episode ended
        if step_done {
            episode_rewards.push(current_episode_reward);
            total_reward_sum += current_episode_reward;
            total_episodes += 1;
            
            if args.debug {
                let avg_reward = total_reward_sum / total_episodes as f32;
                println!("  ✓ Episódio {}: reward={:.2}, avg={:.2}", 
                         total_episodes, current_episode_reward, avg_reward);
            }
            
            current_episode_reward = 0.0;
            // Reset and resend ALL side channels (engine settings + env parameters)
            // This ensures Unity always receives updated configuration
            let combined_side_channel = rl_core::side_channel::combine_side_channels(&side_channels);
            current_output = server.reset_with_side_channel(combined_side_channel).await?;
        }
    }
    
    // Final summary
    println!("\n📊 Treinamento concluído!");
    println!("  - Total updates: {}", num_updates);
    println!("  - Episodes completados: {}", episode_rewards.len());
    if !episode_rewards.is_empty() {
        let avg_reward = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
        let max_reward = episode_rewards.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_reward = episode_rewards.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        println!("  - Reward médio: {:.2}", avg_reward);
        println!("  - Reward máximo: {:.2}", max_reward);
        println!("  - Reward mínimo: {:.2}", min_reward);
    }
    
    Ok(())
}

async fn run_multi_env_training_loop(
    mut multi_env: rl_core::multi_env_manager::MultiEnvManager,
    specs: UnityRlInitializationOutputProto,
    config: &rl_core::settings::RootConfig,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    use burn::backend::NdArray;
    use burn::tensor::Tensor;
    use rl_core::ppo_trainer::PPOTrainer;
    use rl_core::ppo_buffer::RolloutBuffer;
    
    let num_envs = multi_env.num_envs();
    
    // Extract specs from first brain
    let brain = specs.brain_parameters.first()
        .ok_or("No brain parameters")?;
    
    let action_spec = brain.action_spec.as_ref()
        .ok_or("No action spec")?;
    
    let num_continuous = action_spec.num_continuous_actions as usize;
    let num_discrete_branches = action_spec.discrete_branch_sizes.len();
    
    if args.debug {
        println!("\n📊 Configuração do agente (multi-env):");
        println!("  - Behavior: {}", brain.brain_name);
        println!("  - Ações contínuas: {}", num_continuous);
        println!("  - Branches discretas: {}", num_discrete_branches);
        println!("  - Ambientes paralelos: {}", num_envs);
    }
    
    // Get behavior config
    let behavior_name = brain.brain_name.split('?').next().unwrap_or(&brain.brain_name);
    let behavior_cfg = config.behaviors.get(behavior_name)
        .ok_or(format!("Behavior '{}' não encontrado no config", behavior_name))?;
    
    // Prepare side channels (same for all envs)
    let env_params = config.environment_parameters.clone()
        .or_else(|| config.env_settings.as_ref()
            .and_then(|es| es.environment_parameters.clone()))
        .unwrap_or_default();
    
    let engine_config = rl_core::side_channel::EngineConfig {
        width: config.engine_settings.as_ref().map(|e| e.width).unwrap_or(args.width),
        height: config.engine_settings.as_ref().map(|e| e.height).unwrap_or(args.height),
        quality_level: config.engine_settings.as_ref().map(|e| e.quality_level).unwrap_or(args.quality_level),
        time_scale: config.engine_settings.as_ref().map(|e| e.time_scale).unwrap_or(args.time_scale),
        target_frame_rate: config.engine_settings.as_ref().map(|e| e.target_frame_rate).unwrap_or(args.target_frame_rate),
        capture_frame_rate: config.engine_settings.as_ref().map(|e| e.capture_frame_rate).unwrap_or(args.capture_frame_rate),
    };
    
    let mut side_channels = vec![rl_core::side_channel::serialize_engine_config(&engine_config)];
    
    if !env_params.is_empty() {
        if args.debug {
            println!("\n📤 Environment Parameters:");
            for (key, value) in &env_params {
                println!("  - {}: {:?}", key, value);
            }
        }
        side_channels.push(rl_core::side_channel::serialize_environment_parameters(&env_params));
    }
    
    if args.debug {
        println!("\n⚙️  Engine Settings:");
        println!("  - Resolution: {}x{}", engine_config.width, engine_config.height);
        println!("  - Quality Level: {}", engine_config.quality_level);
        println!("  - Time Scale: {}x", engine_config.time_scale);
        println!("  - Target FPS: {}", if engine_config.target_frame_rate < 0 { "Unlimited".to_string() } else { engine_config.target_frame_rate.to_string() });
    }
    
    // Reset all environments
    let combined_side_channel = rl_core::side_channel::combine_side_channels(&side_channels);
    let reset_outputs = multi_env.reset_all_with_side_channel(combined_side_channel.clone()).await?;
    
    if args.debug {
        println!("✓ {} ambientes resetados!", reset_outputs.len());
    }
    
    // Determine observation size from first environment
    let obs_size = if let Some(output) = reset_outputs.first() {
        if let Some(rl_output) = &output.rl_output {
            if let Some((_, agent_list)) = rl_output.agent_infos.iter().next() {
                if let Some(agent_info) = agent_list.value.first() {
                    let obs_len: usize = agent_info.observations.iter()
                        .map(|obs| obs.shape.iter().product::<i32>() as usize)
                        .sum();
                    if args.debug {
                        println!("  - Observação size: {}", obs_len);
                    }
                    obs_len
                } else {
                    return Err("Agent list is empty".into());
                }
            } else {
                return Err("No agent info in reset output".into());
            }
        } else {
            return Err("No RL output from reset".into());
        }
    } else {
        return Err("No reset outputs".into());
    };
    
    // Initialize networks
    let device = Default::default();
    let hidden_units = behavior_cfg.network_settings.as_ref()
        .and_then(|n| n.hidden_units)
        .unwrap_or(128);
    let num_layers = behavior_cfg.network_settings.as_ref()
        .and_then(|n| n.num_layers)
        .unwrap_or(2);
    
    let learning_rate = if behavior_cfg.hyperparameters.learning_rate > 0.0 {
        behavior_cfg.hyperparameters.learning_rate as f64
    } else {
        0.0003
    };
    
    let mut trainer = PPOTrainer::<NdArray>::new(
        obs_size,
        num_continuous,
        hidden_units,
        num_layers,
        learning_rate,
        &device,
    );
    
    // Configure PPO hyperparameters
    let hp = &behavior_cfg.hyperparameters;
    if hp.epsilon > 0.0 { trainer.clip_epsilon = hp.epsilon; }
    if hp.num_epoch > 0 { trainer.num_epochs = hp.num_epoch as usize; }
    if hp.lambd > 0.0 { trainer.gae_lambda = hp.lambd; }
    if hp.beta > 0.0 { trainer.entropy_coef = hp.beta; }
    
    trainer.batch_size = hp.batch_size;
    trainer.buffer_size = hp.buffer_size;
    
    if let Some(rs) = &behavior_cfg.reward_signals {
        if let Some(extrinsic) = rs.get("extrinsic") {
            trainer.gamma = extrinsic.gamma.unwrap_or(0.99);
        }
    }
    
    println!("🧠 PPO Trainer configurado (lr={}, hidden={}, layers={})", 
             learning_rate, hidden_units, num_layers);
    
    if args.debug {
        println!("  📋 Hyperparameters:");
        println!("     • Clip epsilon: {}", trainer.clip_epsilon);
        println!("     • Num epochs: {}", trainer.num_epochs);
        println!("     • Gamma: {}", trainer.gamma);
        println!("     • Lambda (GAE): {}", trainer.gae_lambda);
        println!("     • Beta (entropy): {}", trainer.entropy_coef);
        println!("     • Batch size: {}", trainer.batch_size);
        println!("     • Buffer size: {}", trainer.buffer_size);
    }
    
    // Training parameters
    let max_steps = if behavior_cfg.max_steps > 0 {
        behavior_cfg.max_steps as usize
    } else {
        50000
    };
    
    let time_horizon = if behavior_cfg.time_horizon > 0 {
        behavior_cfg.time_horizon as usize
    } else {
        64
    };
    
    let summary_freq = behavior_cfg.summary_freq as usize;
    let checkpoint_interval = behavior_cfg.checkpoint_interval as usize;
    
    println!("🎯 Treinamento multi-env: max_steps={}, horizon={}, summary_freq={}", 
             max_steps, time_horizon, summary_freq);
    println!();
    
    let mut buffer = RolloutBuffer::new();
    let mut current_outputs = reset_outputs;
    let mut total_reward_sum = 0.0;
    let mut total_episodes = 0;
    let mut num_updates = 0;
    
    // Per-environment episode rewards
    let mut env_episode_rewards = vec![0.0; num_envs];
    
    for step in 1..=max_steps {
        // Generate actions for all environments
        let mut all_actions = Vec::new();
        
        for (_env_idx, output) in current_outputs.iter().enumerate() {
            let rl_output = output.rl_output.as_ref().unwrap();
            
            let mut agent_actions = HashMap::new();
            for (behavior_name, agent_list) in &rl_output.agent_infos {
                let mut actions_for_behavior = Vec::new();
                
                for agent_info in &agent_list.value {
                    let obs: Vec<f32> = agent_info.observations.iter()
                        .flat_map(|obs| {
                            if let Some(ref data) = obs.observation_data {
                                use rl_core::communicator_objects::observation_proto::ObservationData;
                                match data {
                                    ObservationData::FloatData(fd) => fd.data.clone(),
                                    ObservationData::CompressedData(_) => vec![],
                                }
                            } else {
                                vec![]
                            }
                        })
                        .collect();
                    
                    let obs_tensor = Tensor::<NdArray, 1>::from_floats(obs.as_slice(), &device)
                        .reshape([1, obs.len()]);
                    
                    let (action_tensor, _value_tensor, _log_prob) = trainer.get_action_and_value(obs_tensor);
                    let action_data = action_tensor.to_data();
                    let actions: Vec<f32> = action_data.to_vec().unwrap();
                    
                    let action_proto = rl_core::communicator_objects::AgentActionProto {
                        vector_actions_deprecated: vec![],
                        value: 0.0,
                        continuous_actions: actions,
                        discrete_actions: vec![],
                    };
                    
                    actions_for_behavior.push(action_proto);
                }
                
                agent_actions.insert(
                    behavior_name.clone(),
                    rl_core::communicator_objects::unity_rl_input_proto::ListAgentActionProto {
                        value: actions_for_behavior,
                    }
                );
            }
            
            let step_input = UnityInputProto {
                rl_input: Some(UnityRlInputProto {
                    agent_actions,
                    command: CommandProto::Step as i32,
                    side_channel: vec![],
                }),
                rl_initialization_input: None,
            };
            
            all_actions.push(step_input);
        }
        
        // Step all environments in parallel
        current_outputs = multi_env.step_all(all_actions).await?;
        
        // Collect rewards from all environments
        let mut total_step_reward = 0.0;
        let mut any_done = false;
        
        for (env_idx, output) in current_outputs.iter().enumerate() {
            let step_rl_output = output.rl_output.as_ref().unwrap();
            
            if let Some((_, agent_list)) = step_rl_output.agent_infos.iter().next() {
                if let Some(agent_info) = agent_list.value.first() {
                    let step_reward = agent_info.reward;
                    let step_done = agent_info.done;
                    
                    env_episode_rewards[env_idx] += step_reward;
                    total_step_reward += step_reward;
                    
                    if step_done {
                        total_reward_sum += env_episode_rewards[env_idx];
                        total_episodes += 1;
                        
                        if args.debug {
                            let avg_reward = total_reward_sum / total_episodes as f32;
                            println!("  ✓ Env {} episódio completo: reward={:.2}, avg={:.2}", 
                                     env_idx + 1, env_episode_rewards[env_idx], avg_reward);
                        }
                        
                        env_episode_rewards[env_idx] = 0.0;
                        any_done = true;
                    }
                }
            }
        }
        
        // Log progress
        if step % summary_freq == 0 || step == max_steps {
            let avg_reward = if total_episodes > 0 {
                total_reward_sum / total_episodes as f32
            } else {
                0.0
            };
            println!("[Step {}/{: >8}] Reward: {:.3} | Avg: {:.2} | Episodes: {} | Envs: {}", 
                     step, max_steps, total_step_reward / num_envs as f32, avg_reward, total_episodes, num_envs);
        }
        
        // Update policy when buffer is full
        if buffer.len() >= time_horizon {
            let (policy_loss, value_loss, entropy) = trainer.update(&mut buffer);
            num_updates += 1;
            
            if args.debug || step % summary_freq == 0 {
                println!("  🔄 Update #{}: policy={:.4}, value={:.4}, entropy={:.4}", 
                         num_updates, policy_loss, value_loss, entropy);
            }
            
            buffer.clear();
        }
        
        // Save checkpoint
        if step % checkpoint_interval == 0 && step > 0 {
            if args.debug {
                println!("  💾 Checkpoint at step {} (saving disabled for now)", step);
            }
        }
        
        // Reset environments that are done
        if any_done {
            // For now, reset all envs together (could optimize to only reset done envs)
            current_outputs = multi_env.reset_all_with_side_channel(combined_side_channel.clone()).await?;
        }
    }
    
    // Final summary
    println!("\n📊 Treinamento multi-env concluído!");
    println!("  - Total updates: {}", num_updates);
    println!("  - Episodes completados: {}", total_episodes);
    if total_episodes > 0 {
        let avg_reward = total_reward_sum / total_episodes as f32;
        println!("  - Reward médio: {:.2}", avg_reward);
    }
    
    Ok(())
}

fn main() {
    let a = Args::parse();
    if a.torch || a.tensorflow { /* placeholders per original */ }
    if a.env_path.is_some() || a.num_envs > 0 { /* parsed, used by YAML/env settings */ }

    if a.trainer_config_path.is_empty() {
        eprintln!("error: missing trainer_config_path");
        std::process::exit(2);
    }
    if a.gui {
        println!("Abrindo GUI... (em breve)");
        return;
    }

    let yaml_path = a.trainer_config_path.last().unwrap();
    let yaml = std::fs::read_to_string(yaml_path).expect("failed to read config");
    let mut root: rl_core::settings::RootConfig = serde_yaml::from_str(&yaml).expect("invalid YAML");
    
    // Validate torch settings (device)
    if let Some(ref torch_settings) = root.torch_settings {
        let device = torch_settings.device.to_lowercase();
        if device.contains("cuda") {
            // Check if CUDA backend is available in Burn
            // For now, Burn with NdArray backend doesn't support CUDA
            eprintln!("⚠️  AVISO: Device 'cuda' especificado mas backend atual (NdArray) não suporta CUDA.");
            eprintln!("   O treinamento continuará usando CPU.");
            eprintln!("   Para usar CUDA, considere:");
            eprintln!("     - Usar backend wgpu (GPU via WebGPU/Vulkan)");
            eprintln!("     - Ou alterar 'device' para 'cpu' no YAML");
        } else if device == "cpu" {
            if !a.debug {
                println!("✓ Usando CPU device");
            }
        } else if device == "mps" && cfg!(target_os = "macos") {
            println!("✓ MPS device detectado (Apple Silicon)");
        } else {
            eprintln!("⚠️  Aviso: Device '{}' não reconhecido, usando CPU", device);
        }
    }
    // Override env settings from CLI when provided
    if root.env_settings.is_none() { 
        root.env_settings = Some(rl_core::settings::EnvSettings{ 
            env_path: None, 
            base_port: None, 
            num_envs: None, 
            num_areas: None, 
            seed: None, 
            timeout_wait: None,
            max_lifetime_restarts: None,
            restarts_rate_limit_n: None,
            restarts_rate_limit_period_s: None,
            environment_parameters: None 
        }); 
    }
    if let Some(es) = &mut root.env_settings {
        if a.env_path.is_some() { es.env_path = a.env_path.clone(); }
        if a.base_port != 5005 { es.base_port = Some(a.base_port); }
        if a.num_envs != 1 { es.num_envs = Some(a.num_envs); }
    }
    let _behavior = pick_behavior(&root).expect("no behaviors");

    // Launch Unity if executable provided, else wait for Editor play
    let env_mgr = rl_core::env_manager::UnityEnvManager::from_settings(&root.env_settings);
    
    let num_envs = env_mgr.num_envs;
    println!("\n🌍 Configuração de ambientes:");
    println!("  - Número de ambientes: {}", num_envs);
    println!("  - Porta base: {}", env_mgr.base_port);
    
    if num_envs > 1 {
        println!("  - Portas: {} até {}", env_mgr.base_port, env_mgr.base_port + num_envs as u16 - 1);
    }
    
    let children = match env_mgr.start_all() {
        Ok(c) => {
            if !c.is_empty() {
                println!("✓ {} instância(s) Unity iniciada(s)", c.len());
            }
            c
        },
        Err(e) => {
            eprintln!("[warn] Não foi possível iniciar executável: {}\nEsperando Play no Unity Editor na porta base {}...", e, env_mgr.base_port);
            Vec::new()
        }
    };
    if children.is_empty() {
        eprintln!("Aguardando Play no Unity Editor na porta base {}..." , env_mgr.base_port);
    }
    
    // Start gRPC server(s) - Rust is SERVER, Unity is CLIENT
    // For multi-env: create multiple servers on sequential ports
    println!("\nIniciando servidor(es) gRPC...");
    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    
    if num_envs == 1 {
        // Single environment - simple case
        let mut server = rt.block_on(async {
            rl_core::grpc_server::GrpcServer::new_with_verbosity(env_mgr.base_port, a.debug)
        });
        
        println!("✓ Servidor gRPC na porta {}", env_mgr.base_port);
        println!("⏳ Aguardando Unity conectar (timeout: {}s)...", a.timeout_wait);
    
        // Wait for Unity to initialize with timeout
        let init_config = rl_core::grpc_server::InitConfig {
            seed: root.env_settings.as_ref().and_then(|e| e.seed).unwrap_or(42),
            num_areas: a.num_areas as i32,
            communication_version: "1.5.0".to_string(),
            package_version: "0.1.0-rust".to_string(),
        };
        
        let specs = rt.block_on(async {
            tokio::time::timeout(
                tokio::time::Duration::from_secs(a.timeout_wait),
                server.initialize(init_config)
            ).await
        });
    
        match specs {
            Ok(Ok(output)) => {
                println!("✓ Unity conectado!");
                if a.debug {
                    if let Some(init) = &output.rl_initialization_output {
                        println!("  - Behaviors: {}", init.brain_parameters.len());
                        for bp in &init.brain_parameters {
                            let action_info = if let Some(ref spec) = bp.action_spec {
                                format!("discrete={}, continuous={}", 
                                        spec.discrete_branch_sizes.len(),
                                        spec.num_continuous_actions)
                            } else {
                                "deprecated".to_string()
                            };
                            println!("    • {}: actions=[{}]", bp.brain_name, action_info);
                        }
                    }
                }
                
                // Extract specs for policy initialization
                let specs = if let Some(init) = &output.rl_initialization_output {
                    init.clone()
                } else {
                    eprintln!("❌ Specs incompletos");
                    std::process::exit(1);
                };
                
                // Start training loop
                rt.block_on(async {
                    match run_training_loop(server, specs, &root, &a).await {
                        Ok(_) => println!("\n✓ Treinamento concluído!"),
                        Err(e) => eprintln!("\n❌ Erro no treinamento: {}", e),
                    }
                });
            },
            Ok(Err(e)) => {
                eprintln!("❌ Erro durante inicialização: {}", e);
                std::process::exit(1);
            },
            Err(_) => {
                eprintln!("❌ Timeout: Unity não conectou em {}s", a.timeout_wait);
                if a.debug {
                    eprintln!("\nDicas de diagnóstico:");
                    eprintln!("  1. Verifique se o executável Unity foi iniciado");
                    eprintln!("  2. Unity deve conectar em: http://127.0.0.1:{}", env_mgr.base_port);
                    eprintln!("  3. Veja logs do Unity para erros de conexão");
                    eprintln!("  4. Teste com Editor Unity em modo Play");
                }
                std::process::exit(1);
            }
        }
    } else {
        // Multiple environments - use MultiEnvManager
        println!("✓ {} servidores gRPC iniciados (portas {}-{})", 
                 num_envs, env_mgr.base_port, env_mgr.base_port + num_envs as u16 - 1);
        println!("⏳ Aguardando {} instâncias Unity conectarem...", num_envs);
        
        let mut multi_env = rt.block_on(async {
            rl_core::multi_env_manager::MultiEnvManager::new(env_mgr.base_port, num_envs, a.debug)
        });
        
        let init_config = rl_core::grpc_server::InitConfig {
            seed: root.env_settings.as_ref().and_then(|e| e.seed).unwrap_or(42),
            num_areas: a.num_areas as i32,
            communication_version: "1.5.0".to_string(),
            package_version: "0.1.0-rust".to_string(),
        };
        
        match rt.block_on(multi_env.initialize_all(init_config, a.timeout_wait)) {
            Ok(all_specs) => {
                println!("✓ {} ambientes conectados!", all_specs.len());
                
                // Use first environment's specs (all should be identical)
                if all_specs.is_empty() {
                    eprintln!("❌ Nenhum ambiente conectado");
                    std::process::exit(1);
                }
                
                let specs = all_specs[0].clone();
                
                // Start multi-environment training loop
                rt.block_on(async {
                    match run_multi_env_training_loop(multi_env, specs, &root, &a).await {
                        Ok(_) => println!("\n✓ Treinamento multi-env concluído!"),
                        Err(e) => eprintln!("\n❌ Erro no treinamento multi-env: {}", e),
                    }
                });
            },
            Err(e) => {
                eprintln!("❌ Erro conectando múltiplos ambientes: {}", e);
                std::process::exit(1);
            }
        }
    }
    
    // Old code commented for now - will be replaced with server-based training
    // type B = burn_ndarray::NdArray;
    // let device = Default::default();
    // rl_core::ppo::run_from_config::<B>(&device, &root, &behavior);
}

