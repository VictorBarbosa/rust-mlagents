// Main entry point for rust-mlagents CLI
use clap::Parser;
use rl_core::cli::{Cli, Commands, DeviceType, print_banner, print_configuration};
use rl_core::trainers::sac::{SACTrainer, SACConfig};
use rl_core::trainers::settings::RunOptions;
use tch::Device;
use std::fs;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let cli = Cli::parse();
    
    // Validate arguments
    if let Err(e) = cli.validate() {
        eprintln!("❌ Error: {}", e);
        std::process::exit(1);
    }
    
    // Print banner
    print_banner();
    
    // Handle subcommands
    if let Some(command) = &cli.command {
        return handle_command(command, &cli).await;
    }
    
    // Default: run training
    run_training(&cli).await
}

async fn handle_command(command: &Commands, cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Train { config } => {
            let _config_path = config.as_ref()
                .or(cli.config_path.as_ref());
            
            if let Some(path) = _config_path {
                println!("📚 Loading config from: {}", path.display());
            }
            run_training(cli).await
        }
        
        Commands::Infer { model, env } => {
            println!("🔮 Running inference...");
            println!("   Model: {}", model.display());
            if let Some(env_path) = env {
                println!("   Env: {}", env_path.display());
            }
            run_inference(cli, model).await
        }
        
        Commands::Export { checkpoint, output } => {
            println!("📦 Exporting model to ONNX...");
            println!("   Checkpoint: {}", checkpoint.display());
            println!("   Output: {}", output.display());
            export_to_onnx(checkpoint, output).await
        }
        
        Commands::Version => {
            println!("rust-mlagents version: {}", env!("CARGO_PKG_VERSION"));
            println!("tch-rs version: 0.18");
            println!("CUDA available: {}", tch::Cuda::is_available());
            println!("MPS available: {}", tch::utils::has_mps());
            Ok(())
        }
    }
}

async fn run_training(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    print_configuration(&cli);
    
    // Create results directories
    fs::create_dir_all(cli.get_results_path())?;
    fs::create_dir_all(cli.get_checkpoint_path())?;
    fs::create_dir_all(cli.get_logs_path())?;
    
    // Determine device
    let device = match cli.device {
        DeviceType::Auto => {
            if tch::Cuda::is_available() {
                println!("🎮 CUDA is available, using GPU");
                Device::Cuda(cli.cuda_device)
            } else if tch::utils::has_mps() {
                println!("🍎 MPS is available, using Apple Silicon GPU");
                Device::Mps
            } else {
                println!("💻 Using CPU");
                Device::Cpu
            }
        }
        DeviceType::Cpu => {
            println!("💻 Using CPU (explicit)");
            Device::Cpu
        }
        DeviceType::Cuda => {
            if !tch::Cuda::is_available() {
                eprintln!("❌ CUDA requested but not available");
                std::process::exit(1);
            }
            println!("🎮 Using CUDA device {}", cli.cuda_device);
            Device::Cuda(cli.cuda_device)
        }
        DeviceType::Mps => {
            if !tch::utils::has_mps() {
                eprintln!("❌ MPS requested but not available");
                std::process::exit(1);
            }
            println!("🍎 Using Apple MPS");
            Device::Mps
        }
    };
    
    // Load or create config
    let (mut config, yaml_settings) = if let Some(config_path) = &cli.config_path {
        println!("📚 Loading configuration from: {}", config_path.display());
        let config = load_config_from_yaml(config_path)?;
        
        // Also load the full YAML to get env_settings
        let contents = fs::read_to_string(config_path)?;
        let run_options: RunOptions = serde_yaml::from_str(&contents)?;
        
        (config, Some(run_options))
    } else {
        println!("⚙️  Using default SAC configuration");
        (SACConfig::default(), None)
    };
    
    // Auto-configure dtype based on device for cross-platform compatibility
    config = config.with_device(device);
    
    // Extract behavior name from YAML, or use run_id as a fallback.
    let behavior_name = yaml_settings.as_ref()
        .and_then(|y| y.behaviors.keys().next().cloned())
        .unwrap_or_else(|| cli.run_id.clone());

    println!("\n🧠 SAC Configuration for behavior: {}", behavior_name);
    println!("   Hidden layers:  {:?}", config.hidden_layers);
    println!("   Batch size:     {}", config.batch_size);
    println!("   Buffer size:    {}", config.buffer_size);
    println!("   Learning rates: actor={}, critic={}", config.lr_actor, config.lr_critic);
    println!("   Gamma:          {}", config.gamma);
    println!("   Tau:            {}", config.tau);
    println!("   Data type:      {:?} (auto-configured for device)", config.dtype);
    println!();
    
    // Note: obs_dim and action_dim will be determined from Unity
    let obs_dim = 8;  // Placeholder, will be updated from Unity
    let action_dim = 2;  // Placeholder, will be updated from Unity
    
    // Create SAC trainer
    let mut sac_trainer = if cli.resume || cli.initialize_from.is_some() {
        println!("⚠️  Resume/init not yet implemented, creating new trainer");
        SACTrainer::new(obs_dim, action_dim, config, device)?
    } else {
        println!("🆕 Creating new SAC trainer");
        SACTrainer::new(obs_dim, action_dim, config, device)?
    };
    
    // Get effective env_path (CLI overrides YAML)
    let env_path = cli.env_path.clone().or_else(|| {
        yaml_settings.as_ref().and_then(|y| y.env_settings.env_path.as_ref().map(PathBuf::from))
    });
    
    // Run training loop
    if cli.inference {
        println!("🔮 Running in inference mode");
        println!("⚠️  Inference not yet implemented");
    } else {
        println!("🚀 Starting training...\n");
        run_training_loop(&mut sac_trainer, &cli, env_path.as_ref(), &behavior_name).await?;
    }
    
    println!("\n✅ Training completed successfully!");
    println!("📁 Results saved to: {}", cli.get_results_path().display());
    
    Ok(())
}

async fn run_training_loop(
    sac_trainer: &mut SACTrainer,
    cli: &Cli,
    env_path: Option<&PathBuf>,
    behavior_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 Training Configuration:");
    println!("   Base Port:      {}", cli.base_port);
    println!("   Num Envs:       {}", cli.num_envs);
    println!("   Run ID:         {}", cli.run_id);
    println!();
    
    // Check if we should use Unity or dummy training
    if let Some(unity_path) = env_path {
        println!("🎯 Unity Environment Mode");
        println!("   Path: {}", unity_path.display());
        println!("   Port: {}", cli.base_port);
        println!();
        
        // Connect to Unity
        println!("🔌 Connecting to Unity environment...");
        match run_unity_training(unity_path, cli, sac_trainer, behavior_name).await {
            Ok(_) => {
                println!("✅ Unity training completed successfully!");
            }
            Err(e) => {
                println!("❌ Unity training failed: {}", e);
                println!();
                println!("💡 Make sure:");
                println!("   1. Unity executable path is correct");
                println!("   2. Unity build includes ML-Agents package");
                println!("   3. Port {} is available", cli.base_port);
                println!("   4. Unity scene has ML-Agents behaviors");
                return Err(e);
            }
        }
    } else {
        println!("🎲 Dummy Training Mode (no Unity environment)");
        println!("   This mode trains with random data for testing");
        println!();
        println!("💡 To train with Unity:");
        println!("   cargo run --bin rust-mlagents -- config.yaml --env path/to/unity.app");
        println!();
        println!("For now, running dummy training...");
        println!();
        
        // Run a simple dummy training loop
        use rl_core::trainers::sac::Transition;
        use rl_core::trainers::TensorBoardWriter;
        
        let tb = TensorBoardWriter::new("runs", &cli.run_id)?;
        
        println!("Training for 100 episodes with dummy data...");
        println!("Checkpoint interval: 500 steps");
        let mut total_steps = 0;
        let checkpoint_interval = 500;
        let mut next_checkpoint_step = checkpoint_interval;
        
        for episode in 0..100 {
            let mut episode_reward = 0.0;
            
            for _step in 0..100 {
                total_steps += 1;
                // Generate random transition
                let transition = Transition {
                    obs: vec![0.0; sac_trainer.obs_dim as usize],
                    action: vec![0.0; sac_trainer.action_dim as usize],
                    reward: (rand::random::<f32>() - 0.5) * 2.0,
                    next_obs: vec![0.0; sac_trainer.obs_dim as usize],
                    done: false,
                };
                
                episode_reward += transition.reward;
                sac_trainer.store_transition(transition);
                
                // Update
                if let Some(metrics) = sac_trainer.update() {
                    let step = sac_trainer.get_step();
                    if step % 100 == 0 {
                        tb.add_scalar("loss/actor", metrics.actor_loss, step)?;
                        tb.add_scalar("loss/critic", metrics.critic_loss, step)?;
                        tb.add_scalar("values/alpha", metrics.alpha, step)?;
                    }
                }
            }
            
            tb.add_scalar("episode/reward", episode_reward as f64, episode as i64)?;
            println!("Episode {}: Reward = {:.2}, Total Steps = {}", episode, episode_reward, total_steps);
            
            // Save checkpoint based on steps (ML-Agents style)
            if total_steps >= next_checkpoint_step {
                let checkpoint_path = cli.get_checkpoint_path().join(format!("{}-{}.pt", behavior_name, total_steps));
                sac_trainer.save_checkpoint(checkpoint_path.to_str().unwrap())?;
                
                // Export ONNX with same name
                let onnx_path = cli.get_checkpoint_path().join(format!("{}-{}", behavior_name, total_steps));
                sac_trainer.export_onnx(onnx_path.to_str().unwrap())?;
                println!("✓ Checkpoint & ONNX saved at step {}", total_steps);
                
                next_checkpoint_step += checkpoint_interval;
            }
        }
        
        // Final save
        println!("💾 Saving final checkpoint...");
        let final_path = cli.get_checkpoint_path().join(format!("{}-{}.pt", behavior_name, total_steps));
        sac_trainer.save_checkpoint(final_path.to_str().unwrap())?;
        
        let final_onnx_path = cli.get_checkpoint_path().join(format!("{}-{}", behavior_name, total_steps));
        sac_trainer.export_onnx(final_onnx_path.to_str().unwrap())?;
        println!("✅ Final checkpoint & ONNX saved at step {}", total_steps);
        
        println!();
        println!("✅ Dummy training completed!");
        println!("📊 View logs: tensorboard --logdir=runs");
    }
    
    Ok(())
}

async fn run_inference_loop(
    _sac_trainer: &SACTrainer,
    _cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔮 Inference mode not yet fully implemented");
    println!("   This will run the trained policy without updates");
    Ok(())
}

async fn run_inference(
    _cli: &Cli,
    _model_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔮 Inference command not yet implemented");
    Ok(())
}

async fn export_to_onnx(
    checkpoint_path: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 Export checkpoint: {}", checkpoint_path.display());
    println!("📝 Output path: {}", output_path.display());
    
    // TODO: Implement checkpoint loading and ONNX export
    println!("⚠️  Export not yet implemented");
    
    Ok(())
}

fn load_config_from_yaml(path: &std::path::Path) -> Result<SACConfig, Box<dyn std::error::Error>> {
    use rl_core::trainers::settings::{TrainerType, HyperparameterSettings};
    
    let contents = fs::read_to_string(path)?;
    let run_options: RunOptions = serde_yaml::from_str(&contents)?;
    
    // Get the first behavior (for now, we only support single behavior)
    let (behavior_name, trainer_settings) = run_options.behaviors.iter().next()
        .ok_or("No behaviors found in config")?;
    
    println!("📋 Loaded behavior: {}", behavior_name);
    
    // Validate it's SAC
    if !matches!(trainer_settings.trainer_type, TrainerType::SAC) {
        return Err(format!("Expected SAC trainer, got {:?}", trainer_settings.trainer_type).into());
    }
    
    // Extract SAC hyperparameters
    let sac_hyperparams = match &trainer_settings.hyperparameters {
        HyperparameterSettings::SAC(sac) => sac,
        _ => return Err("SAC hyperparameters not found".into()),
    };
    
    // Convert to SACConfig
    let config = SACConfig {
        hidden_layers: vec![trainer_settings.network_settings.hidden_units as i64; trainer_settings.network_settings.num_layers],
        batch_size: sac_hyperparams.batch_size,
        buffer_size: sac_hyperparams.buffer_size,
        lr_actor: sac_hyperparams.learning_rate as f64,
        lr_critic: sac_hyperparams.learning_rate as f64,
        lr_alpha: sac_hyperparams.learning_rate as f64,
        gamma: sac_hyperparams.gamma as f64,
        tau: sac_hyperparams.tau as f64,
        init_alpha: sac_hyperparams.init_entcoef as f64,
        auto_alpha: true, // Default to auto-tuning alpha
        target_entropy: None, // Will be computed based on action space
        warmup_steps: 1000, // Default warmup steps
        gradient_steps: 1,
        target_update_interval: 1,
        checkpoint_interval: trainer_settings.checkpoint_interval as usize,
        save_onnx: true,
        dtype: tch::Kind::Float,
    };
    
    println!("✅ Config loaded successfully");
    Ok(config)
}

async fn run_unity_training(
    unity_path: &PathBuf,
    cli: &Cli,
    sac_trainer: &mut SACTrainer,
    behavior_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use rl_core::env::unity_env::UnityEnvironment;
    use rl_core::trainers::sac::Transition;
    use rl_core::trainers::TensorBoardWriter;
    
    // Connect to Unity
    let mut env = UnityEnvironment::connect(unity_path, cli.base_port).await?;
    
    // Get initial obs to determine dimensions
    let initial_obs = env.reset().await?;
    let obs_dim = initial_obs.len() as i64;
    let action_dim = 2; // TODO: Get from Unity specs
    
    println!("📊 Environment specs:");
    println!("   Observation dim: {}", obs_dim);
    println!("   Action dim: {}", action_dim);
    
    // Recreate trainer with correct dimensions
    println!("🔧 Creating SAC trainer with Unity dimensions...");
    let config = sac_trainer.config.clone();
    let device = sac_trainer.device;
    *sac_trainer = SACTrainer::new(obs_dim, action_dim, config, device)?;
    
    // Initialize TensorBoard
    let tb = TensorBoardWriter::new("runs", &cli.run_id)?;
    
    println!();
    println!("🎮 Starting Unity training loop...");
    println!("   Max episodes: 1000");
    println!("   Max steps per episode: 1000");
    println!("   Checkpoint interval: 500 steps (ML-Agents: 500000)");
    println!();
    
    let max_episodes = 1000;
    let max_steps_per_episode = 1000;
    let mut total_steps = 0;
    let checkpoint_interval = 500; // ML-Agents default: 500000 (reduced for testing)
    let mut next_checkpoint_step = checkpoint_interval;
    
    for episode in 0..max_episodes {
        // Reset environment
        let mut obs = env.reset().await?;
        let mut episode_reward = 0.0;
        let mut episode_steps = 0;
        
        for _step in 0..max_steps_per_episode {
            // Get action from SAC
            let action = sac_trainer.select_action_from_vec(&obs);
            
            // Step environment
            let (next_obs, reward, done) = env.step(action.clone()).await?;
            
            episode_reward += reward;
            episode_steps += 1;
            total_steps += 1;
            
            // Store transition
            let transition = Transition {
                obs: obs.clone(),
                action: action.clone(),
                reward,
                next_obs: next_obs.clone(),
                done,
            };
            
            sac_trainer.store_transition(transition);
            
            // Update SAC
            if let Some(metrics) = sac_trainer.update() {
                let global_step = sac_trainer.get_step();
                if global_step % 100 == 0 {
                    tb.add_scalar("loss/actor", metrics.actor_loss, global_step)?;
                    tb.add_scalar("loss/critic", metrics.critic_loss, global_step)?;
                    tb.add_scalar("values/q1", metrics.q1_value, global_step)?;
                    tb.add_scalar("values/q2", metrics.q2_value, global_step)?;
                    tb.add_scalar("values/alpha", metrics.alpha, global_step)?;
                    
                    println!("Step {}: Actor Loss: {:.4}, Critic Loss: {:.4}, Alpha: {:.4}",
                             global_step, metrics.actor_loss, metrics.critic_loss, metrics.alpha);
                }
            }
            
            obs = next_obs;
            
            if done {
                break;
            }
        }
        
        // Log episode metrics
        tb.add_scalar("episode/reward", episode_reward as f64, episode as i64)?;
        tb.add_scalar("episode/length", episode_steps as f64, episode as i64)?;
        tb.add_scalar("training/step", total_steps as f64, episode as i64)?;
        
        println!("Episode {}: Reward = {:.2}, Steps = {}, Total Steps = {}", 
                 episode, episode_reward, episode_steps, total_steps);
        
        // Save checkpoint periodically based on STEPS (ML-Agents style)
        if total_steps >= next_checkpoint_step {
            println!("💾 Saving checkpoint at step {}...", total_steps);
            let checkpoint_path = cli.get_checkpoint_path().join(format!("{}-{}.pt", behavior_name, total_steps));
            sac_trainer.save_checkpoint(checkpoint_path.to_str().unwrap())?;
            
            // Export ONNX with same name (ML-Agents does this automatically)
            let onnx_path = cli.get_checkpoint_path().join(format!("{}-{}", behavior_name, total_steps));
            sac_trainer.export_onnx(onnx_path.to_str().unwrap())?;
            println!("✓ Checkpoint saved at step {}: {}.pt", total_steps, checkpoint_path.display());
            println!("✓ ONNX exported: {}.onnx", onnx_path.display());
            
            next_checkpoint_step += checkpoint_interval;
        }
    }
    
    // Final save (ML-Agents compatible format)
    println!();
    println!("💾 Saving final checkpoint...");
    let final_step = total_steps;
    let final_path = cli.get_checkpoint_path().join(format!("{}-{}.pt", behavior_name, final_step));
    sac_trainer.save_checkpoint(final_path.to_str().unwrap())?;
    
    // Export final ONNX (ML-Agents style: same name as checkpoint)
    let final_onnx_path = cli.get_checkpoint_path().join(format!("{}-{}", behavior_name, final_step));
    sac_trainer.export_onnx(final_onnx_path.to_str().unwrap())?;
    
    println!("✅ Final checkpoint saved: {}.pt", final_path.display());
    println!("✅ Final ONNX exported: {}.onnx", final_onnx_path.display());
    
    env.close().await?;
    
    Ok(())
}

async fn connect_to_unity(
    unity_path: &PathBuf,
    base_port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::process::{Command, Stdio};
    use tokio::time::{sleep, Duration};
    use tokio::net::TcpStream;
    
    println!("📂 Unity path: {}", unity_path.display());
    
    // Check if file exists
    if !unity_path.exists() {
        return Err(format!("Unity executable not found: {}", unity_path.display()).into());
    }
    
    // Handle macOS .app bundles
    let exec_path = if unity_path.extension().and_then(|s| s.to_str()) == Some("app") {
        let macos_dir = unity_path.join("Contents/MacOS");
        println!("📱 Detected macOS app bundle");
        
        // Try to find the executable in MacOS directory
        let mut found_exec: Option<PathBuf> = None;
        if macos_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&macos_dir) {
                // Find the first executable file
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
    
    // Launch Unity process
    println!("🚀 Launching Unity environment...");
    println!("   Command: {}", exec_path.display());
    let mut child = Command::new(&exec_path)
        .arg("--mlagents-port")
        .arg(base_port.to_string())
        .arg("-logFile")
        .arg("-")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    
    println!("⏳ Waiting for Unity to start (port {})...", base_port);
    
    // Try to connect for up to 60 seconds
    let mut attempts = 0;
    let max_attempts = 60;
    
    loop {
        attempts += 1;
        
        // Try to connect
        match TcpStream::connect(format!("127.0.0.1:{}", base_port)).await {
            Ok(_stream) => {
                println!("✅ Connection established on port {}!", base_port);
                return Ok(());
            }
            Err(_) if attempts < max_attempts => {
                if attempts % 10 == 0 {
                    println!("   Still waiting... (attempt {}/{})", attempts, max_attempts);
                }
                
                // Check if process is still alive
                match child.try_wait() {
                    Ok(Some(status)) => {
                        return Err(format!("Unity process exited early with status: {}", status).into());
                    }
                    Ok(None) => {
                        // Process still running, continue waiting
                    }
                    Err(e) => {
                        return Err(format!("Error checking Unity process: {}", e).into());
                    }
                }
                
                sleep(Duration::from_secs(1)).await;
            }
            Err(e) => {
                child.kill().ok();
                return Err(format!("Failed to connect after {} attempts: {}", attempts, e).into());
            }
        }
    }
}
