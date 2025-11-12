// Training example using configuration from YAML with checkpoint and summary features
use rl_core::trainers::sac::{SACConfig, SACTrainer, Transition};
use rl_core::trainers::{RunOptions, CheckpointManager};
use tch::{Device, Kind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Training with YAML Configuration Example");
    println!("==========================================\n");

    // Load configuration from YAML file
    let options = RunOptions::from_yaml("config_example.yaml")?;
    
    println!("Configuration loaded:");
    println!("- Max Steps: {}", options.behaviors.get("SimpleEnvironment").map(|b| b.max_steps).unwrap_or(0));
    println!("- Summary Frequency: {}", options.behaviors.get("SimpleEnvironment").map(|b| b.summary_freq).unwrap_or(0));
    println!("- Checkpoint Interval: {}", options.behaviors.get("SimpleEnvironment").map(|b| b.checkpoint_interval).unwrap_or(0));
    println!("- Keep Checkpoints: {}", options.behaviors.get("SimpleEnvironment").map(|b| b.keep_checkpoints).unwrap_or(0));
    println!("- Time Horizon: {}", options.behaviors.get("SimpleEnvironment").map(|b| b.time_horizon).unwrap_or(0));

    // Extract parameters from configuration
    let behavior_settings = options.behaviors.get("SimpleEnvironment").unwrap();
    let obs_dim = 8; // This would normally come from environment info
    let action_dim = 2; // This would normally come from environment info
    let device = Device::Cpu; // Use CPU for this example

    // Initialize SAC config using parameters from YAML
    let mut config = SACConfig::default();
    config.checkpoint_interval = behavior_settings.checkpoint_interval as usize; // Convert from u64 to usize
    config.save_onnx = true;

    match &behavior_settings.hyperparameters {
        crate::trainers::settings::HyperparameterSettings::SAC(sac_params) => {
            config.batch_size = sac_params.batch_size;
            config.buffer_size = sac_params.buffer_size;
            config.lr_actor = sac_params.learning_rate as f64;
            config.lr_critic = sac_params.learning_rate as f64;
            config.lr_alpha = sac_params.learning_rate as f64;
            config.gamma = sac_params.gamma as f64;
            config.tau = sac_params.tau as f64;
            config.init_alpha = sac_params.init_entcoef as f64;
        },
        _ => {
            println!("Warning: Not using SAC hyperparameters");
        }
    }

    match &behavior_settings.network_settings {
        network_settings => {
            config.hidden_layers = vec![network_settings.hidden_units as i64; network_settings.num_layers];
        }
    }

    let mut trainer = SACTrainer::new(obs_dim, action_dim, config, device)?;
    
    // Create checkpoint manager with keep_checkpoints parameter from config
    let mut checkpoint_manager = CheckpointManager::new("checkpoints".to_string(), behavior_settings.keep_checkpoints);

    // Simulated training loop with parameters from YAML
    println!("\nStarting training loop with configuration parameters...");
    
    let max_steps = behavior_settings.max_steps; // Get max_steps from config
    let summary_freq = behavior_settings.summary_freq; // Get summary_freq from config
    
    for step in 0..max_steps {
        // Generate dummy transition (in real scenario, this comes from environment)
        let obs: Vec<f32> = (0..obs_dim as usize).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        let action: Vec<f32> = (0..action_dim as usize).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        let reward = rand::random::<f32>() - 0.5; // Random reward
        let next_obs: Vec<f32> = (0..obs_dim as usize).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        let done = step % (behavior_settings.time_horizon as u64) == (behavior_settings.time_horizon as u64 - 1); // Use time_horizon from config

        let transition = Transition {
            obs,
            action,
            reward,
            next_obs,
            done,
        };
        
        trainer.store_transition(transition);

        // Update trainer
        if let Some(metrics) = trainer.update() {
            // Check if we should save a checkpoint using checkpoint_interval from config
            if trainer.should_checkpoint() {
                println!("🎯 Checkpoint triggered at step {}", trainer.get_step());
                checkpoint_manager.save_checkpoint(&trainer, trainer.get_step() as u64, "SimpleEnvironment")?;
            }
            
            // Print summary at summary_freq intervals from config
            if trainer.get_step() % summary_freq as i64 == 0 {
                println!(
                    "Step {}: Actor Loss: {:.4}, Critic Loss: {:.4}, Alpha: {:.4}, Avg Reward: {:.2}",
                    trainer.get_step(),
                    metrics.actor_loss,
                    metrics.critic_loss,
                    metrics.alpha,
                    trainer.get_avg_reward()
                );
            }
        }
        
        // Check if we've reached max_steps from config
        if trainer.get_step() >= max_steps as i64 {
            println!("Reached max_steps={}, stopping training", max_steps);
            break;
        }
    }

    // Save final model
    println!("\n💾 Saving final model...");
    trainer.save_checkpoint("checkpoints/final_model.pt")?;
    if trainer.config.save_onnx {
        trainer.export_onnx("models/final_policy")?;
    }

    println!("\n✅ Training completed!");

    Ok(())
}