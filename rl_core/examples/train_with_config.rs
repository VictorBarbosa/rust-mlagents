// Training example using configuration from YAML with checkpoint and summary features
use rl_core::trainers::sac::{SACConfig, SACTrainer, Transition};
use rl_core::trainers::{CheckpointManager};
use tch::{Device, Kind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Training with Configuration Example");
    println!("====================================\n");

    // Create example configuration (in real scenario, this would come from YAML)
    let obs_dim = 8;
    let action_dim = 2;
    let device = Device::cuda_if_available();

    // Initialize SAC config with checkpoint and summary parameters
    let mut config = SACConfig::default();
    config.checkpoint_interval = 500; // Save checkpoint every 500 steps
    config.save_onnx = true;

    let mut trainer = SACTrainer::new(obs_dim, action_dim, config, device)?;
    
    // Create checkpoint manager
    let mut checkpoint_manager = CheckpointManager::new("checkpoints".to_string(), 5); // Keep 5 checkpoints

    // Simulated training loop with max_steps and summary_freq
    println!("Starting training loop...");
    
    let max_steps = 2000; // Use max_steps parameter
    let summary_freq = 200; // Use summary_freq parameter
    
    for step in 0..max_steps {
        // Generate dummy transition (in real scenario, this comes from environment)
        let obs: Vec<f32> = (0..obs_dim as usize).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        let action: Vec<f32> = (0..action_dim as usize).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        let reward = rand::random::<f32>() - 0.5; // Random reward
        let next_obs: Vec<f32> = (0..obs_dim as usize).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        let done = step % 200 == 199; // Simulate episode done every 200 steps

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
            // Check if we should save a checkpoint
            if trainer.should_checkpoint() {
                println!("🎯 Checkpoint triggered at step {}", trainer.get_step());
                checkpoint_manager.save_checkpoint(&trainer, trainer.get_step() as u64, "SimpleEnvironment")?;
            }
            
            // Print summary at summary_freq intervals
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
        
        // Check if we've reached max_steps
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