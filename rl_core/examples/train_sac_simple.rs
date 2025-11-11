// Simple SAC training example with TensorBoard logging
use rl_core::trainers::sac::{SACConfig, SACTrainer, Transition};
use rl_core::trainers::TensorBoardWriter;
use tch::{Device, Kind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SAC Training Example");
    
    // Configuration
    let obs_dim = 8;
    let action_dim = 2;
    let device = Device::cuda_if_available();
    
    let config = SACConfig {
        buffer_size: 100_000,
        lr_actor: 3e-4,
        lr_critic: 3e-4,
        lr_alpha: 3e-4,
        gamma: 0.99,
        tau: 0.005,
        init_alpha: 0.2,
        auto_alpha: true,
        target_entropy: None,
        batch_size: 256,
        warmup_steps: 1000,
        gradient_steps: 1,
        target_update_interval: 1,
        checkpoint_interval: 1000,
        save_onnx: true,
        hidden_layers: vec![256, 256],
        dtype: Kind::Float,
    };
    
    // Create trainer
    println!("Creating SAC trainer...");
    let mut trainer = SACTrainer::new(obs_dim, action_dim, config, device)?;
    
    // Create TensorBoard writer
    let tensorboard = TensorBoardWriter::new("runs", "sac_simple_example")?;
    
    // Log hyperparameters
    tensorboard.log_hyperparams(&serde_json::json!({
        "algorithm": "SAC",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "lr_actor": trainer.config.lr_actor,
        "lr_critic": trainer.config.lr_critic,
        "gamma": trainer.config.gamma,
        "batch_size": trainer.config.batch_size,
    }))?;
    
    // Simulate training with random data
    let num_episodes = 100;
    let max_steps_per_episode = 100;
    
    println!("Starting training for {} episodes...", num_episodes);
    
    for episode in 0..num_episodes {
        let mut episode_reward = 0.0;
        
        for step in 0..max_steps_per_episode {
            // Generate random observation and action (dummy environment)
            let obs = vec![0.0f32; obs_dim as usize];
            let action = vec![0.0f32; action_dim as usize];
            let reward = rand::random::<f32>() - 0.5; // Random reward [-0.5, 0.5]
            let next_obs = vec![0.0f32; obs_dim as usize];
            let done = step == max_steps_per_episode - 1;
            
            // Store transition
            let transition = Transition {
                obs,
                action,
                reward,
                next_obs,
                done,
            };
            trainer.store_transition(transition);
            
            episode_reward += reward;
            
            // Update trainer
            if let Some(metrics) = trainer.update() {
                let global_step = trainer.get_step();
                
                // Log metrics to TensorBoard
                tensorboard.add_scalar("loss/actor", metrics.actor_loss, global_step)?;
                tensorboard.add_scalar("loss/critic", metrics.critic_loss, global_step)?;
                tensorboard.add_scalar("loss/alpha", metrics.alpha_loss, global_step)?;
                tensorboard.add_scalar("values/q1", metrics.q1_value, global_step)?;
                tensorboard.add_scalar("values/q2", metrics.q2_value, global_step)?;
                tensorboard.add_scalar("values/alpha", metrics.alpha, global_step)?;
                
                if global_step % 100 == 0 {
                    println!(
                        "Step {}: Actor Loss: {:.4}, Critic Loss: {:.4}, Alpha: {:.4}",
                        global_step, metrics.actor_loss, metrics.critic_loss, metrics.alpha
                    );
                }
            }
        }
        
        // Log episode reward
        tensorboard.add_scalar("episode/reward", episode_reward as f64, episode as i64)?;
        tensorboard.add_scalar("episode/avg_reward", trainer.get_avg_reward() as f64, episode as i64)?;
        
        println!(
            "Episode {}: Reward: {:.2}, Avg Reward: {:.2}",
            episode, episode_reward, trainer.get_avg_reward()
        );
        
        // Save checkpoint every 10 episodes
        if episode % 10 == 0 && episode > 0 {
            let checkpoint_path = format!("checkpoints/sac_episode_{}", episode);
            trainer.save_checkpoint(&checkpoint_path)?;
            println!("✓ Checkpoint saved: {}", checkpoint_path);
        }
    }
    
    // Save final model
    println!("\n🎉 Training completed!");
    trainer.save_checkpoint("checkpoints/sac_final")?;
    trainer.export_onnx("models/sac_policy")?;
    
    println!("\n📊 View TensorBoard logs with:");
    println!("   tensorboard --logdir=runs");
    
    Ok(())
}
