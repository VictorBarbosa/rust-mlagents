// Example: Train SAC with Unity Environment
use rl_core::trainers::sac::{SACTrainer, SACConfig, UnityTrainer};
use tch::Device;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 SAC Training with Unity ML-Agents");
    println!("=====================================\n");
    
    // Configuration
    let port = 5005;
    let obs_dim = 8;      // Will be determined from Unity
    let action_dim = 2;   // Will be determined from Unity
    let max_steps = 100_000;
    
    // Create SAC config
    let mut config = SACConfig::default();
    config.hidden_layers = vec![256, 256];
    config.batch_size = 256;
    config.warmup_steps = 1000;
    config.buffer_size = 100_000;
    config.checkpoint_interval = 10000;
    
    println!("📝 SAC Configuration:");
    println!("   Hidden layers: {:?}", config.hidden_layers);
    println!("   Batch size: {}", config.batch_size);
    println!("   Buffer size: {}", config.buffer_size);
    println!("   Gamma: {}", config.gamma);
    println!("   Tau: {}", config.tau);
    println!("");
    
    // Create device
    let device = Device::cuda_if_available();
    println!("🖥️  Using device: {:?}\n", device);
    
    // Create SAC trainer
    println!("🧠 Creating SAC trainer...");
    let sac_trainer = SACTrainer::new(obs_dim, action_dim, config, device)?;
    
    // Create Unity trainer
    println!("🔌 Connecting to Unity on port {}...", port);
    let mut unity_trainer = UnityTrainer::new(port, sac_trainer, max_steps).await?;
    
    println!("✅ Connected!\n");
    
    // Start training
    println!("🚀 Starting training...\n");
    unity_trainer.train().await?;
    
    println!("\n✅ Training completed successfully!");
    
    Ok(())
}
