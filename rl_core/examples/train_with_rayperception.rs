// Exemplo de treinamento com detecção automática de RayPerception
use rl_core::trainers::sac::{SACTrainer, SACConfig, ObservationSpec};
use tch::Device;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 SAC Training with Unity - RayPerception Auto-Detection");
    println!("══════════════════════════════════════════════════════════\n");
    
    // Configuração
    let config = SACConfig {
        hidden_layers: vec![256, 256],
        checkpoint_interval: 10000,
        save_onnx: true,
        ..Default::default()
    };
    
    let device = Device::cuda_if_available();
    println!("🖥️  Device: {:?}\n", device);
    
    // Nota: Dimensões serão detectadas automaticamente do Unity
    println!("🔍 Waiting for Unity connection...");
    println!("   (Start Unity with your ML-Agents scene)\n");
    
    // Simular primeira observação do Unity
    // Em produção, isso virá da conexão real
    println!("📡 Simulating Unity observations for demo...\n");
    
    // Exemplo 1: Apenas Vector Observations
    let vector_only = vec![vec![0.5; 62]];
    let spec1 = ObservationSpec::detect_from_observations(&vector_only);
    spec1.print_info();
    
    println!("\n{}", "─".repeat(64));
    println!("\n🔄 Now simulating with RayPerception...\n");
    
    // Exemplo 2: Vector + RayPerception
    let vector_with_ray = vec![
        vec![0.5; 62],   // Vector observations
        vec![0.3; 100],  // RayPerception sensor
    ];
    let spec2 = ObservationSpec::detect_from_observations(&vector_with_ray);
    spec2.print_info();
    
    // Criar modelo com dimensões corretas
    let obs_dim = spec2.total_obs_size as i64;
    let action_dim = 2i64;
    
    println!("🤖 Creating SAC model with detected dimensions...");
    println!("   └─ obs_dim: {}", obs_dim);
    println!("   └─ action_dim: {}", action_dim);
    println!("   └─ hidden_dim: {:?}", config.hidden_layers);
    
    let mut trainer = SACTrainer::new(
        obs_dim,
        action_dim,
        config.clone(),
        device,
    )?;
    
    println!("\n✅ Model created successfully!");
    println!("\n💡 In production, connect to Unity:");
    println!("   1. Start Unity with ML-Agents scene");
    println!("   2. Run: cargo run --example train_with_unity");
    println!("   3. System will auto-detect sensors");
    println!("   4. Training starts with correct dimensions\n");
    
    // Salvar checkpoint de exemplo
    println!("💾 Saving example checkpoint...");
    trainer.save_checkpoint("results/rayperception_example.pt")?;
    
    println!("✅ Checkpoint saved!");
    println!("   └─ Model: results/rayperception_example.pt");
    println!("   └─ Metadata: results/metadata.json");
    println!("      (Contains obs_dim={}, action_dim={})\n", obs_dim, action_dim);
    
    // Exportar ONNX
    if config.save_onnx {
        println!("📦 Exporting ONNX...");
        trainer.export_onnx("results/rayperception_example")?;
        println!("✅ ONNX exported!");
        println!("   └─ File: results/rayperception_example.onnx");
        println!("   └─ Input shape: [batch, {}]", obs_dim);
        println!("   └─ Output shape: [batch, {}]\n", action_dim);
    }
    
    println!("🎉 Demo completed!");
    println!("\n📚 Next steps:");
    println!("   1. Check RAYPERCEPTION_DETECTION.md for details");
    println!("   2. Configure your Unity Agent");
    println!("   3. Start real training\n");
    
    Ok(())
}
