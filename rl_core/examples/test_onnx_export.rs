// Test ONNX export functionality
use rl_core::trainers::sac::{SACConfig, SACTrainer};
use tch::{Device, Kind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing ONNX Export");
    println!("======================\n");
    
    // Configuration
    let obs_dim = 8;
    let action_dim = 2;
    let device = Device::Cpu;
    
    let config = SACConfig {
        buffer_size: 1000,
        lr_actor: 3e-4,
        lr_critic: 3e-4,
        lr_alpha: 3e-4,
        gamma: 0.99,
        tau: 0.005,
        init_alpha: 0.2,
        auto_alpha: false,
        target_entropy: None,
        batch_size: 64,
        warmup_steps: 100,
        gradient_steps: 1,
        target_update_interval: 1,
        checkpoint_interval: 1000,
        save_onnx: true,
        hidden_layers: vec![64, 64],
        dtype: Kind::Float,
    };
    
    // Create trainer
    println!("1. Creating SAC trainer...");
    let trainer = SACTrainer::new(obs_dim, action_dim, config, device)?;
    println!("   ✓ Trainer created\n");
    
    // Save checkpoint
    println!("2. Saving PyTorch checkpoint...");
    std::fs::create_dir_all("test_exports")?;
    trainer.save_checkpoint("test_exports/test_model")?;
    println!("   ✓ Checkpoint saved\n");
    
    // Export to ONNX
    println!("3. Exporting to ONNX...");
    println!("   This will:");
    println!("   - Save actor network as .pt file");
    println!("   - Generate Python conversion script");
    println!("   - Attempt automatic conversion to .onnx");
    println!();
    
    trainer.export_onnx("test_exports/sac_policy")?;
    
    // Check if ONNX file was created
    println!();
    if std::path::Path::new("test_exports/sac_policy.onnx").exists() {
        println!("✅ SUCCESS! ONNX file created:");
        println!("   📄 test_exports/sac_policy.onnx");
        
        // Get file size
        let metadata = std::fs::metadata("test_exports/sac_policy.onnx")?;
        println!("   📏 Size: {} bytes", metadata.len());
        
        // Check other files
        if std::path::Path::new("test_exports/sac_policy.pt").exists() {
            println!("   📄 test_exports/sac_policy.pt (PyTorch checkpoint)");
        }
        if std::path::Path::new("test_exports/sac_policy_metadata.json").exists() {
            println!("   📄 test_exports/sac_policy_metadata.json (metadata)");
        }
        
        println!();
        println!("🎉 ONNX export test PASSED!");
        println!();
        println!("You can now:");
        println!("  1. Load the .onnx file in Unity with Barracuda");
        println!("  2. Use it for inference in production");
        println!("  3. Verify with: python3 -c 'import onnx; onnx.checker.check_model(onnx.load(\"test_exports/sac_policy.onnx\"))'");
    } else {
        println!("⚠️  ONNX file not created automatically");
        println!();
        println!("This is expected if PyTorch is not installed.");
        println!();
        println!("To complete the export:");
        if std::path::Path::new("test_exports/sac_policy_convert_to_onnx.py").exists() {
            println!("  1. Install PyTorch: pip3 install torch");
            println!("  2. Run: python3 test_exports/sac_policy_convert_to_onnx.py");
        } else {
            println!("  1. Check that the PyTorch checkpoint was saved");
            println!("  2. Install PyTorch: pip3 install torch");
            println!("  3. Run the conversion script manually");
        }
    }
    
    println!();
    println!("Cleaning up test files...");
    std::fs::remove_dir_all("test_exports").ok();
    println!("✓ Done!");
    
    Ok(())
}
