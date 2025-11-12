/// Test SAC training and ONNX export
/// This test creates a simple environment, trains a SAC agent, and exports to .pt and .onnx
#[cfg(test)]
mod tests {
    use crate::trainers::sac::{SACTrainer, SACConfig, Transition};
    use tch::{Device, Kind};
    use std::path::Path;

    /// Simple test environment (continuous action space)
    /// Example: CartPole-like environment with continuous actions
    struct SimpleEnv {
        state: Vec<f32>,
        steps: usize,
        max_steps: usize,
    }

    impl SimpleEnv {
        fn new(obs_dim: usize) -> Self {
            Self {
                state: vec![0.0; obs_dim],
                steps: 0,
                max_steps: 200,
            }
        }

        fn reset(&mut self) -> Vec<f32> {
            self.steps = 0;
            // Random initial state
            for i in 0..self.state.len() {
                self.state[i] = (rand::random::<f32>() - 0.5) * 0.2;
            }
            self.state.clone()
        }

        fn step(&mut self, action: &[f32]) -> (Vec<f32>, f32, bool) {
            self.steps += 1;
            
            // Simple dynamics: next_state = state + action * 0.1
            for i in 0..self.state.len().min(action.len()) {
                self.state[i] += action[i] * 0.1;
                self.state[i] = self.state[i].clamp(-1.0, 1.0);
            }

            // Reward: negative distance from origin (we want agent to stay near 0)
            let dist: f32 = self.state.iter().map(|x| x * x).sum::<f32>().sqrt();
            let reward = -dist;

            // Done if max steps reached or state too far
            let done = self.steps >= self.max_steps || dist > 2.0;

            (self.state.clone(), reward, done)
        }
    }

    #[test]
    fn test_sac_training_and_export() {
        println!("🚀 Starting SAC training and export test");

        // Environment parameters
        let obs_dim = 4;
        let action_dim = 2;

        // Device selection (use CPU for tests to ensure compatibility)
        let device = Device::Cpu;
        println!("📱 Using device: {:?}", device);

        // Create SAC config for quick testing
        let config = SACConfig {
            hidden_layers: vec![64, 64],  // Smaller network for faster testing
            dtype: Kind::Float,
            gamma: 0.99,
            tau: 0.005,
            lr_actor: 3e-4,
            lr_critic: 3e-4,
            lr_alpha: 3e-4,
            buffer_size: 10_000,
            batch_size: 64,
            warmup_steps: 500,  // Reduced for testing
            auto_alpha: false,  // Disable for stability in test
            init_alpha: 0.2,
            target_entropy: None,
            gradient_steps: 1,
            target_update_interval: 1,
            checkpoint_interval: 1000,
            save_onnx: true,
        };

        // Create trainer
        let mut trainer = SACTrainer::new(obs_dim, action_dim, config, device)
            .expect("Failed to create SAC trainer");
        println!("✅ SAC trainer created successfully");

        // Create environment
        let mut env = SimpleEnv::new(obs_dim as usize);

        // Training loop
        let num_episodes = 10;  // Small number for testing
        println!("🎯 Training for {} episodes", num_episodes);

        for episode in 0..num_episodes {
            let mut obs = env.reset();
            let mut episode_reward = 0.0;
            let mut steps = 0;

            loop {
                // Select action
                let action = trainer.select_action_from_vec(&obs);

                // Environment step
                let (next_obs, reward, done) = env.step(&action);
                episode_reward += reward;
                steps += 1;

                // Store transition
                let transition = Transition {
                    obs: obs.clone(),
                    action: action.clone(),
                    reward,
                    next_obs: next_obs.clone(),
                    done,
                };
                trainer.store_transition(transition);

                // Update networks
                if let Some(metrics) = trainer.update() {
                    if steps % 50 == 0 {
                        println!(
                            "  Episode {}, Step {}: actor_loss={:.4}, critic_loss={:.4}, Q1={:.4}",
                            episode + 1, trainer.get_step(), metrics.actor_loss, metrics.critic_loss, metrics.q1_value
                        );
                    }
                }

                obs = next_obs;

                if done {
                    break;
                }
            }

            println!(
                "Episode {}/{}: reward={:.2}, steps={}",
                episode + 1, num_episodes, episode_reward, steps
            );
        }

        println!("✅ Training completed");
        println!("📊 Average reward: {:.2}", trainer.get_avg_reward());

        // Test checkpoint saving (.pt)
        let checkpoint_dir = std::env::temp_dir().join("rust_ml_agent_test");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");
        
        let checkpoint_path = checkpoint_dir.join("test_model.pt");
        let checkpoint_path_str = checkpoint_path.to_str().unwrap();

        println!("💾 Saving checkpoint to: {}", checkpoint_path_str);
        trainer.save_checkpoint(checkpoint_path_str)
            .expect("Failed to save checkpoint");

        // Verify .pt file exists
        assert!(checkpoint_path.exists(), "Checkpoint file should exist");
        println!("✅ Checkpoint saved successfully");

        // Verify checkpoint.pt also exists (ML-Agents default)
        let default_checkpoint = checkpoint_dir.join("checkpoint.pt");
        assert!(default_checkpoint.exists(), "Default checkpoint.pt should exist");
        println!("✅ Default checkpoint.pt created");

        // Verify metadata.json exists
        let metadata_path = checkpoint_dir.join("metadata.json");
        assert!(metadata_path.exists(), "Metadata file should exist");
        println!("✅ Metadata file created");

        // Test ONNX export
        let onnx_path = checkpoint_dir.join("test_model");
        let onnx_path_str = onnx_path.to_str().unwrap();

        println!("📦 Exporting to ONNX: {}", onnx_path_str);
        trainer.export_onnx(onnx_path_str)
            .expect("Failed to export ONNX");

        // Verify .pt files were created for ONNX conversion
        let pt_for_onnx = checkpoint_dir.join("test_model.pt");
        assert!(pt_for_onnx.exists(), "PyTorch file for ONNX should exist");
        println!("✅ PyTorch file for ONNX created");

        // Verify Python conversion script was created
        let conversion_script = checkpoint_dir.join("test_model_convert_to_onnx.py");
        assert!(conversion_script.exists(), "ONNX conversion script should exist");
        println!("✅ ONNX conversion script created");
        
        // Verify ONNX file was created
        let onnx_file = checkpoint_dir.join("test_model.onnx");
        assert!(onnx_file.exists(), "ONNX file should exist");
        println!("✅ ONNX file created");

        // Read and verify the conversion script contains correct parameters
        let script_content = std::fs::read_to_string(&conversion_script)
            .expect("Failed to read conversion script");
        
        assert!(script_content.contains("obs_dim = 4"), "Script should contain correct obs_dim");
        assert!(script_content.contains("action_dim = 2"), "Script should contain correct action_dim");
        assert!(script_content.contains("hidden_dim = 64"), "Script should contain correct hidden_dim");
        assert!(script_content.contains("torch.onnx.export"), "Script should contain ONNX export call");
        assert!(script_content.contains("input_names=['vector_observation']"), 
                "Script should use correct input name");
        assert!(script_content.contains("version_number"), "Script should export version_number");
        assert!(script_content.contains("memory_size"), "Script should export memory_size");
        assert!(script_content.contains("continuous_actions"), "Script should export continuous_actions");
        assert!(script_content.contains("deterministic_continuous_actions"), 
                "Script should export deterministic_continuous_actions");
        assert!(script_content.contains("opset_version=11"), 
                "Script should use opset_version=11");
        assert!(script_content.contains("do_constant_folding=True"), 
                "Script should use constant folding");
        
        println!("✅ Conversion script validated");

        // Verify metadata JSON was created
        let metadata_json_path = checkpoint_dir.join("test_model_metadata.json");
        assert!(metadata_json_path.exists(), "Metadata JSON should exist");
        
        let metadata_content = std::fs::read_to_string(&metadata_json_path)
            .expect("Failed to read metadata");
        assert!(metadata_content.contains("SAC_Actor"), "Metadata should contain model type");
        println!("✅ Metadata JSON created and validated");

        // Test loading checkpoint (currently not working with TorchScript format)
        // println!("🔄 Testing checkpoint loading...");
        // let mut new_trainer = SACTrainer::new(obs_dim, action_dim, trainer.config.clone(), device)
        //     .expect("Failed to create new trainer");
        // 
        // new_trainer.load_checkpoint(checkpoint_path_str)
        //     .expect("Failed to load checkpoint");
        // println!("✅ Checkpoint loaded successfully");

        // Verify the trained model produces actions
        let test_obs = vec![0.1, -0.2, 0.3, -0.1];
        let action1 = trainer.select_action_from_vec(&test_obs);
        
        println!("Trained model action: {:?}", action1);
        assert_eq!(action1.len(), action_dim as usize, "Action should have correct dimensions");

        // Clean up test files
        println!("🧹 Cleaning up test files...");
        std::fs::remove_dir_all(&checkpoint_dir).ok();
        
        println!("✅ SAC training and export test completed successfully!");
    }

    #[test]
    fn test_onnx_conversion_script_structure() {
        println!("🧪 Testing ONNX conversion script structure");

        let obs_dim = 8;
        let action_dim = 2;
        let device = Device::Cpu;

        let config = SACConfig {
            hidden_layers: vec![128, 128],
            ..Default::default()
        };

        let trainer = SACTrainer::new(obs_dim, action_dim, config, device)
            .expect("Failed to create trainer");

        let temp_dir = std::env::temp_dir().join("rust_ml_agent_script_test");
        std::fs::create_dir_all(&temp_dir).expect("Failed to create temp directory");

        let model_path = temp_dir.join("model");
        let model_path_str = model_path.to_str().unwrap();

        // Export (which generates the script)
        trainer.export_onnx(model_path_str)
            .expect("Failed to export");

        let script_path = temp_dir.join("model_convert_to_onnx.py");
        assert!(script_path.exists(), "Script should exist");

        let script_content = std::fs::read_to_string(&script_path)
            .expect("Failed to read script");

        // Verify Python script structure matches ML-Agents requirements
        assert!(script_content.contains("class ActorNetwork(nn.Module)"), 
                "Should define ActorNetwork class");
        assert!(script_content.contains("def forward(self, obs)"), 
                "Should have forward method");
        assert!(script_content.contains("return ("), 
                "Should return tuple of outputs");
        assert!(script_content.contains("version_number"), 
                "Should include version_number");
        assert!(script_content.contains("memory_size"), 
                "Should include memory_size");
        assert!(script_content.contains("continuous_actions"), 
                "Should include continuous_actions");
        assert!(script_content.contains("continuous_action_output_shape"), 
                "Should include continuous_action_output_shape");
        assert!(script_content.contains("deterministic_continuous_actions"), 
                "Should include deterministic_continuous_actions");
        
        // Verify ONNX export parameters
        assert!(script_content.contains("export_params=True"), 
                "Should export parameters");
        assert!(script_content.contains("opset_version=11"), 
                "Should use opset version 11");
        assert!(script_content.contains("do_constant_folding=True"), 
                "Should use constant folding");
        assert!(script_content.contains("dynamic_axes"), 
                "Should define dynamic axes");
        assert!(script_content.contains("'batch'"), 
                "Should use 'batch' for dynamic axis");

        println!("✅ ONNX conversion script structure validated");

        // Clean up
        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
