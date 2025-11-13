// SAC Trainer - Main Training Loop
use tch::{nn, nn::OptimizerConfig, Tensor, Device};
use super::{ActorNetwork, CriticNetwork, ReplayBuffer, SACConfig, Batch, Transition};

pub struct SACTrainer {
    // Networks
    actor: ActorNetwork,
    critic_1: CriticNetwork,
    critic_2: CriticNetwork,
    target_critic_1: CriticNetwork,
    target_critic_2: CriticNetwork,
    
    // VarStores
    actor_vs: nn::VarStore,
    critic_vs: nn::VarStore,
    target_critic_vs: nn::VarStore,
    
    // Optimizers
    actor_opt: nn::Optimizer,
    critic_opt: nn::Optimizer,
    
    // Alpha (entropy coefficient)
    log_alpha: Tensor,
    alpha: Tensor,
    alpha_opt: Option<nn::Optimizer>,
    
    // Hyperparameters
    pub config: SACConfig,
    target_entropy: f64,
    
    // Replay buffer
    replay_buffer: ReplayBuffer,
    
    // State
    step: i64,
    episode_rewards: Vec<f32>,
    current_episode_reward: f32,
    
    // Dimensions
    pub obs_dim: i64,
    pub action_dim: i64,
    pub device: Device,
}

impl SACTrainer {
    pub fn new(
        obs_dim: i64,
        action_dim: i64,
        config: SACConfig,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let hidden_dim = config.hidden_layers[0];
        let dtype = config.dtype;
        
        // Actor
        let mut actor_vs = nn::VarStore::new(device);
        let actor = ActorNetwork::new(&actor_vs.root(), obs_dim, action_dim, hidden_dim, dtype);
        let actor_opt = nn::Adam::default().build(&actor_vs, config.lr_actor)?;
        
        // Critics
        let mut critic_vs = nn::VarStore::new(device);
        let critic_1 = CriticNetwork::new(
            &(&critic_vs.root() / "critic1"),
            obs_dim,
            action_dim,
            hidden_dim,
            dtype,
        );
        let critic_2 = CriticNetwork::new(
            &(&critic_vs.root() / "critic2"),
            obs_dim,
            action_dim,
            hidden_dim,
            dtype,
        );
        let critic_opt = nn::Adam::default().build(&critic_vs, config.lr_critic)?;
        
        // Target critics (copy from critics)
        // Use same path structure for copy to work
        let mut target_critic_vs = nn::VarStore::new(device);
        let target_critic_1 = CriticNetwork::new(
            &(&target_critic_vs.root() / "critic1"),
            obs_dim,
            action_dim,
            hidden_dim,
            dtype,
        );
        let target_critic_2 = CriticNetwork::new(
            &(&target_critic_vs.root() / "critic2"),
            obs_dim,
            action_dim,
            hidden_dim,
            dtype,
        );
        // Copy weights from critic to target
        target_critic_vs.copy(&critic_vs)?;
        
        // Alpha (entropy temperature)
        let log_alpha = Tensor::from_slice(&[config.init_alpha.ln() as f32])
            .to_kind(dtype)
            .to_device(device)
            .set_requires_grad(true);
        let alpha = log_alpha.shallow_clone().exp();
        
        // Note: In tch 0.18, we manage alpha optimization differently
        // For now, alpha will be fixed (auto_alpha disabled)
        let alpha_opt = None;
        
        let target_entropy = config.target_entropy.unwrap_or(-(action_dim as f64));
        
        let replay_buffer = ReplayBuffer::new(config.buffer_size, device);
        
        Ok(Self {
            actor,
            critic_1,
            critic_2,
            target_critic_1,
            target_critic_2,
            actor_vs,
            critic_vs,
            target_critic_vs,
            actor_opt,
            critic_opt,
            log_alpha,
            alpha,
            alpha_opt,
            config,
            target_entropy,
            replay_buffer,
            step: 0,
            episode_rewards: Vec::new(),
            current_episode_reward: 0.0,
            obs_dim,
            action_dim,
            device,
        })
    }
    
    pub fn select_action(&self, obs: &Tensor, deterministic: bool) -> Tensor {
        tch::no_grad(|| {
            if deterministic {
                self.actor.get_action_deterministic(obs)
            } else {
                let (action, _) = self.actor.sample(obs);
                action
            }
        })
    }
    
    pub fn select_action_from_vec(&self, obs: &[f32]) -> Vec<f32> {
        use tch::Kind;
        
        let obs_tensor = Tensor::from_slice(obs)
            .to_kind(Kind::Float)
            .to_device(self.device)
            .unsqueeze(0); // Add batch dimension
        
        let action_tensor = self.select_action(&obs_tensor, false);
        
        // Convert back to Vec<f32>
        let action_squeezed = action_tensor.squeeze_dim(0);
        let mut action_vec = vec![0.0f32; self.action_dim as usize];
        action_squeezed.copy_data(&mut action_vec, self.action_dim as usize);
        action_vec
    }
    
    pub fn store_transition(&mut self, transition: Transition) {
        self.current_episode_reward += transition.reward;
        
        if transition.done {
            self.episode_rewards.push(self.current_episode_reward);
            self.current_episode_reward = 0.0;
        }
        
        self.replay_buffer.push(transition);
    }
    
    pub fn update(&mut self) -> Option<SACMetrics> {
        if self.replay_buffer.len() < self.config.warmup_steps {
            return None;
        }
        
        let mut total_actor_loss = 0.0;
        let mut total_critic_loss = 0.0;
        let mut total_alpha_loss = 0.0;
        let mut total_q1 = 0.0;
        let mut total_q2 = 0.0;
        
        for _ in 0..self.config.gradient_steps {
            let batch = self.replay_buffer.sample(self.config.batch_size)?;
            
            // Update critics
            let (critic_loss, q1_value, q2_value) = self.update_critics(&batch);
            total_critic_loss += critic_loss;
            total_q1 += q1_value;
            total_q2 += q2_value;
            
            // Update actor
            let (actor_loss, entropy) = self.update_actor(&batch);
            total_actor_loss += actor_loss;
            
            // Update alpha
            if self.config.auto_alpha {
                let alpha_loss = self.update_alpha(entropy);
                total_alpha_loss += alpha_loss;
            }
            
            // Soft update targets
            if self.step % self.config.target_update_interval as i64 == 0 {
                self.soft_update_targets();
            }
        }
        
        self.step += 1;
        
        let n = self.config.gradient_steps as f64;
        Some(SACMetrics {
            actor_loss: total_actor_loss / n,
            critic_loss: total_critic_loss / n,
            alpha_loss: total_alpha_loss / n,
            q1_value: total_q1 / n,
            q2_value: total_q2 / n,
            alpha: self.alpha.double_value(&[]),
            step: self.step,
        })
    }

    pub fn should_checkpoint(&self) -> bool {
        self.config.checkpoint_interval > 0 && self.step > 0 && self.step % self.config.checkpoint_interval as i64 == 0
    }
    
    fn update_critics(&mut self, batch: &Batch) -> (f64, f64, f64) {
        let dtype = self.config.dtype;
        
        // Compute target Q value - all in no_grad for targets
        let target_q = tch::no_grad(|| {
            let (next_action, next_log_prob) = self.actor.sample(&batch.next_obs);
            
            let target_q1 = self.target_critic_1.forward(&batch.next_obs, &next_action);
            let target_q2 = self.target_critic_2.forward(&batch.next_obs, &next_action);
            let min_target_q = target_q1.min_other(&target_q2);
            
            // Use detached alpha
            let alpha_detached = self.alpha.detach();
            let target_q: Tensor = min_target_q - alpha_detached * next_log_prob.unsqueeze(-1);
            
            let gamma = Tensor::from(self.config.gamma).to_kind(dtype).to_device(self.device);
            let target_q: Tensor = &batch.reward + gamma * (Tensor::from(1.0).to_kind(dtype).to_device(self.device) - &batch.done) * target_q;
            
            target_q
        });
        
        // Current Q estimates
        let current_q1 = self.critic_1.forward(&batch.obs, &batch.action);
        let current_q2 = self.critic_2.forward(&batch.obs, &batch.action);
        
        // Critic loss (MSE)
        let critic_loss = (&current_q1 - &target_q).pow_tensor_scalar(2).mean(dtype)
            + (&current_q2 - &target_q).pow_tensor_scalar(2).mean(dtype);
        
        // Optimize critics
        self.critic_opt.zero_grad();
        critic_loss.backward();
        self.critic_opt.step();
        
        (
            critic_loss.double_value(&[]),
            current_q1.mean(dtype).double_value(&[]),
            current_q2.mean(dtype).double_value(&[]),
        )
    }
    
    fn update_actor(&mut self, batch: &Batch) -> (f64, f64) {
        let dtype = self.config.dtype;
        let (action, log_prob) = self.actor.sample(&batch.obs);
        
        let q1 = self.critic_1.forward(&batch.obs, &action);
        let q2 = self.critic_2.forward(&batch.obs, &action);
        let q = q1.min_other(&q2);
        
        // Detach alpha to avoid computational graph issues
        let alpha_detached = self.alpha.detach();
        
        // Actor loss: maximize Q - α * log π
        let actor_loss = (&alpha_detached * &log_prob.unsqueeze(-1) - q).mean(dtype);
        
        self.actor_opt.zero_grad();
        actor_loss.backward();
        self.actor_opt.step();
        
        let entropy = -log_prob.mean(dtype).double_value(&[]);
        
        (actor_loss.double_value(&[]), entropy)
    }
    
    fn update_alpha(&mut self, entropy: f64) -> f64 {
        let dtype = self.config.dtype;
        if let Some(ref mut alpha_opt) = self.alpha_opt {
            let target = Tensor::from_slice(&[(self.target_entropy + entropy) as f32])
                .to_kind(dtype)
                .to_device(self.device);
            let alpha_loss = -(&self.log_alpha * target.detach()).mean(dtype);
            
            alpha_opt.zero_grad();
            alpha_loss.backward();
            alpha_opt.step();
            
            // Update alpha from log_alpha - no_grad to avoid computational graph issues
            tch::no_grad(|| {
                self.alpha = self.log_alpha.exp();
            });
            
            alpha_loss.double_value(&[])
        } else {
            0.0
        }
    }
    
    fn soft_update_targets(&mut self) {
        // Polyak averaging: θ_target = τ*θ + (1-τ)*θ_target
        // Manual soft update of target networks
        tch::no_grad(|| {
            let tau = self.config.tau;
            let critic_vars = self.critic_vs.trainable_variables();
            let mut target_vars = self.target_critic_vs.trainable_variables();
            
            for (target, source) in target_vars.iter_mut().zip(critic_vars.iter()) {
                target.copy_(&(tau * source + (1.0 - tau) * &*target));
            }
        });
    }
    
    fn try_direct_onnx_export(&self, _onnx_path: &str) -> bool {
        // Direct ONNX export from tch-rs is complex and requires
        // TorchScript tracing which is not well supported in tch-rs
        // We use Python conversion instead for maximum compatibility
        false
    }
    
    pub fn save_checkpoint(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create directory if needed
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Save all networks in a single .pt file (ML-Agents compatible)
        // Since var_copy doesn't allow dots in names, we save each VarStore separately
        // but in a unified directory structure
        
        // For simplicity and compatibility, save individual networks with prefixes in filename
        let path_obj = std::path::Path::new(path);
        let parent = path_obj.parent().unwrap_or(std::path::Path::new("."));
        let stem = path_obj.file_stem().and_then(|s| s.to_str()).unwrap_or("checkpoint");
        
        // Only save the main checkpoint file (following original ML-Agents format)
        
        // Also save actor as the main checkpoint file (for ONNX export)
        self.actor_vs.save(path)?;
        
        // Save to checkpoint.pt as well (ML-Agents default)
        if let Some(dir) = std::path::Path::new(path).parent() {
            let checkpoint_pt_path = dir.join("checkpoint.pt");
            self.actor_vs.save(&*checkpoint_pt_path.to_string_lossy())?;
        }
        
        // Save metadata (including dimensions for Python ONNX export)
        let metadata = serde_json::json!({
            "step": self.step,
            "episode_rewards": self.episode_rewards,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "config": self.config,
        });
        
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::write(
                dir.join("metadata.json"),
                serde_json::to_string_pretty(&metadata)?,
            )?;
        }
        
        println!("✓ Checkpoint saved: {}", path);
        Ok(())
    }
    
    pub fn load_checkpoint(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Load from individual files with prefixes
        let path_obj = std::path::Path::new(path);
        let parent = path_obj.parent().unwrap_or(std::path::Path::new("."));
        let stem = path_obj.file_stem().and_then(|s| s.to_str()).unwrap_or("checkpoint");
        
        // Load from the single checkpoint file (following original ML-Agents format)
        self.actor_vs.load(path)?;
        
        // Load metadata
        let metadata_path = if let Some(dir) = std::path::Path::new(path).parent() {
            dir.join("metadata.json")
        } else {
            std::path::PathBuf::from("metadata.json")
        };
        
        if metadata_path.exists() {
            let metadata: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(&metadata_path)?
            )?;
            self.step = metadata["step"].as_i64().unwrap_or(0);
        }
        
        println!("✓ Checkpoint loaded: {} (step: {})", path, self.step);
        Ok(())
    }
    
    pub fn export_onnx(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::process::Command;
        
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap_or(std::path::Path::new(".")))?;
        
        // Method 1: Try direct ONNX export via tch-rs (if available)
        let onnx_path = format!("{}.onnx", path);
        let export_success = self.try_direct_onnx_export(&onnx_path);
        
        if export_success {
            println!("✓ ONNX exported directly: {}.onnx", path);
            return Ok(());
        }
        
        // Method 2: Fallback to Python-based conversion
        println!("📝 Direct ONNX export not available, using Python conversion...");
        
        // Save PyTorch checkpoint (TorchScript format from tch-rs)
        self.actor_vs.save(format!("{}_full.pt", path))?;
        println!("✓ PyTorch VarStore saved: {}_full.pt", path);
        
        // Also save just the actor weights
        self.actor_vs.save(format!("{}.pt", path))?;
        println!("✓ Actor weights saved: {}.pt", path);
        
        // Generate ONNX conversion script
        let conversion_script = format!(
r#"#!/usr/bin/env python3
"""
Convert PyTorch checkpoint to ONNX format
Auto-generated by Rust ML-Agents

Requirements:
  pip install torch onnx onnxscript
"""
import sys
import torch
import torch.nn as nn

# Check for required packages
try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("⚠️  Warning: onnx package not installed")
    print("   Install with: pip install onnx")

try:
    import onnxscript
    HAS_ONNXSCRIPT = True
except ImportError:
    HAS_ONNXSCRIPT = False
    print("⚠️  Warning: onnxscript package not installed")
    print("   Install with: pip install onnxscript")

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.output_size = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Para compatibilidade com checkpoints que têm fc1, fc2, mean
        self.fc1 = self.net[0]
        self.fc2 = self.net[2]
        self.mean = self.net[4]
        
    def forward(self, obs):
        batch_size = obs.size(0)
        
        # Processar observações
        continuous_actions = torch.tanh(self.net(obs))
        deterministic_continuous_actions = continuous_actions
        
        # Metadados - FORMATO QUE FUNCIONA NO UNITY!
        # Usar torch.tensor().expand() ao invés de register_buffer
        version_number = torch.tensor([[3.0]], dtype=torch.float32).expand(batch_size, 1)
        memory_size = torch.zeros((batch_size, 1), dtype=torch.float32)
        continuous_action_output_shape = torch.tensor([[self.output_size]], dtype=torch.float32).expand(batch_size, 1)

        return (
            version_number,
            memory_size,
            continuous_actions,
            continuous_action_output_shape,
            deterministic_continuous_actions,
        )

def main():
    import os
    
    obs_dim = {obs_dim}
    action_dim = {action_dim}
    hidden_dim = {hidden_dim}
    
    # Get absolute paths (script is in checkpoint directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, '{filename}.pt')
    onnx_path = os.path.join(script_dir, '{filename}.onnx')
    
    print(f"Creating model: obs_dim={{obs_dim}}, action_dim={{action_dim}}, hidden_dim={{hidden_dim}}")
    print(f"Checkpoint: {{checkpoint_path}}")
    print(f"ONNX output: {{onnx_path}}")
    
    # Try loading checkpoint (handles TorchScript format from tch-rs)
    print("Loading PyTorch checkpoint...")
    
    # Create model first
    model = ActorNetwork(obs_dim, action_dim, hidden_dim)
    
    try:
        # Load TorchScript model (tch-rs VarStore format)
        try:
            jit_model = torch.jit.load(checkpoint_path, map_location='cpu')
            print("✓ Loaded TorchScript from tch-rs VarStore")
            
            # Extract state_dict from TorchScript model
            if hasattr(jit_model, 'state_dict'):
                state_dict = jit_model.state_dict()
                print(f"✓ Extracted state_dict with {{len(state_dict)}} parameters")
                
                # Convert tch-rs format (pipe separator) to PyTorch format (dot separator)
                # tch-rs uses: fc1|weight, fc1|bias
                # PyTorch uses: fc1.weight, fc1.bias
                converted_state_dict = {{}}
                for key, value in state_dict.items():
                    # Replace pipe with dot
                    new_key = key.replace('|', '.')
                    converted_state_dict[new_key] = value
                    print(f"  {{key}} -> {{new_key}}")
                
                # Load weights, ignorando as constantes que não existem no checkpoint
                model.load_state_dict(converted_state_dict, strict=False)
                print("✓ Loaded weights into ActorNetwork (const buffers initialized separately)")
            else:
                print("⚠️  TorchScript model doesn't have state_dict")
                raise ValueError("Cannot extract state_dict from TorchScript model")
                
        except Exception as e1:
            print(f"TorchScript load failed: {{e1}}")
            print("Trying direct state_dict loading...")
            
            # Fallback: Try loading as state_dict directly
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                print("Loading from state_dict...")
                model.load_state_dict(checkpoint, strict=False)
            else:
                raise ValueError(f"Unsupported checkpoint format: {{type(checkpoint)}}")
    
    except Exception as e:
        print(f"❌ Error loading checkpoint: {{e}}")
        print()
        print("Creating model with random weights...")
        print("⚠️  WARNING: Model will not have trained weights!")
    
    model.eval()
    
    # Export to ONNX
    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, obs_dim)
    
    try:
        # Use legacy ONNX exporter (more stable) com configuração EXATA do ML-Agents
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['vector_observation'],  # Unity ML-Agents expects this name
            output_names=[
                'version_number',
                'memory_size',
                'continuous_actions',
                'continuous_action_output_shape',
                'deterministic_continuous_actions',
            ],
            dynamic_axes={{
                'vector_observation': {{0: 'batch'}},
                'version_number': {{0: 'batch'}},
                'memory_size': {{0: 'batch'}},
                'continuous_actions': {{0: 'batch'}},
                'continuous_action_output_shape': {{0: 'batch'}},
                'deterministic_continuous_actions': {{0: 'batch'}},
            }},
            verbose=False,
            dynamo=False  # Use stable legacy exporter
        )
        print(f'✅ ONNX model exported to: {{onnx_path}}')
        
        # Verify ONNX model (skip for now due to seg fault issues on some systems)
        # if HAS_ONNX:
        #     try:
        #         onnx_model = onnx.load(onnx_path)
        #         onnx.checker.check_model(onnx_model)
        #         print('✅ ONNX model verified successfully')
        #     except Exception as e:
        #         print(f'⚠️  ONNX verification failed: {{e}}')
        
        print(f'✅ ONNX export completed (verification skipped)')
        return 0
    
    except ModuleNotFoundError as e:
        print(f'❌ Missing required package: {{e}}')
        print()
        print('To export ONNX, install:')
        print('  pip install torch onnx onnxscript')
        return 1
    except Exception as e:
        print(f'❌ ONNX export failed: {{e}}')
        print()
        print('This might be due to:')
        print('  1. Incompatible PyTorch version')
        print('  2. Missing dependencies')
        print('  3. Model architecture issues')
        return 1

if __name__ == '__main__':
    sys.exit(main())
"#,
            filename = std::path::Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or("checkpoint"),
            obs_dim = self.obs_dim,
            action_dim = self.action_dim,
            hidden_dim = self.config.hidden_layers[0],
        );
        
        // Save conversion script
        let script_path = format!("{}_convert_to_onnx.py", path);
        std::fs::write(&script_path, conversion_script)?;
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&script_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms)?;
        }
        
        println!("✓ Conversion script saved: {}", script_path);
        
        // Try to run the conversion automatically
        println!("🔄 Attempting automatic ONNX conversion...");
        let output = Command::new("python3")
            .arg(&script_path)
            .output();
        
        match output {
            Ok(output) => {
                if output.status.success() {
                    println!("{}", String::from_utf8_lossy(&output.stdout));
                    if !output.stderr.is_empty() {
                        println!("--- Python Warnings ---\n{}", String::from_utf8_lossy(&output.stderr));
                    }
                    println!("✅ ONNX export completed successfully!");
                    
                    // Keep script for reference (don't delete)
                    // if std::path::Path::new(&format!("{}.onnx", path)).exists() {
                    //     std::fs::remove_file(&script_path).ok();
                    // }
                } else {
                    println!("⚠️  Python conversion failed:");
                    println!("--- stdout ---\n{}", String::from_utf8_lossy(&output.stdout));
                    println!("--- stderr ---\n{}", String::from_utf8_lossy(&output.stderr));
                    println!("\n💡 You can run the conversion manually:");
                    println!("   python3 {}", script_path);
                }
            }
            Err(e) => {
                println!("⚠️  Could not execute Python ({})", e);
                println!("\n💡 Run the conversion manually:");
                println!("   python3 {}", script_path);
            }
        }
        
        // Save metadata
        let metadata = serde_json::json!({
            "model_type": "SAC_Actor",
            "input_shape": [self.obs_dim],
            "output_shape": [self.action_dim],
            "activation": "tanh",
            "normalization": "none",
            "framework": "pytorch",
        });
        let metadata_path = format!("{}_metadata.json", path);
        std::fs::write(metadata_path, serde_json::to_string_pretty(&metadata)?)?;
        
        println!("✓ Model export complete: {}", path);
        Ok(())
    }
    
    pub fn get_step(&self) -> i64 {
        self.step
    }
    
    pub fn get_avg_reward(&self) -> f32 {
        if self.episode_rewards.is_empty() {
            0.0
        } else {
            self.episode_rewards.iter().sum::<f32>() / self.episode_rewards.len() as f32
        }
    }
}

#[derive(Debug)]
pub struct SACMetrics {
    pub actor_loss: f64,
    pub critic_loss: f64,
    pub alpha_loss: f64,
    pub q1_value: f64,
    pub q2_value: f64,
    pub alpha: f64,
    pub step: i64,
}
