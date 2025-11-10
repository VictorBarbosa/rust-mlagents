// PPO Training Algorithm Implementation
use burn::tensor::{backend::Backend, Tensor};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, PrettyJsonFileRecorder, Recorder};
// use burn::nn::loss::MseLoss; // For future loss calculation
use crate::networks::{Actor, Critic};
use crate::ppo_buffer::RolloutBuffer;
use crate::logging::MetricsLogger;

pub struct PPOTrainer<B: Backend> {
    pub actor: Actor<B>,
    pub critic: Critic<B>,
    // Note: Optimizers in Burn are stateless and applied directly
    // We'll handle updates differently
    device: B::Device,
    
    // Hyperparameters
    pub clip_epsilon: f32,
    pub value_coef: f32,
    pub entropy_coef: f32,
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub batch_size: usize,
    pub buffer_size: usize,
}

impl<B: Backend> PPOTrainer<B> {
    pub fn new(
        obs_size: usize,
        action_size: usize,
        hidden_units: usize,
        num_layers: usize,
        learning_rate: f64,
        device: &B::Device,
    ) -> Self {
        let actor = Actor::new(obs_size, hidden_units, num_layers, action_size, device);
        let critic = Critic::new(obs_size, hidden_units, num_layers, device);
        
        Self {
            actor,
            critic,
            device: device.clone(),
            clip_epsilon: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            learning_rate,
            num_epochs: 3,
            gamma: 0.99,
            gae_lambda: 0.95,
            batch_size: 256,
            buffer_size: 20480,
        }
    }
    
    // Forward pass to get action and value
    pub fn get_action_and_value(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, f32) {
        let action = self.actor.forward(obs.clone());
        let value = self.critic.forward(obs);
        
        // For now, log_prob is 0 (we'll implement proper Gaussian policy later)
        let log_prob = 0.0;
        
        (action, value, log_prob)
    }
    
    // Compute value estimates for observations
    pub fn compute_values(&self, observations: &[Vec<f32>]) -> Vec<f32> {
        let batch_size = observations.len();
        let obs_size = observations[0].len();
        
        // Flatten observations into single tensor
        let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let obs_tensor = Tensor::<B, 2>::from_floats(obs_flat.as_slice(), &self.device)
            .reshape([batch_size, obs_size]);
        
        let values = self.critic.forward(obs_tensor);
        let values_data = values.to_data();
        
        values_data.to_vec().unwrap()
    }
    
    // PPO update step
    pub fn update(&mut self, buffer: &mut RolloutBuffer) -> (f32, f32, f32) {
        // TODO: Buffer currently contains placeholder data (vec![0.0])
        // This causes dimension mismatch when calling critic.forward()
        // Need to store real observations/actions first
        
        // Just compute advantages for now (doesn't require forward pass)
        buffer.finish_path(self.gamma, self.gae_lambda, 0.0);
        buffer.normalize_advantages();
        
        // Return dummy losses - no actual weight updates happening
        return (0.0, 0.0, 0.0);
        
        /* COMMENTED OUT - requires valid buffer data
        let (_observations, _actions, _advantages, _returns, _old_log_probs) = buffer.get_batch();
        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy = 0.0;
        let num_updates = self.num_epochs;
        
        // Multiple epochs over the same data (PPO characteristic)
        for _epoch in 0..self.num_epochs {
            let batch_size = observations.len();
            let obs_size = observations[0].len();
            let action_size = actions[0].len();
            
            // Flatten data into tensors
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_tensor = Tensor::<B, 2>::from_floats(obs_flat.as_slice(), &self.device)
                .reshape([batch_size, obs_size]);
            
            let actions_flat: Vec<f32> = actions.iter().flatten().copied().collect();
            let actions_tensor = Tensor::<B, 2>::from_floats(actions_flat.as_slice(), &self.device)
                .reshape([batch_size, action_size]);
            
            let advantages_tensor = Tensor::<B, 1>::from_floats(advantages.as_slice(), &self.device);
            let returns_tensor = Tensor::<B, 1>::from_floats(returns.as_slice(), &self.device);
            
            // Forward pass
            let new_actions = self.actor.forward(obs_tensor.clone());
            let new_values = self.critic.forward(obs_tensor.clone());
            
            // Compute policy loss (simplified - using MSE between actions for now)
            // TODO: Implement proper PPO clip loss with log probabilities
            let action_diff = new_actions.clone() - actions_tensor;
            let policy_loss = (action_diff.clone() * action_diff).mean();
            
            // Value loss (MSE between predicted and target returns)
            let value_pred = new_values.reshape([batch_size]);
            let value_loss = MseLoss::new().forward(
                value_pred.clone(),
                returns_tensor.clone(),
                burn::nn::loss::Reduction::Mean
            );
            
            // Entropy bonus (encourages exploration) - simplified
            let entropy = Tensor::<B, 1>::zeros([1], &self.device);
            
            // Combined loss
            let total_loss = policy_loss.clone() 
                + value_loss.clone().mul_scalar(self.value_coef)
                - entropy.clone().mul_scalar(self.entropy_coef);
            
            // Backward pass and optimize
            // Note: Burn's optimizer API is different, this is simplified
            // In real code, we'd need to use Burn's autodiff properly
            
            total_policy_loss += policy_loss.to_data().to_vec::<f32>().unwrap()[0];
            total_value_loss += value_loss.to_data().to_vec::<f32>().unwrap()[0];
            total_entropy += 0.0; // placeholder
        }
        
        (
            total_policy_loss / num_updates as f32,
            total_value_loss / num_updates as f32,
            total_entropy / num_updates as f32,
        )
        */ // END DISABLED CODE
    }
}

// Add checkpoint functionality for the PPOTrainer
impl<B: Backend> PPOTrainer<B> {
    /// Save the complete model checkpoint with both actor and critic weights
    pub fn save_checkpoint(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;
        use burn::record::{FullPrecisionSettings, PrettyJsonFileRecorder};

        // Create directory if it doesn't exist
        let checkpoint_dir = std::path::Path::new(path);
        fs::create_dir_all(checkpoint_dir)?;

        // Save actor model
        let actor_path = checkpoint_dir.join("actor");
        let actor_record = self.actor.clone().into_record();
        PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
            .record(actor_record, actor_path)?;

        // Save critic model
        let critic_path = checkpoint_dir.join("critic");
        let critic_record = self.critic.clone().into_record();
        PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
            .record(critic_record, critic_path)?;

        // Save hyperparameters
        let config_path = checkpoint_dir.join("config.json");
        let config = serde_json::json!({
            "clip_epsilon": self.clip_epsilon,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
        });
        
        fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;

        // Print location where checkpoint was saved
        if cfg!(debug_assertions) || true {  // Always show in both debug and release
            println!("💾 Checkpoint saved to: {}", checkpoint_dir.display());
        }

        Ok(())
    }

    /// Load a complete model checkpoint with both actor and critic weights
    pub fn load_checkpoint(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use burn::record::{FullPrecisionSettings, PrettyJsonFileRecorder, Recorder};

        let checkpoint_path = std::path::Path::new(path);

        // Load actor model if it exists
        let actor_path = checkpoint_path.join("actor");
        if actor_path.exists() {
            let record = PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
                .load(actor_path, &self.device)?;
            // Load the record directly into the current actor (assuming compatible architecture)
            self.actor = self.actor.clone().load_record(record);
        }

        // Load critic model if it exists
        let critic_path = checkpoint_path.join("critic");
        if critic_path.exists() {
            let record = PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
                .load(critic_path, &self.device)?;
            // Load the record directly into the current critic (assuming compatible architecture)
            self.critic = self.critic.clone().load_record(record);
        }

        // Load hyperparameters (optional, to restore original settings)
        let config_path = checkpoint_path.join("config.json");
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(clip_eps) = config["clip_epsilon"].as_f64() {
                self.clip_epsilon = clip_eps as f32;
            }
            if let Some(value_coef) = config["value_coef"].as_f64() {
                self.value_coef = value_coef as f32;
            }
            if let Some(entropy_coef) = config["entropy_coef"].as_f64() {
                self.entropy_coef = entropy_coef as f32;
            }
            if let Some(lr) = config["learning_rate"].as_f64() {
                self.learning_rate = lr;
            }
            if let Some(epochs) = config["num_epochs"].as_u64() {
                self.num_epochs = epochs as usize;
            }
            if let Some(gamma) = config["gamma"].as_f64() {
                self.gamma = gamma as f32;
            }
            if let Some(gae_lambda) = config["gae_lambda"].as_f64() {
                self.gae_lambda = gae_lambda as f32;
            }
            if let Some(batch_size) = config["batch_size"].as_u64() {
                self.batch_size = batch_size as usize;
            }
            if let Some(buffer_size) = config["buffer_size"].as_u64() {
                self.buffer_size = buffer_size as usize;
            }
        }

        Ok(())
    }

    /// Update the model and log metrics
    pub fn update_with_logging(&mut self, buffer: &mut RolloutBuffer, step: u64) -> (f32, f32, f32) {
        let (policy_loss, value_loss, entropy) = self.update(buffer);
        
        // Log metrics for TensorBoard-like visualization
        if step % 10 == 0 { // Log every 10 steps
            use crate::logging::MetricsLogger;
            let mut logger = MetricsLogger::new("logs"); 
            logger.log_policy_loss(policy_loss as f64, step);
            logger.log_value_loss(value_loss as f64, step);
            logger.log_entropy(entropy as f64, step);
        }
        
        (policy_loss, value_loss, entropy)
    }
    
    /// Find the latest checkpoint in a directory
    pub fn find_latest_checkpoint(checkpoints_dir: &str) -> Option<String> {
        use std::fs;
        
        let checkpoints_path = std::path::Path::new(checkpoints_dir);
        if !checkpoints_path.exists() {
            return None;
        }

        // Look for checkpoint directories (named like "checkpoint_1000", "step_1000", etc.)
        let mut checkpoints = Vec::new();
        if let Ok(entries) = fs::read_dir(checkpoints_path) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    
                    // Extract step number from names like: "checkpoint_1000", "step_500", "model_2000"
                    if name_str.starts_with("checkpoint_") || name_str.starts_with("step_") || name_str.starts_with("model_") {
                        let step_part = name_str.split('_').nth(1);
                        if let Some(step_str) = step_part {
                            if let Ok(step) = step_str.parse::<u64>() {
                                checkpoints.push((step, entry.path()));
                            }
                        }
                    }
                }
            }
        }

        if checkpoints.is_empty() {
            return None;
        }

        // Find the checkpoint with the highest step number
        checkpoints.sort_by_key(|(step, _)| *step);
        checkpoints.last().map(|(_, path)| path.to_string_lossy().to_string())
    }
    

    
    /// Wrapper function to load the latest checkpoint when resume is requested
    pub fn load_latest_checkpoint(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        if let Some(checkpoint_path) = Self::find_latest_checkpoint("checkpoints") {
            match self.load_checkpoint(&checkpoint_path) {
                Ok(()) => {
                    // Try to extract step number from path like "checkpoints/model_step_XYZ" or similar
                    let path_str = &checkpoint_path;
                    if let Some(step_part) = path_str.split('/').last() {
                        if step_part.contains('_') {
                            let parts: Vec<&str> = step_part.split('_').collect();
                            if let Some(last_part) = parts.last() {
                                if let Ok(step) = last_part.parse::<u64>() {
                                    return Ok(step);
                                }
                            }
                        }
                    }
                    Ok(0) // Default to step 0 if we can't extract the step number
                },
                Err(e) => Err(e),
            }
        } else {
            Err("No checkpoint found".into())
        }
    }
}
