// Actor Network - Gaussian Policy with Reparametrization Trick
use tch::{nn, Tensor, Kind};

const LOG_STD_MIN: f64 = -20.0;
const LOG_STD_MAX: f64 = 2.0;

#[derive(Debug)]
pub struct ActorNetwork {
    fc1: nn::Linear,
    fc2: nn::Linear,
    mean_layer: nn::Linear,
    log_std_layer: nn::Linear,
    dtype: Kind,
    // Unity ML-Agents compatible parameters
    version_number: Tensor,
    continuous_act_size_vector: Tensor,
    is_continuous_int: Tensor,
    memory_size_vector: Tensor,
}

impl ActorNetwork {
    pub fn new(vs: &nn::Path, obs_dim: i64, action_dim: i64, hidden_dim: i64, dtype: Kind) -> Self {
        let fc1 = nn::linear(vs / "fc1", obs_dim, hidden_dim, Default::default());
        let fc2 = nn::linear(vs / "fc2", hidden_dim, hidden_dim, Default::default());
        let mean_layer = nn::linear(vs / "mean", hidden_dim, action_dim, Default::default());
        let log_std_layer = nn::linear(vs / "log_std", hidden_dim, action_dim, Default::default());
        
        // Unity ML-Agents compatible parameters
        let version_number = Tensor::from(3.0).to_kind(dtype).to_device(vs.device()).set_requires_grad(false);
        let continuous_act_size_vector = Tensor::from(action_dim as f32).to_kind(dtype).to_device(vs.device()).set_requires_grad(false);
        let is_continuous_int = Tensor::from(1.0).to_kind(dtype).to_device(vs.device()).set_requires_grad(false);
        let memory_size_vector = Tensor::from(0.0).to_kind(dtype).to_device(vs.device()).set_requires_grad(false);

        Self {
            fc1,
            fc2,
            mean_layer,
            log_std_layer,
            dtype,
            version_number,
            continuous_act_size_vector,
            is_continuous_int,
            memory_size_vector,
        }
    }
    
    pub fn forward(&self, obs: &Tensor) -> (Tensor, Tensor) {
        let x = obs.to_kind(self.dtype).apply(&self.fc1).relu();
        let x = x.apply(&self.fc2).relu();
        
        let mean = x.apply(&self.mean_layer);
        let log_std = x.apply(&self.log_std_layer).clamp(LOG_STD_MIN, LOG_STD_MAX);
        
        (mean, log_std)
    }
    
    pub fn sample(&self, obs: &Tensor) -> (Tensor, Tensor) {
        let (mean, log_std) = self.forward(obs);
        let std = log_std.exp();
        
        // Reparametrization trick: sample from N(0,1) and scale
        let normal = Tensor::randn_like(&mean);
        let x_t = &mean + &std * normal;
        let action = x_t.tanh();
        
        // Compute log probability with tanh correction
        let log_prob = self.compute_log_prob(&mean, &log_std, &x_t, &action);
        
        (action, log_prob)
    }
    
    fn compute_log_prob(&self, mean: &Tensor, log_std: &Tensor, x_t: &Tensor, action: &Tensor) -> Tensor {
        // Log probability of Gaussian distribution
        let std = log_std.exp();
        let var = &std * &std;
        
        let log_prob: Tensor = -0.5 * (
            ((x_t - mean).pow_tensor_scalar(2) / &var) + 
            (2.0 * std::f64::consts::PI * &var).log()
        );
        
        // Sum over action dimensions
        let log_prob: Tensor = log_prob.sum_dim_intlist(&[-1i64][..], false, self.dtype);
        
        // Tanh squashing correction
        let action_squared = action.pow_tensor_scalar(2);
        let eps = Tensor::from(1e-6).to_kind(self.dtype).to_device(action.device());
        let one_minus_action = Tensor::from(1.0).to_kind(self.dtype).to_device(action.device()) 
            - action_squared + eps;
        let tanh_correction: Tensor = one_minus_action
            .log()
            .sum_dim_intlist(&[-1i64][..], false, self.dtype);
        
        log_prob - tanh_correction
    }
    
    pub fn get_action_deterministic(&self, obs: &Tensor) -> Tensor {
        let (mean, _) = self.forward(obs);
        mean.tanh()
    }
    
    /// Unity ML-Agents compatible forward method for ONNX export
    /// This method provides the exact output structure expected by Unity
    pub fn forward_for_onnx(&self, obs: &Tensor) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        // Get the continuous action outputs
        let (mean, _) = self.forward(obs);
        let continuous_actions = mean.tanh();  // Stochastic action for training
        let deterministic_continuous_actions = mean.tanh();  // Deterministic action for inference
        
        // Return the exact structure expected by Unity ML-Agents ONNX export:
        // (version_number, memory_size, continuous_actions, continuous_act_size_vector, deterministic_continuous_actions)
        (
            self.version_number.shallow_clone(),
            self.memory_size_vector.shallow_clone(),
            continuous_actions,
            self.continuous_act_size_vector.shallow_clone(),
            deterministic_continuous_actions
        )
    }
    
    /// Get continuous action outputs for inference
    pub fn get_action_for_inference(&self, obs: &Tensor) -> Tensor {
        let (mean, _) = self.forward(obs);
        mean.tanh()
    }
}
