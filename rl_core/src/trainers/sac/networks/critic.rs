// Critic Network - Q-Value Network
use tch::{nn, Tensor, Kind};

#[derive(Debug)]
pub struct CriticNetwork {
    fc1: nn::Linear,
    fc2: nn::Linear,
    q_value: nn::Linear,
    dtype: Kind,
}

impl CriticNetwork {
    pub fn new(vs: &nn::Path, obs_dim: i64, action_dim: i64, hidden_dim: i64, dtype: Kind) -> Self {
        let input_dim = obs_dim + action_dim;
        let fc1 = nn::linear(vs / "fc1", input_dim, hidden_dim, Default::default());
        let fc2 = nn::linear(vs / "fc2", hidden_dim, hidden_dim, Default::default());
        let q_value = nn::linear(vs / "q", hidden_dim, 1, Default::default());
        
        Self {
            fc1,
            fc2,
            q_value,
            dtype,
        }
    }
    
    pub fn forward(&self, obs: &Tensor, action: &Tensor) -> Tensor {
        // Concatenate observation and action
        let obs = obs.to_kind(self.dtype);
        let action = action.to_kind(self.dtype);
        let x = Tensor::cat(&[&obs, &action], -1);
        let x = x.apply(&self.fc1).relu();
        let x = x.apply(&self.fc2).relu();
        x.apply(&self.q_value)
    }
}
