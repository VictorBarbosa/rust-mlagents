#[derive(Debug, Default, Clone)]
pub struct RolloutBuffer {
    pub observations: Vec<Vec<f32>>,
    pub actions: Vec<Vec<f32>>,
    pub rewards: Vec<f32>,
    pub values: Vec<f32>,
    pub dones: Vec<bool>,
    pub log_probs: Vec<f32>,
    pub advantages: Vec<f32>,
    pub returns: Vec<f32>,
}

impl RolloutBuffer {
    pub fn new() -> Self { Self::default() }

    pub fn push(&mut self, obs: Vec<f32>, action: Vec<f32>, reward: f32, value: f32, done: bool, log_prob: f32) {
        self.observations.push(obs);
        self.actions.push(action);
        self.rewards.push(reward);
        self.values.push(value);
        self.dones.push(done);
        self.log_probs.push(log_prob);
    }

    pub fn len(&self) -> usize { self.rewards.len() }
    pub fn is_empty(&self) -> bool { self.rewards.is_empty() }

    // Compute GAE advantages and returns given gamma, lambda, and last_value for bootstrapping.
    pub fn finish_path(&mut self, gamma: f32, gae_lambda: f32, last_value: f32) {
        let t_len = self.rewards.len();
        self.advantages.clear();  // Clear previous advantages
        self.returns.clear();     // Clear previous returns
        self.advantages.resize(t_len, 0.0);  // Pre-allocate advantages array
        
        let mut adv_next = 0.0f32;
        for t in (0..t_len).rev() {
            let v_t = self.values[t];
            let (v_next, not_done_next) = if t + 1 < t_len {
                (self.values[t + 1], if self.dones[t + 1] { 0.0 } else { 1.0 })
            } else {
                (last_value, 1.0) // bootstrap from last_value when episode continues
            };
            let delta = self.rewards[t] + gamma * not_done_next * v_next - v_t;
            adv_next = delta + gamma * gae_lambda * not_done_next * adv_next;
            self.advantages[t] = adv_next;
        }
        
        // returns = advantages + values
        for t in 0..self.advantages.len() {
            self.returns.push(self.advantages[t] + self.values[t]);
        }
    }

    pub fn clear(&mut self) { *self = Self::default(); }
    
    // Get all data as batch for training
    pub fn get_batch(&self) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            self.observations.clone(),
            self.actions.clone(),
            self.advantages.clone(),
            self.returns.clone(),
            self.log_probs.clone(),
        )
    }
    
    // Normalize advantages (common practice in PPO)
    pub fn normalize_advantages(&mut self) {
        if self.advantages.is_empty() {
            return;
        }
        
        let mean: f32 = self.advantages.iter().sum::<f32>() / self.advantages.len() as f32;
        let variance: f32 = self.advantages.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / self.advantages.len() as f32;
        let std = (variance + 1e-8).sqrt();
        
        for adv in &mut self.advantages {
            *adv = (*adv - mean) / std;
        }
    }
}
