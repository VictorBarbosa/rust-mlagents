// Experience Replay Buffer for SAC
use tch::{Tensor, Device};
use std::collections::VecDeque;

#[derive(Clone)]
pub struct Transition {
    pub obs: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub next_obs: Vec<f32>,
    pub done: bool,
}

pub struct ReplayBuffer {
    buffer: VecDeque<Transition>,
    capacity: usize,
    device: Device,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, device: Device) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            device,
        }
    }
    
    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(transition);
    }
    
    pub fn sample(&self, batch_size: usize) -> Option<Batch> {
        if self.buffer.len() < batch_size {
            return None;
        }
        
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        
        let indices: Vec<usize> = (0..self.buffer.len())
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect();
        
        let mut obs_vec = Vec::new();
        let mut action_vec = Vec::new();
        let mut reward_vec = Vec::new();
        let mut next_obs_vec = Vec::new();
        let mut done_vec = Vec::new();
        
        for &idx in &indices {
            let t = &self.buffer[idx];
            obs_vec.extend_from_slice(&t.obs);
            action_vec.extend_from_slice(&t.action);
            reward_vec.push(t.reward);
            next_obs_vec.extend_from_slice(&t.next_obs);
            done_vec.push(if t.done { 1.0 } else { 0.0 });
        }
        
        let obs_dim = self.buffer[0].obs.len() as i64;
        let action_dim = self.buffer[0].action.len() as i64;
        let batch_size = batch_size as i64;
        
        // Use float32 for MPS compatibility
        use tch::Kind;
        
        Some(Batch {
            obs: Tensor::from_slice(&obs_vec)
                .to_kind(Kind::Float)
                .to_device(self.device)
                .reshape(&[batch_size, obs_dim]),
            action: Tensor::from_slice(&action_vec)
                .to_kind(Kind::Float)
                .to_device(self.device)
                .reshape(&[batch_size, action_dim]),
            reward: Tensor::from_slice(&reward_vec)
                .to_kind(Kind::Float)
                .to_device(self.device)
                .reshape(&[batch_size, 1]),
            next_obs: Tensor::from_slice(&next_obs_vec)
                .to_kind(Kind::Float)
                .to_device(self.device)
                .reshape(&[batch_size, obs_dim]),
            done: Tensor::from_slice(&done_vec)
                .to_kind(Kind::Float)
                .to_device(self.device)
                .reshape(&[batch_size, 1]),
        })
    }
    
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

pub struct Batch {
    pub obs: Tensor,
    pub action: Tensor,
    pub reward: Tensor,
    pub next_obs: Tensor,
    pub done: Tensor,
}
