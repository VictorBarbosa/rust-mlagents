// Buffer - equivalent to mlagents.trainers.buffer
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct Experience {
    pub observation: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub done: bool,
    pub next_observation: Vec<f32>,
}

pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        
        let buffer_vec: Vec<&Experience> = self.buffer.iter().collect();
        let sampled: Vec<&Experience> = buffer_vec
            .choose_multiple(&mut rng, batch_size.min(self.buffer.len()))
            .cloned()
            .collect();
        
        sampled.into_iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

pub struct AgentBuffer {
    observations: Vec<Vec<f32>>,
    actions: Vec<Vec<f32>>,
    rewards: Vec<f32>,
    dones: Vec<bool>,
}

impl AgentBuffer {
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            dones: Vec::new(),
        }
    }

    pub fn add_experience(
        &mut self,
        observation: Vec<f32>,
        action: Vec<f32>,
        reward: f32,
        done: bool,
    ) {
        self.observations.push(observation);
        self.actions.push(action);
        self.rewards.push(reward);
        self.dones.push(done);
    }

    pub fn clear(&mut self) {
        self.observations.clear();
        self.actions.clear();
        self.rewards.clear();
        self.dones.clear();
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
}
