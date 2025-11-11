// Trainer trait - equivalent to mlagents.trainers.trainer.Trainer
use std::collections::HashMap;

pub trait Trainer: Send + Sync {
    fn save_model(&self) -> Result<(), Box<dyn std::error::Error>>;
    
    fn load_model(&mut self) -> Result<(), Box<dyn std::error::Error>>;
    
    fn get_policy(&self) -> Option<&dyn Policy>;
    
    fn advance(&mut self) -> Result<(), Box<dyn std::error::Error>>;
    
    fn get_step_count(&self) -> u64;
    
    fn should_still_train(&self) -> bool;
    
    fn reward_buffer(&self) -> &Vec<f32>;
}

pub trait Policy: Send + Sync {
    fn evaluate(&self, observations: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>>;
    
    fn update(&mut self, batch: &TrainingBatch) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct TrainingBatch {
    pub observations: Vec<Vec<f32>>,
    pub actions: Vec<Vec<f32>>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
}

pub struct TrainerFactory {
    pub trainer_config: HashMap<String, crate::trainers::settings::TrainerSettings>,
    pub output_path: std::path::PathBuf,
    pub train_model: bool,
    pub load_model: bool,
    pub seed: i32,
}

impl TrainerFactory {
    pub fn new(
        trainer_config: HashMap<String, crate::trainers::settings::TrainerSettings>,
        output_path: std::path::PathBuf,
        train_model: bool,
        load_model: bool,
        seed: i32,
    ) -> Self {
        Self {
            trainer_config,
            output_path,
            train_model,
            load_model,
            seed,
        }
    }

    pub fn generate(&self, behavior_id: &str) -> Result<Box<dyn Trainer>, Box<dyn std::error::Error>> {
        let settings = self.trainer_config
            .get(behavior_id)
            .ok_or(format!("No trainer config found for behavior: {}", behavior_id))?;
        
        // Create trainer based on trainer_type
        match settings.trainer_type {
            crate::trainers::settings::TrainerType::PPO => {
                // TODO: Create PPO trainer
                Err("PPO trainer not yet implemented".into())
            }
            crate::trainers::settings::TrainerType::SAC => {
                // TODO: Create SAC trainer
                Err("SAC trainer not yet implemented".into())
            }
            crate::trainers::settings::TrainerType::POCA => {
                // TODO: Create POCA trainer
                Err("POCA trainer not yet implemented".into())
            }
        }
    }
}
