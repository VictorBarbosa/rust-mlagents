// Trainer type plugin system
use std::collections::HashMap;

pub type TrainerPluginRegistry = HashMap<String, Box<dyn TrainerPlugin>>;

pub trait TrainerPlugin: Send + Sync {
    fn get_name(&self) -> &str;
    fn create_trainer(&self) -> Result<Box<dyn crate::trainers::trainer::Trainer>, Box<dyn std::error::Error>>;
}

pub fn register_trainer_plugins() -> TrainerPluginRegistry {
    let mut registry = HashMap::new();
    // Register built-in trainers
    // registry.insert("ppo".to_string(), Box::new(PPOPlugin) as Box<dyn TrainerPlugin>);
    // registry.insert("sac".to_string(), Box::new(SACPlugin) as Box<dyn TrainerPlugin>);
    // registry.insert("poca".to_string(), Box::new(POCAPlugin) as Box<dyn TrainerPlugin>);
    registry
}
