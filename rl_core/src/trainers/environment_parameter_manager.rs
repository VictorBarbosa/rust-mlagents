// EnvironmentParameterManager - equivalent to mlagents.trainers.environment_parameter_manager
use std::collections::HashMap;

pub struct EnvironmentParameterManager {
    parameters: HashMap<String, f32>,
    seed: i32,
    restore: bool,
}

impl EnvironmentParameterManager {
    pub fn new(seed: i32, restore: bool) -> Self {
        Self {
            parameters: HashMap::new(),
            seed,
            restore,
        }
    }

    pub fn set_parameter(&mut self, key: String, value: f32) {
        self.parameters.insert(key, value);
    }

    pub fn get_parameter(&self, key: &str) -> Option<f32> {
        self.parameters.get(key).copied()
    }

    pub fn get_all_parameters(&self) -> &HashMap<String, f32> {
        &self.parameters
    }
}
