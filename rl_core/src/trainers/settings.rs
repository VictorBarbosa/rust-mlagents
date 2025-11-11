// Settings module - equivalent to mlagents.trainers.settings
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunOptions {
    pub behaviors: HashMap<String, TrainerSettings>,
    pub env_settings: EnvironmentSettings,
    pub engine_settings: EngineSettings,
    pub checkpoint_settings: CheckpointSettings,
    pub torch_settings: TorchSettings,
    pub debug: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSettings {
    pub env_path: Option<String>,
    pub base_port: Option<u16>,
    pub num_envs: usize,
    pub seed: i32,
    pub num_areas: usize,
    pub timeout_wait: u32,
    pub env_args: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSettings {
    pub no_graphics: bool,
    pub no_graphics_monitor: bool,
    pub time_scale: f32,
    pub target_frame_rate: i32,
    pub capture_frame_rate: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSettings {
    pub run_id: String,
    #[serde(default = "default_write_path")]
    pub write_path: PathBuf,
    #[serde(default = "default_run_logs_dir")]
    pub run_logs_dir: PathBuf,
    #[serde(default)]
    pub resume: bool,
    #[serde(default)]
    pub force: bool,
    #[serde(default)]
    pub inference: bool,
    #[serde(default)]
    pub maybe_init_path: Option<PathBuf>,
}

fn default_write_path() -> PathBuf {
    PathBuf::from("results")
}

fn default_run_logs_dir() -> PathBuf {
    PathBuf::from("runs")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorchSettings {
    pub device: String,
    pub num_threads: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerSettings {
    pub trainer_type: TrainerType,
    pub hyperparameters: HyperparameterSettings,
    pub network_settings: NetworkSettings,
    pub reward_signals: HashMap<String, RewardSignalSettings>,
    pub max_steps: u64,
    pub time_horizon: usize,
    pub summary_freq: u64,
    pub keep_checkpoints: usize,
    pub checkpoint_interval: u64,
    #[serde(default = "default_threaded")]
    pub threaded: bool,
}

fn default_threaded() -> bool {
    false
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainerType {
    #[serde(rename = "ppo")]
    PPO,
    #[serde(rename = "sac")]
    SAC,
    #[serde(rename = "poca")]
    POCA,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HyperparameterSettings {
    SAC(SACHyperparameters),
    PPO(PPOHyperparameters),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SACHyperparameters {
    pub batch_size: usize,
    pub buffer_size: usize,
    pub learning_rate: f32,
    #[serde(default = "default_tau")]
    pub tau: f32,
    #[serde(default = "default_gamma")]
    pub gamma: f32,
    #[serde(default = "default_init_entcoef")]
    pub init_entcoef: f32,
    #[serde(default)]
    pub save_replay_buffer: bool,
    #[serde(default = "default_schedule")]
    pub learning_rate_schedule: ScheduleType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPOHyperparameters {
    pub batch_size: usize,
    pub buffer_size: usize,
    pub learning_rate: f32,
    pub beta: f32,
    pub epsilon: f32,
    pub lambd: f32,
    pub num_epoch: usize,
    #[serde(default = "default_schedule")]
    pub learning_rate_schedule: ScheduleType,
}

fn default_tau() -> f32 { 0.005 }
fn default_gamma() -> f32 { 0.99 }
fn default_init_entcoef() -> f32 { 1.0 }
fn default_schedule() -> ScheduleType { ScheduleType::Constant }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleType {
    #[serde(rename = "linear")]
    Linear,
    #[serde(rename = "constant")]
    Constant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSettings {
    pub hidden_units: usize,
    pub num_layers: usize,
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardSignalSettings {
    #[serde(default = "default_gamma")]
    pub gamma: f32,
    #[serde(default = "default_strength")]
    pub strength: f32,
}

fn default_strength() -> f32 { 1.0 }

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            behaviors: HashMap::new(),
            env_settings: EnvironmentSettings::default(),
            engine_settings: EngineSettings::default(),
            checkpoint_settings: CheckpointSettings::default(),
            torch_settings: TorchSettings::default(),
            debug: false,
        }
    }
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            env_path: None,
            base_port: Some(5005),
            num_envs: 1,
            seed: -1,
            num_areas: 1,
            timeout_wait: 60,
            env_args: None,
        }
    }
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            no_graphics: false,
            no_graphics_monitor: false,
            time_scale: 20.0,
            target_frame_rate: -1,
            capture_frame_rate: 60,
        }
    }
}

impl Default for CheckpointSettings {
    fn default() -> Self {
        Self {
            run_id: "default".to_string(),
            write_path: PathBuf::from("./results"),
            run_logs_dir: PathBuf::from("./logs"),
            resume: false,
            force: false,
            inference: false,
            maybe_init_path: None,
        }
    }
}

impl Default for TorchSettings {
    fn default() -> Self {
        Self {
            device: "cpu".to_string(),
            num_threads: None,
        }
    }
}

impl RunOptions {
    pub fn from_yaml(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let options: RunOptions = serde_yaml::from_str(&content)?;
        Ok(options)
    }

    pub fn as_dict(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}
