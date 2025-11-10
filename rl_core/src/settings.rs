use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum ScheduleType {
    Constant,
    Linear,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct PPOSettings {
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,
    pub beta: f32,
    pub epsilon: f32,
    pub lambd: f32,
    pub num_epoch: i32,
    pub learning_rate: f32,
    #[serde(default = "default_learning_rate_schedule")]
    pub learning_rate_schedule: ScheduleType,
}

fn default_batch_size() -> usize { 256 }
fn default_buffer_size() -> usize { 20480 }

fn default_learning_rate_schedule() -> ScheduleType {
    ScheduleType::Linear
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct MemorySettings {
    pub sequence_length: Option<usize>,
    pub memory_size: Option<usize>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct NetworkSettings {
    pub hidden_units: Option<usize>,
    pub num_layers: Option<usize>,
    pub normalize: Option<bool>,
    #[serde(default)]
    pub memory: Option<MemorySettings>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct RewardSignalSettings {
    pub gamma: Option<f32>,
    pub strength: Option<f32>,
    #[serde(default)]
    pub network_settings: Option<NetworkSettings>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct BehaviorConfig {
    pub trainer_type: String,
    pub hyperparameters: PPOSettings,
    pub max_steps: u64,
    pub time_horizon: i32,
    pub summary_freq: i32,
    pub keep_checkpoints: i32,
    pub checkpoint_interval: i32,
    #[serde(default)]
    pub network_settings: Option<NetworkSettings>,
    #[serde(default)]
    pub reward_signals: Option<std::collections::HashMap<String, RewardSignalSettings>>,
    #[serde(default)]
    pub init_path: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct EngineSettings {
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default = "default_quality_level")]
    pub quality_level: i32,
    #[serde(default = "default_time_scale")]
    pub time_scale: f32,
    #[serde(default = "default_target_frame_rate")]
    pub target_frame_rate: i32,
    #[serde(default)]
    pub capture_frame_rate: i32,
    #[serde(default)]
    pub no_graphics: bool,
    #[serde(default)]
    pub no_graphics_monitor: bool,
}

fn default_width() -> u32 { 84 }
fn default_height() -> u32 { 84 }
fn default_quality_level() -> i32 { 5 }
fn default_time_scale() -> f32 { 20.0 }
fn default_target_frame_rate() -> i32 { -1 }

#[derive(Debug, Deserialize, PartialEq)]
pub struct CheckpointSettings {
    pub run_id: String,
    #[serde(default)]
    pub initialize_from: Option<String>,
    #[serde(default)]
    pub load_model: bool,
    #[serde(default)]
    pub resume: bool,
    #[serde(default)]
    pub force: bool,
    #[serde(default = "default_train_model")]
    pub train_model: bool,
    #[serde(default)]
    pub inference: bool,
    #[serde(default = "default_results_dir")]
    pub results_dir: String,
}

fn default_train_model() -> bool { true }
fn default_results_dir() -> String { "results".to_string() }

#[derive(Debug, Deserialize, PartialEq)]
pub struct TorchSettings {
    #[serde(default = "default_device")]
    pub device: String,
}

fn default_device() -> String { "cpu".to_string() }

#[derive(Debug, Deserialize, PartialEq)]
pub struct RootConfig {
    #[serde(default)]
    pub behaviors: std::collections::HashMap<String, BehaviorConfig>,
    #[serde(default)]
    pub env_settings: Option<EnvSettings>,
    #[serde(default)]
    pub engine_settings: Option<EngineSettings>,
    #[serde(default)]
    pub checkpoint_settings: Option<CheckpointSettings>,
    #[serde(default)]
    pub torch_settings: Option<TorchSettings>,
    #[serde(default)]
    pub environment_parameters: Option<std::collections::HashMap<String, serde_yaml::Value>>,
    #[serde(default)]
    pub debug: bool,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct EnvSettings {
    pub env_path: Option<String>,
    pub base_port: Option<u16>,
    pub num_envs: Option<usize>,
    pub num_areas: Option<i32>,
    pub seed: Option<i32>,
    #[serde(default)]
    pub timeout_wait: Option<u64>,
    #[serde(default)]
    pub max_lifetime_restarts: Option<i32>,
    #[serde(default)]
    pub restarts_rate_limit_n: Option<i32>,
    #[serde(default)]
    pub restarts_rate_limit_period_s: Option<i32>,
    #[serde(default)]
    pub environment_parameters: Option<std::collections::HashMap<String, serde_yaml::Value>>,
}

// tests moved to rl_core/tests
