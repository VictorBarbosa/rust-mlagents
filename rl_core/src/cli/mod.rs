// Command-Line Interface for rust-mlagents
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(name = "rust-mlagents")]
#[command(author = "ML-Agents Team")]
#[command(version = "0.1.0")]
#[command(about = "Unity ML-Agents Trainer (Rust Implementation)", long_about = None)]
pub struct Cli {
    /// Path to the trainer configuration YAML file
    #[arg(value_name = "CONFIG")]
    pub config_path: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Path to the Unity executable to train
    #[arg(long = "env", value_name = "PATH")]
    pub env_path: Option<PathBuf>,

    /// Whether to resume training from a checkpoint
    #[arg(long, default_value_t = false)]
    pub resume: bool,

    /// Whether to force-overwrite this run-id's existing data
    #[arg(long, default_value_t = false)]
    pub force: bool,

    /// The identifier for the training run
    #[arg(long = "run-id", default_value = "sac_default")]
    pub run_id: String,

    /// Initialize model from a previously saved run ID
    #[arg(long = "initialize-from", value_name = "RUN_ID")]
    pub initialize_from: Option<String>,

    /// Seed for random number generator
    #[arg(long, default_value_t = -1)]
    pub seed: i32,

    /// Whether to run in inference mode (no training)
    #[arg(long, default_value_t = false)]
    pub inference: bool,

    /// Starting port for environment communication
    #[arg(long = "base-port", default_value_t = 5005)]
    pub base_port: u16,

    /// Number of concurrent Unity environment instances
    #[arg(long = "num-envs", default_value_t = 1)]
    pub num_envs: usize,

    /// Number of parallel training areas per environment
    #[arg(long = "num-areas", default_value_t = 1)]
    pub num_areas: usize,

    /// Enable debug-level logging
    #[arg(long, default_value_t = false)]
    pub debug: bool,

    /// Results base directory
    #[arg(long = "results-dir", default_value = "results")]
    pub results_dir: PathBuf,

    /// Timeout to wait for Unity environment startup (seconds)
    #[arg(long = "timeout-wait", default_value_t = 60)]
    pub timeout_wait: u64,

    /// Device to use for training (cpu, cuda, mps)
    #[arg(long = "device", default_value = "auto")]
    pub device: DeviceType,

    /// CUDA device index (if using CUDA)
    #[arg(long = "cuda-device", default_value_t = 0)]
    pub cuda_device: usize,

    // Engine Configuration
    /// Width of the executable window in pixels
    #[arg(long, default_value_t = 84)]
    pub width: u32,

    /// Height of the executable window in pixels
    #[arg(long, default_value_t = 84)]
    pub height: u32,

    /// Quality level for rendering
    #[arg(long = "quality-level", default_value_t = 5)]
    pub quality_level: i32,

    /// Target framerate for simulation
    #[arg(long = "time-scale", default_value_t = 20.0)]
    pub time_scale: f32,

    /// Whether to capture frames from the environment
    #[arg(long = "capture-frame-rate", default_value_t = 60)]
    pub capture_frame_rate: u32,

    /// Whether to run in no-graphics mode
    #[arg(long, default_value_t = false)]
    pub no_graphics: bool,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Train a new model or resume training
    Train {
        /// Path to configuration file (overrides positional arg)
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
    
    /// Run inference with a trained model
    Infer {
        /// Path to the trained model checkpoint
        #[arg(short, long)]
        model: PathBuf,
        
        /// Path to the Unity executable
        #[arg(short, long)]
        env: Option<PathBuf>,
    },
    
    /// Export trained model to ONNX format
    Export {
        /// Path to the checkpoint to export
        #[arg(short, long)]
        checkpoint: PathBuf,
        
        /// Output path for ONNX file
        #[arg(short, long)]
        output: PathBuf,
    },
    
    /// Show version information
    Version,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum DeviceType {
    /// Automatically select best available device
    Auto,
    /// Use CPU
    Cpu,
    /// Use NVIDIA CUDA GPU
    Cuda,
    /// Use Apple Metal Performance Shaders (M1/M2/M3)
    Mps,
}

impl Cli {
    pub fn validate(&self) -> Result<(), String> {
        // Validate resume + run-id
        if self.resume && self.run_id == "sac_default" {
            return Err("--resume requires a specific --run-id".to_string());
        }

        // Validate inference mode
        if self.inference && !self.resume {
            return Err("--inference requires --resume with a trained model".to_string());
        }

        // Validate num_envs
        if self.num_envs == 0 {
            return Err("--num-envs must be at least 1".to_string());
        }

        // Validate ports
        if self.num_envs > 1 {
            let max_port = self.base_port as usize + self.num_envs;
            if max_port > 65535 {
                return Err(format!(
                    "Port range exceeds 65535. base-port: {}, num-envs: {}",
                    self.base_port, self.num_envs
                ));
            }
        }

        Ok(())
    }

    pub fn get_device_string(&self) -> String {
        match self.device {
            DeviceType::Auto => "auto".to_string(),
            DeviceType::Cpu => "cpu".to_string(),
            DeviceType::Cuda => format!("cuda:{}", self.cuda_device),
            DeviceType::Mps => "mps".to_string(),
        }
    }

    pub fn get_results_path(&self) -> PathBuf {
        self.results_dir.join(&self.run_id)
    }

    pub fn get_checkpoint_path(&self) -> PathBuf {
        self.get_results_path().join("checkpoints")
    }

    pub fn get_logs_path(&self) -> PathBuf {
        self.get_results_path().join("logs")
    }

    pub fn get_tensorboard_path(&self) -> PathBuf {
        self.get_results_path().join("tensorboard")
    }

    pub fn should_write_checkpoints(&self) -> bool {
        !self.inference
    }
}

pub fn print_banner() {
    println!(r#"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║          🦀 Unity ML-Agents Trainer (Rust Edition) 🦀         ║
║                                                               ║
║                     Version: 0.1.0                            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"#);
}

pub fn print_configuration(cli: &Cli) {
    println!("\n📝 Configuration:");
    println!("   Run ID:         {}", cli.run_id);
    println!("   Results Dir:    {}", cli.results_dir.display());
    println!("   Device:         {}", cli.get_device_string());
    
    if let Some(env) = &cli.env_path {
        println!("   Environment:    {}", env.display());
    } else {
        println!("   Environment:    <Unity Editor>");
    }
    
    println!("   Base Port:      {}", cli.base_port);
    println!("   Num Envs:       {}", cli.num_envs);
    println!("   Num Areas:      {}", cli.num_areas);
    
    if cli.resume {
        println!("   Mode:           Resume Training");
    } else if cli.inference {
        println!("   Mode:           Inference");
    } else {
        println!("   Mode:           New Training");
    }
    
    if let Some(init_from) = &cli.initialize_from {
        println!("   Initialize:     {}", init_from);
    }
    
    if cli.seed >= 0 {
        println!("   Seed:           {}", cli.seed);
    }
    
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_basic() {
        let cli = Cli::parse_from(&["rust-mlagents", "config.yaml"]);
        assert_eq!(cli.config_path, Some(PathBuf::from("config.yaml")));
    }

    #[test]
    fn test_cli_parse_with_flags() {
        let cli = Cli::parse_from(&[
            "rust-mlagents",
            "--run-id", "test_run",
            "--resume",
            "--device", "cuda",
        ]);
        assert_eq!(cli.run_id, "test_run");
        assert!(cli.resume);
    }

    #[test]
    fn test_device_string() {
        let cli = Cli::parse_from(&["rust-mlagents", "--device", "mps"]);
        assert_eq!(cli.get_device_string(), "mps");
    }
}
