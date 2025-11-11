// Learn module - equivalent to mlagents.trainers.learn
// Main entry point for training ML-Agents

use std::path::PathBuf;
use std::sync::Arc;
use crate::trainers::settings::RunOptions;
use crate::trainers::trainer_controller::TrainerController;
use crate::trainers::environment_parameter_manager::EnvironmentParameterManager;

const TRAINING_STATUS_FILE_NAME: &str = "training_status.json";
const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn get_version_string() -> String {
    format!(
        "Version information:\n  \
        ml-agents-rust: {},\n  \
        burn: {}",
        VERSION,
        "0.18.0"
    )
}

pub fn parse_command_line(args: Option<Vec<String>>) -> Result<RunOptions, Box<dyn std::error::Error>> {
    // TODO: Implement argument parsing similar to Python's argparse
    // For now, load from YAML config file
    let config_path = args
        .and_then(|mut a| a.pop())
        .unwrap_or_else(|| "config.yaml".to_string());
    
    RunOptions::from_yaml(&config_path)
}

pub fn run_training(
    run_seed: i32,
    options: RunOptions,
    num_areas: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting training session...");
    
    // Setup torch/burn settings
    let checkpoint_settings = &options.checkpoint_settings;
    let env_settings = &options.env_settings;
    let engine_settings = &options.engine_settings;
    
    let run_logs_dir = &checkpoint_settings.run_logs_dir;
    let port = env_settings.base_port;
    
    // Validate and create directories
    validate_existing_directories(
        &checkpoint_settings.write_path,
        checkpoint_settings.resume,
        checkpoint_settings.force,
        &checkpoint_settings.maybe_init_path,
    )?;
    
    // Create run logs directory
    std::fs::create_dir_all(run_logs_dir)?;
    
    // Load training status if resuming
    if checkpoint_settings.resume {
        let status_path = run_logs_dir.join(TRAINING_STATUS_FILE_NAME);
        if status_path.exists() {
            println!("Resuming training from {:?}", status_path);
            // TODO: Implement GlobalTrainingStatus::load_state
        }
    }
    
    // Setup environment parameter manager
    let env_parameter_manager = Arc::new(EnvironmentParameterManager::new(
        run_seed,
        checkpoint_settings.resume,
    ));
    
    // Create trainer controller
    let trainer_controller = TrainerController::new(
        checkpoint_settings.write_path.clone(),
        checkpoint_settings.run_id.clone(),
        env_parameter_manager.clone(),
        !checkpoint_settings.inference,
        run_seed,
    );
    
    // Write configuration
    write_run_options(&checkpoint_settings.run_logs_dir, &options)?;
    
    // Start training loop
    trainer_controller.start_learning()?;
    
    // Write final training status
    write_training_status(&checkpoint_settings.run_logs_dir)?;
    
    println!("Training completed successfully!");
    
    Ok(())
}

fn validate_existing_directories(
    write_path: &PathBuf,
    resume: bool,
    force: bool,
    init_path: &Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if write_path.exists() && !resume && !force {
        return Err(format!(
            "Directory {:?} already exists. Use --resume to continue training or --force to overwrite.",
            write_path
        ).into());
    }
    
    if !write_path.exists() {
        std::fs::create_dir_all(write_path)?;
    }
    
    Ok(())
}

fn write_run_options(
    output_dir: &PathBuf,
    run_options: &RunOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = output_dir.join("configuration.yaml");
    let yaml_str = serde_yaml::to_string(run_options)?;
    std::fs::write(config_path, yaml_str)?;
    Ok(())
}

fn write_training_status(output_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let status_path = output_dir.join(TRAINING_STATUS_FILE_NAME);
    // TODO: Implement GlobalTrainingStatus::save_state
    let status = serde_json::json!({
        "completed": true,
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    std::fs::write(status_path, serde_json::to_string_pretty(&status)?)?;
    Ok(())
}

pub fn run_cli(options: RunOptions) -> Result<(), Box<dyn std::error::Error>> {
    // Print ASCII art banner
    println!(r#"
            ┐  ╖
        ╓╖╬│╡  ││╬╖╖
    ╓╖╬│││││┘  ╬│││││╬╖
 ╖╬│││││╬╜        ╙╬│││││╖╖                               ╗╗╗
 ╬╬╬╬╖││╦╖        ╖╬││╗╣╣╣╬      ╟╣╣╬    ╟╣╣╣             ╜╜╜  ╟╣╣
 ╬╬╬╬╬╬╬╬╖│╬╖╖╓╬╪│╓╣╣╣╣╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╒╣╣╖╗╣╣╣╗   ╣╣╣ ╣╣╣╣╣╣ ╟╣╣╖   ╣╣╣
 ╬╬╬╬┐  ╙╬╬╬╬│╓╣╣╣╝╜  ╫╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╟╣╣╣╙ ╙╣╣╣  ╣╣╣ ╙╟╣╣╜╙  ╫╣╣  ╟╣╣
 ╬╬╬╬┐     ╙╬╬╣╣      ╫╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╟╣╣╬   ╣╣╣  ╣╣╣  ╟╣╣     ╣╣╣┌╣╣╜
 ╬╬╬╜       ╬╬╣╣      ╙╝╣╣╬      ╙╣╣╣╗╖╓╗╣╣╣╜ ╟╣╣╬   ╣╣╣  ╣╣╣  ╟╣╣╦╓    ╣╣╣╣╣
 ╙   ╓╦╖    ╬╬╣╣   ╓╗╗╖            ╙╝╣╣╣╣╝╜   ╘╝╝╜   ╝╝╝  ╝╝╝   ╙╣╣╣    ╟╣╣╣
   ╩╬╬╬╬╬╬╦╦╬╬╣╣╗╣╣╣╣╣╣╣╝                                             ╫╣╣╣╣
      ╙╬╬╬╬╬╬╬╣╣╣╣╣╣╝╜
          ╙╬╬╬╣╣╣╜
             ╙
    "#);
    
    println!("{}", get_version_string());
    
    if options.debug {
        println!("Debug mode enabled");
    }
    
    println!("Configuration: {}", serde_json::to_string_pretty(&options.as_dict())?);
    
    let run_seed = if options.env_settings.seed == -1 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(0..10000)
    } else {
        options.env_settings.seed
    };
    
    println!("Run seed: {}", run_seed);
    
    let num_areas = options.env_settings.num_areas;
    
    run_training(run_seed, options, num_areas)?;
    
    Ok(())
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_command_line(None)?;
    run_cli(options)
}
