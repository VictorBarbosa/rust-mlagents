// Directory utilities - equivalent to mlagents.trainers.directory_utils
use std::path::{Path, PathBuf};
use std::fs;

pub fn validate_existing_directories(
    write_path: &Path,
    resume: bool,
    force: bool,
    init_path: &Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if write_path.exists() && !resume && !force {
        return Err(format!(
            "Directory {:?} already exists. Use --resume or --force",
            write_path
        ).into());
    }
    Ok(())
}

pub fn setup_init_path(
    behaviors: &std::collections::HashMap<String, crate::trainers::settings::TrainerSettings>,
    init_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Setup initialization paths for behaviors
    Ok(())
}

pub fn create_directory_if_not_exists(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}
