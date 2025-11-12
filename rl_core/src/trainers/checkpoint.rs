// Checkpoint management system
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::fs::DirEntry;
use regex::Regex;
use std::cmp::Ordering;

pub struct CheckpointManager {
    checkpoint_dir: String,
    keep_checkpoints: usize,
    saved_checkpoints: VecDeque<String>,
}

impl CheckpointManager {
    pub fn new(checkpoint_dir: String, keep_checkpoints: usize) -> Self {
        // Create checkpoint directory if it doesn't exist
        fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint directory");
        
        Self {
            checkpoint_dir,
            keep_checkpoints,
            saved_checkpoints: VecDeque::new(),
        }
    }

    pub fn save_checkpoint(&mut self, trainer: &dyn Checkpointable, step: u64, behavior_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let checkpoint_path = format!("{}/{}-{}.pt", self.checkpoint_dir, behavior_name, step);
        
        trainer.save_checkpoint(&checkpoint_path)?;
        
        // Clean up old checkpoints to maintain limit
        self.cleanup_old_checkpoints(behavior_name)?;
        
        println!("✅ Checkpoint saved: {}-{}.pt", behavior_name, step);
        Ok(())
    }
    
    fn cleanup_old_checkpoints(&mut self, behavior_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Find all checkpoint files matching the pattern: {behavior_name}-{number}.pt
        let re = Regex::new(&format!(r"^{}-(\d+)\.pt$", regex::escape(behavior_name))).unwrap();
        
        let mut checkpoints: Vec<(u64, String)> = Vec::new();
        
        // Read all files in the checkpoint directory
        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                if let Some(file_name) = entry.file_name().to_str() {
                    if let Some(captures) = re.captures(file_name) {
                        if let Some(step_match) = captures.get(1) {
                            if let Ok(step) = step_match.as_str().parse::<u64>() {
                                checkpoints.push((step, entry.path().to_string_lossy().to_string()));
                            }
                        }
                    }
                }
            }
        }
        
        // Sort by step number in descending order (newest first)
        checkpoints.sort_by(|a, b| b.0.cmp(&a.0));
        
        // Keep only the most recent checkpoints up to keep_checkpoints limit
        if checkpoints.len() > self.keep_checkpoints {
            // Remove the oldest checkpoints (those beyond keep_checkpoints)
            for i in self.keep_checkpoints..checkpoints.len() {
                let (_, old_checkpoint_path) = &checkpoints[i];
                if Path::new(old_checkpoint_path).exists() {
                    fs::remove_file(old_checkpoint_path)?;
                    println!("🗑️  Removed old checkpoint: {}", old_checkpoint_path);
                }
            }
        }
        
        Ok(())
    }
}

pub trait Checkpointable {
    fn save_checkpoint(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
    fn load_checkpoint(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
}

// Implement for SACTrainer
impl Checkpointable for crate::trainers::sac::SACTrainer {
    fn save_checkpoint(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.save_checkpoint(path)
    }
    
    fn load_checkpoint(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // For now, we only implement save_checkpoint
        println!("Load checkpoint not implemented for SACTrainer");
        Ok(())
    }
}