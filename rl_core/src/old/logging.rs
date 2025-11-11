//! Logging utilities for training metrics similar to TensorBoard
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use serde_json;

pub struct MetricsLogger {
    log_dir: String,
    step: u64,
    metrics_history: HashMap<String, Vec<(u64, f64)>>, // (step, value)
}

impl MetricsLogger {
    pub fn new(log_dir: &str) -> Self {
        let path = Path::new(log_dir);
        if let Err(e) = std::fs::create_dir_all(path) {
            eprintln!("Warning: Could not create log directory '{}': {}", log_dir, e);
        }

        Self {
            log_dir: log_dir.to_string(),
            step: 0,
            metrics_history: HashMap::new(),
        }
    }

    pub fn log_scalar(&mut self, name: &str, value: f64, step: u64) {
        self.metrics_history.entry(name.to_string())
            .or_insert_with(Vec::new)
            .push((step, value));

        // Write individual entry to log file
        let log_path = format!("{}/{}_log.jsonl", self.log_dir, name.replace('.', "_"));
        let mut file = match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path) {
            Ok(f) => f,
            Err(_) => {
                // If we can't create/open the log file, try to create the directory first
                let parent_dir = Path::new(&log_path).parent();
                if let Some(dir) = parent_dir {
                    if let Err(e) = std::fs::create_dir_all(dir) {
                        eprintln!("Warning: Could not create log directory for '{}': {}", log_path, e);
                        return; // Skip logging if we can't create the directory
                    }
                }
                match File::create(&log_path) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("Warning: Could not create log file '{}': {}", log_path, e);
                        return; // Skip logging if we can't create the file
                    }
                }
            }
        };

        let log_entry = format!("{{\"step\": {}, \"value\": {}}}\n", step, value);

        if let Err(e) = writeln!(file, "{}", log_entry) {
            eprintln!("Warning: Could not write to log file '{}': {}", log_path, e);
        }
    }

    pub fn log_episode_reward(&mut self, reward: f64, step: u64) {
        self.log_scalar("episode/reward", reward, step);
    }

    pub fn log_policy_loss(&mut self, loss: f64, step: u64) {
        self.log_scalar("loss/policy", loss, step);
    }

    pub fn log_value_loss(&mut self, loss: f64, step: u64) {
        self.log_scalar("loss/value", loss, step);
    }

    pub fn log_entropy(&mut self, entropy: f64, step: u64) {
        self.log_scalar("policy/entropy", entropy, step);
    }

    pub fn log_total_loss(&mut self, loss: f64, step: u64) {
        self.log_scalar("loss/total", loss, step);
    }

    pub fn flush(&self) {
        // Write all metrics to summary file
        let summary_path = format!("{}/metrics_summary.json", self.log_dir);
        let mut file = File::create(&summary_path).unwrap_or_else(|_| {
            File::create(&summary_path).unwrap()
        });

        // Convert metrics history to JSON string
        let metrics_json = match serde_json::to_string(&self.metrics_history) {
            Ok(json) => json,
            Err(e) => {
                eprintln!("Warning: Could not serialize metrics history: {}", e);
                return;
            }
        };
        let summary_str = format!("{{\"metrics\": {}, \"last_step\": {}}}\n", metrics_json, self.step);

        if let Err(e) = file.write_all(summary_str.as_bytes()) {
            eprintln!("Warning: Could not write to summary file '{}': {}", summary_path, e);
        }
    }
}

impl Drop for MetricsLogger {
    fn drop(&mut self) {
        self.flush();
    }
}