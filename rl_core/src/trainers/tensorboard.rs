// TensorBoard logging functionality
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct TensorBoardWriter {
    log_dir: PathBuf,
    run_name: String,
}

impl TensorBoardWriter {
    pub fn new(log_dir: &str, run_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let log_path = Path::new(log_dir).join(run_name);
        std::fs::create_dir_all(&log_path)?;
        
        Ok(Self {
            log_dir: log_path,
            run_name: run_name.to_string(),
        })
    }
    
    pub fn add_scalar(&self, tag: &str, value: f64, step: i64) -> Result<(), Box<dyn std::error::Error>> {
        let file_path = self.log_dir.join(format!("{}.csv", tag.replace('/', "_")));
        
        let file_exists = file_path.exists();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)?;
        
        if !file_exists {
            writeln!(file, "step,value")?;
        }
        
        writeln!(file, "{},{}", step, value)?;
        Ok(())
    }
    
    pub fn add_scalars(&self, main_tag: &str, tag_scalar_dict: &[(&str, f64)], step: i64) -> Result<(), Box<dyn std::error::Error>> {
        for (tag, value) in tag_scalar_dict {
            self.add_scalar(&format!("{}/{}", main_tag, tag), *value, step)?;
        }
        Ok(())
    }
    
    pub fn log_hyperparams(&self, hparams: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
        let hparams_path = self.log_dir.join("hparams.json");
        std::fs::write(hparams_path, serde_json::to_string_pretty(hparams)?)?;
        Ok(())
    }
    
    pub fn flush(&self) -> Result<(), Box<dyn std::error::Error>> {
        // CSV files are flushed automatically on write
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensorboard_writer() {
        let writer = TensorBoardWriter::new("test_logs", "test_run").unwrap();
        writer.add_scalar("loss/actor", 0.5, 100).unwrap();
        writer.add_scalar("loss/critic", 0.3, 100).unwrap();
        
        writer.add_scalars(
            "metrics",
            &[("q1", 1.5), ("q2", 1.6)],
            100
        ).unwrap();
        
        // Cleanup
        std::fs::remove_dir_all("test_logs").ok();
    }
}
