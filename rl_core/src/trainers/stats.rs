// Stats - equivalent to mlagents.trainers.stats
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;

pub struct StatsReporter {
    stats: HashMap<String, Vec<f32>>,
    writers: Vec<Box<dyn StatsWriter>>,
}

impl StatsReporter {
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            writers: Vec::new(),
        }
    }

    pub fn add_writer(&mut self, writer: Box<dyn StatsWriter>) {
        self.writers.push(writer);
    }

    pub fn add_stat(&mut self, key: String, value: f32) {
        self.stats.entry(key).or_insert_with(Vec::new).push(value);
    }

    pub fn write_stats(&mut self, step: u64) -> Result<(), Box<dyn std::error::Error>> {
        for writer in &mut self.writers {
            writer.write(&self.stats, step)?;
        }
        Ok(())
    }

    pub fn get_stats(&self, key: &str) -> Option<&Vec<f32>> {
        self.stats.get(key)
    }
}

pub trait StatsWriter: Send {
    fn write(&mut self, stats: &HashMap<String, Vec<f32>>, step: u64) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct ConsoleWriter;

impl StatsWriter for ConsoleWriter {
    fn write(&mut self, stats: &HashMap<String, Vec<f32>>, step: u64) -> Result<(), Box<dyn std::error::Error>> {
        println!("Step {}: {:?}", step, stats);
        Ok(())
    }
}

pub struct FileWriter {
    path: std::path::PathBuf,
}

impl FileWriter {
    pub fn new(path: std::path::PathBuf) -> Self {
        Self { path }
    }
}

impl StatsWriter for FileWriter {
    fn write(&mut self, stats: &HashMap<String, Vec<f32>>, step: u64) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        
        writeln!(file, "Step {}: {:?}", step, stats)?;
        Ok(())
    }
}
