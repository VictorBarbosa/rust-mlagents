// Stats writer plugin system
use crate::trainers::stats::StatsWriter;

pub fn register_stats_writer_plugins() -> Vec<Box<dyn StatsWriter>> {
    let mut writers: Vec<Box<dyn StatsWriter>> = Vec::new();
    
    // Register console writer by default
    writers.push(Box::new(crate::trainers::stats::ConsoleWriter));
    
    // Additional writers can be registered here (TensorBoard, CSV, etc.)
    
    writers
}
