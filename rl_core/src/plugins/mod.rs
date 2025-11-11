// Plugins module - equivalent to mlagents.plugins
// Used for extensibility and custom trainer types

pub mod trainer_type;
pub mod stats_writer;

pub use trainer_type::*;
pub use stats_writer::*;
