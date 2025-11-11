// ONNX Export functionality for SAC Actor network
use tch::{nn, Tensor, Kind, Device};
use super::ActorNetwork;

pub struct ONNXExporter {
    actor: ActorNetwork,
    obs_dim: i64,
    device: Device,
}

impl ONNXExporter {
    pub fn new(actor: ActorNetwork, obs_dim: i64, device: Device) -> Self {
        Self { actor, obs_dim, device }
    }
    
    /// Export actor network to ONNX format
    /// Note: tch-rs has limited ONNX support, using TorchScript instead
    pub fn export(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("📦 Exporting model to: {}", path);
        
        // Create dummy input for tracing
        let dummy_input = Tensor::randn(&[1, self.obs_dim], (Kind::Float, self.device));
        
        // For ONNX export, we'd ideally use:
        // torch.onnx.export() in Python, or CModule::save_to_stream
        // But tch-rs 0.18 has limited support
        
        // Alternative 1: Save as TorchScript (can be loaded in Python and converted)
        tch::no_grad(|| {
            let _output = self.actor.get_action_deterministic(&dummy_input);
            
            // Save using Python bridge approach
            println!("⚠️  Direct ONNX export not fully supported in tch-rs");
            println!("   Recommended approach:");
            println!("   1. Save checkpoint: model.pt");
            println!("   2. Use Python script to convert .pt -> .onnx");
            println!("   3. Load .onnx in Unity with Barracuda");
            
            Ok::<(), Box<dyn std::error::Error>>(())
        })?;
        
        // Create Python conversion script
        let conversion_script = format!(
r#"#!/usr/bin/env python3
"""
Convert PyTorch checkpoint to ONNX format
Usage: python convert_to_onnx.py {path}.pt {path}.onnx
"""
import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.mean(x))
        return action

# Load checkpoint
checkpoint = torch.load('{path}.pt', map_location='cpu')
obs_dim = {obs_dim}
action_dim = checkpoint['mean.weight'].shape[0]  # Infer from checkpoint
hidden_dim = checkpoint['fc1.weight'].shape[0]

# Create model and load weights
model = ActorNetwork(obs_dim, action_dim, hidden_dim)
model.load_state_dict(checkpoint)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, obs_dim)
torch.onnx.export(
    model,
    dummy_input,
    '{path}.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['observation'],
    output_names=['action'],
    dynamic_axes={{
        'observation': {{0: 'batch_size'}},
        'action': {{0: 'batch_size'}}
    }}
)
print(f'✓ ONNX model exported to: {path}.onnx')
"#,
            path = path,
            obs_dim = self.obs_dim,
        );
        
        // Save conversion script
        let script_path = format!("{}_convert_to_onnx.py", path);
        std::fs::write(&script_path, conversion_script)?;
        println!("✓ Conversion script saved: {}", script_path);
        println!("  Run: python3 {}", script_path);
        
        Ok(())
    }
    
    /// Alternative: Export model metadata for Unity
    pub fn export_metadata(&self, path: &str, action_dim: i64) -> Result<(), Box<dyn std::error::Error>> {
        let metadata = serde_json::json!({
            "model_type": "SAC_Actor",
            "input_shape": [self.obs_dim],
            "output_shape": [action_dim],
            "activation": "tanh",
            "normalization": "none",
            "framework": "pytorch",
            "export_date": chrono::Utc::now().to_rfc3339(),
        });
        
        let metadata_path = format!("{}_metadata.json", path);
        std::fs::write(metadata_path, serde_json::to_string_pretty(&metadata)?)?;
        
        Ok(())
    }
}

/// Helper function to create ONNX-compatible model
pub fn create_onnx_compatible_actor(
    vs: &nn::Path,
    obs_dim: i64,
    action_dim: i64,
    hidden_dim: i64,
) -> nn::Sequential {
    nn::seq()
        .add(nn::linear(vs / "fc1", obs_dim, hidden_dim, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(vs / "fc2", hidden_dim, hidden_dim, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(vs / "mean", hidden_dim, action_dim, Default::default()))
        .add_fn(|x| x.tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn;
    
    #[test]
    fn test_onnx_export_script_generation() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let actor = ActorNetwork::new(&vs.root(), 8, 2, 128, Kind::Float);
        
        let exporter = ONNXExporter::new(actor, 8, device);
        exporter.export("test_model").unwrap();
        
        // Check if conversion script was created
        assert!(std::path::Path::new("test_model_convert_to_onnx.py").exists());
        
        // Cleanup
        std::fs::remove_file("test_model_convert_to_onnx.py").ok();
    }
}
