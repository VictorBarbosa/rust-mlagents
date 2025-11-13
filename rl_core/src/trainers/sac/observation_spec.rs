// Observation Specification - Detecta tipos de sensores do Unity
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationSpec {
    pub has_vector_obs: bool,
    pub vector_obs_size: usize,
    pub has_ray_perception: bool,
    pub ray_perception_specs: Vec<RayPerceptionSpec>,
    pub total_obs_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayPerceptionSpec {
    pub name: String,
    pub num_rays: usize,
    pub data_per_ray: usize,
    pub total_size: usize,
}

impl ObservationSpec {
    pub fn new() -> Self {
        Self {
            has_vector_obs: false,
            vector_obs_size: 0,
            has_ray_perception: false,
            ray_perception_specs: Vec::new(),
            total_obs_size: 0,
        }
    }

    /// Detecta specs a partir da primeira observação do Unity
    pub fn detect_from_observations(observations: &[Vec<f32>]) -> Self {
        let mut spec = Self::new();
        
        if observations.is_empty() {
            return spec;
        }

        // Primeira observação geralmente é vector observation
        if !observations.is_empty() {
            spec.has_vector_obs = true;
            spec.vector_obs_size = observations[0].len();
            spec.total_obs_size += observations[0].len();
        }

        // Observações adicionais são sensores (RayPerception, etc)
        if observations.len() > 1 {
            spec.has_ray_perception = true;
            
            for (i, obs) in observations.iter().skip(1).enumerate() {
                let ray_spec = RayPerceptionSpec {
                    name: format!("RayPerceptionSensor{}", i),
                    num_rays: Self::estimate_num_rays(obs.len()),
                    data_per_ray: Self::estimate_data_per_ray(obs.len()),
                    total_size: obs.len(),
                };
                
                spec.total_obs_size += obs.len();
                spec.ray_perception_specs.push(ray_spec);
            }
        }

        spec
    }

    fn estimate_num_rays(total_size: usize) -> usize {
        // ML-Agents RayPerception geralmente usa múltiplos de detectable tags
        // Formato: [ray_data, ray_data, ...] onde cada ray_data tem N elementos
        
        // Tamanhos comuns de ray perception:
        // - 3 tags detectáveis = 3 + 1 (hit/miss) = 4 per ray
        // - Ou pode ser variável
        
        // Tentativa de detectar: assumir ~4-6 elementos por raio
        let estimated_per_ray = 5; // média
        total_size / estimated_per_ray
    }

    fn estimate_data_per_ray(total_size: usize) -> usize {
        let num_rays = Self::estimate_num_rays(total_size);
        if num_rays > 0 {
            total_size / num_rays
        } else {
            0
        }
    }

    pub fn print_info(&self) {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║           📊 OBSERVATION SPECIFICATION DETECTED            ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();
        
        if self.has_vector_obs {
            println!("✅ Vector Observations:");
            println!("   └─ Size: {} dimensions", self.vector_obs_size);
        }
        
        if self.has_ray_perception {
            println!("\n✅ RayPerception Sensors Detected:");
            for (i, ray_spec) in self.ray_perception_specs.iter().enumerate() {
                println!("   Sensor {}:", i);
                println!("   └─ Name: {}", ray_spec.name);
                println!("   └─ Estimated rays: ~{}", ray_spec.num_rays);
                println!("   └─ Data per ray: ~{}", ray_spec.data_per_ray);
                println!("   └─ Total size: {}", ray_spec.total_size);
            }
        } else {
            println!("\n⚠️  No RayPerception sensors detected");
            println!("   └─ Training with vector observations only");
        }
        
        println!("\n📏 Total Observation Size: {} dimensions", self.total_obs_size);
        println!("\n💡 Model will be configured for:");
        if self.has_ray_perception {
            println!("   ✓ Multiple observation inputs (vector + ray perception)");
        } else {
            println!("   ✓ Single vector observation input");
        }
        
        println!("════════════════════════════════════════════════════════════════\n");
    }

    /// Concatena todas as observações em um único vetor
    pub fn flatten_observations(&self, observations: &[Vec<f32>]) -> Vec<f32> {
        let mut flattened = Vec::with_capacity(self.total_obs_size);
        
        for obs in observations {
            flattened.extend_from_slice(obs);
        }
        
        flattened
    }

    /// Verifica se a configuração mudou
    pub fn matches(&self, observations: &[Vec<f32>]) -> bool {
        let detected = Self::detect_from_observations(observations);
        
        self.has_vector_obs == detected.has_vector_obs &&
        self.vector_obs_size == detected.vector_obs_size &&
        self.has_ray_perception == detected.has_ray_perception &&
        self.ray_perception_specs.len() == detected.ray_perception_specs.len()
    }
}

impl Default for ObservationSpec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_only() {
        let obs = vec![vec![1.0; 62]];
        let spec = ObservationSpec::detect_from_observations(&obs);
        
        assert!(spec.has_vector_obs);
        assert_eq!(spec.vector_obs_size, 62);
        assert!(!spec.has_ray_perception);
        assert_eq!(spec.total_obs_size, 62);
    }

    #[test]
    fn test_vector_plus_ray() {
        let obs = vec![
            vec![1.0; 62],     // Vector obs
            vec![0.5; 100],    // RayPerception
        ];
        let spec = ObservationSpec::detect_from_observations(&obs);
        
        assert!(spec.has_vector_obs);
        assert_eq!(spec.vector_obs_size, 62);
        assert!(spec.has_ray_perception);
        assert_eq!(spec.ray_perception_specs.len(), 1);
        assert_eq!(spec.total_obs_size, 162);
    }

    #[test]
    fn test_flatten() {
        let obs = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0],
        ];
        let spec = ObservationSpec::detect_from_observations(&obs);
        let flattened = spec.flatten_observations(&obs);
        
        assert_eq!(flattened, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(flattened.len(), spec.total_obs_size);
    }
}
