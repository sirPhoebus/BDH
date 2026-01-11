use crate::harmonic::{BiologicalConfig, HarmonicBdh};
use ndarray::Array1;

#[derive(Clone, Debug, PartialEq)]
pub enum RegionType {
    Sensory,    // fast, high noise, reactive
    Associative,// binding, balanced
    Executive,  // stable, persistent
}

pub struct BrainRegion {
    pub bdh: HarmonicBdh,
    pub region_type: RegionType,
    pub connections: Vec<usize>, // indices of downstream regions
    pub name: String,
}

impl BrainRegion {
    pub fn new(name: &str, region_type: RegionType, neurons: usize, dims: usize, layers: usize) -> Self {
        let mut config = BiologicalConfig::default();
        
        match region_type {
            RegionType::Sensory => {
                config.noise_amplitude = 0.15;        // higher noise to capture weak signals
                config.homeostatic_rate = 0.5;        // fast adaptation
                config.base_damping = 0.95;           // fast decay (transient memory)
                // Higher frequencies for sensory processing (Gamma range)
                config.layer_frequencies = vec![30.0, 50.0, 80.0]; 
            },
            RegionType::Associative => {
                config.cross_freq_coupling = 0.6;     // strong binding
                config.rho_decay = 0.999;             // longer memory trace
                config.layer_frequencies = vec![5.0, 15.0, 40.0]; // Theta-Beta-Gamma
            },
            RegionType::Executive => {
                config.self_excitation = 0.05;        // strong persistence
                config.base_damping = 0.995;          // very slow decay
                config.reflection_feedback = 0.9;     // strong self-focus
                config.homeostatic_threshold = 0.2;   // deeper capacity before fatigue
                config.layer_frequencies = vec![1.0, 4.0, 10.0]; // Delta-Theta-Alpha
            },
        }

        let bdh = HarmonicBdh::with_config(neurons, dims, layers, config);
        
        Self {
            bdh,
            region_type,
            connections: vec![],
            name: name.to_string(),
        }
    }

    /// Add a connection key to another region
    pub fn connect_to(&mut self, region_idx: usize) {
        self.connections.push(region_idx);
    }

    /// Process input and return semantic projections for downstream regions
    pub fn process(&mut self, input: &Array1<f32>) -> (Array1<f32>, Vec<f32>) {
        // Run internal dynamics
        let (output, energies, _) = self.bdh.forward(input);
        (output, energies)
    }

    /// Mechanism to inject output from this region into another
    /// For now, simpler projection: just return the output. 
    /// In a real connectome, this would pass through a sparse matrix.
    pub fn project_output(&self, raw_output: &Array1<f32>) -> Array1<f32> {
        // We could apply a specific transfer function here if needed
        raw_output.clone()
    }
}
