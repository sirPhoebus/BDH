/// Represents the intrinsic drives of the agent.
#[derive(Clone, Debug)]
pub struct Drives {
    /// Hunger drive (0.0 to 1.0).
    /// Increases as body energy decreases.
    /// Dominant when high (> 0.7).
    pub hunger: f32,

    /// Curiosity drive (0.0 to 1.0).
    /// Increases when the environment/state is predictable (low novelty).
    /// Resets when novelty is encountered.
    pub curiosity: f32,

    /// Thresholds for dominance (Hunger, Curiosity).
    pub thresholds: (f32, f32),
}

impl Drives {
    pub fn new() -> Self {
        Self {
            hunger: 0.0,
            curiosity: 0.0,
            thresholds: (0.4, 0.3), // Hunger dominates early, Curiosity triggers easier
        }
    }

    /// Update drives based on physiological and cognitive state.
    /// 
    /// # Arguments
    /// * `body_energy`: Current body energy (0.0-1.0).
    /// * `novelty`: Recent novelty/surprise signal (0.0-1.0).
    pub fn update(&mut self, body_energy: f32, novelty: f32) {
        // 1. Hunger: Inverse of energy
        self.hunger = (1.0 - body_energy).max(0.0);

        // 2. Curiosity:
        // If novelty is high, curiosity is "satisfied" (drops).
        // If novelty is low (boredom), curiosity rises.
        if novelty > 0.2 {
            self.curiosity = (self.curiosity - novelty * 2.0).max(0.0);
        } else {
            self.curiosity = (self.curiosity + 0.05).min(1.0);
        }
    }

    /// specific motivation signal.
    pub fn get_dominant_drive(&self) -> Option<String> {
        // Hierarchy: Hunger > Curiosity
        if self.hunger > self.thresholds.0 {
            Some("HUNGER".to_string())
        } else if self.curiosity > self.thresholds.1 {
            Some("CURIOSITY".to_string())
        } else {
            None
        }
    }
}
