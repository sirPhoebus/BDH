/// Represents the physiological state of the agent.
#[derive(Clone, Debug)]
pub struct BodyState {
    /// Energy level (0.0 to 1.0).
    /// Low energy (<0.3) triggers "tiredness" (slow waves).
    /// Zero energy is fatal (or causes reset).
    pub energy: f32,

    /// Physical integrity (0.0 to 1.0).
    /// Represents "health". Decreases on damage/pain.
    pub integrity: f32,

    /// Valence signal (-1.0 to 1.0).
    /// Positive = Pleasure/Reward. Negative = Pain/Punishment.
    /// Decays towards 0.0 over time.
    pub pleasure_pain: f32,
}

impl BodyState {
    pub fn new() -> Self {
        Self {
            energy: 1.0,
            integrity: 1.0,
            pleasure_pain: 0.0,
        }
    }

    /// Update body state based on an action and environmental feedback.
    /// 
    /// # Arguments
    /// * `action_intensity`: usage cost (0.0 to 1.0)
    /// * `env_valence`: feedback from environment (-1.0 to 1.0)
    pub fn update(&mut self, action_intensity: f32, env_valence: f32) {
        // 1. Energy consumption
        // Basal metabolic rate (0.005) + Action cost
        let consumption = 0.005 + 0.05 * action_intensity;
        self.energy = (self.energy - consumption).max(0.0);

        // 2. Valence integration
        // New feedback overrides/mixes with current state
        self.pleasure_pain = 0.7 * self.pleasure_pain + 0.3 * env_valence;

        // 3. Integrity impact
        if env_valence < -0.5 {
            // Strong pain causes damage
            let damage = env_valence.abs() * 0.05;
            self.integrity = (self.integrity - damage).max(0.0);
        } else if self.is_resting() && self.energy > 0.8 {
            // Healing while resting with high energy
            self.integrity = (self.integrity + 0.01).min(1.0);
        }
    }

    /// Check if the body is in a "resting" state (useful for logic).
    pub fn is_resting(&self) -> bool {
        // Heuristic: resting if no recent major valence
        self.pleasure_pain.abs() < 0.1
    }

    /// Get signals as a vector for neural integration.
    /// Order: [Energy, Integrity, Pleasure, Pain]
    /// (Splitting Pleasure/Pain simplifies neural mapping)
    pub fn get_signals(&self) -> Vec<f32> {
        vec![
            self.energy,
            self.integrity,
            self.pleasure_pain.max(0.0), // Pleasure only
            self.pleasure_pain.min(0.0).abs(), // Pain magnitude only
        ]
    }
}
