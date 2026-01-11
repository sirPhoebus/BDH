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
    /// Arousal level (0.0 to 1.0).
    /// Represents overall activation/stress.
    /// Modulates plasticity (input gain).
    pub arousal: f32,

    /// Valence signal (-1.0 to 1.0).
    /// Positive = Pleasure/Reward. Negative = Pain/Punishment.
    /// Decays towards 0.0 over time.
    pub pleasure_pain: f32,

    /// Chemical state (DA, NE, ACh)
    pub chemicals: Neuromodulators,
}

#[derive(Clone, Debug)]
pub struct Neuromodulators {
    pub dopamine: f32,       // Reward / Plasticity
    pub norepinephrine: f32, // Arousal / Gain
    pub acetylcholine: f32,  // Attention / Input-Internal Balance
}

impl Neuromodulators {
    pub fn new() -> Self {
        Self {
            dopamine: 0.5,
            norepinephrine: 0.5,
            acetylcholine: 0.5,
        }
    }
}

impl BodyState {
    pub fn new() -> Self {
        Self {
            energy: 1.0,
            integrity: 1.0,
            pleasure_pain: 0.0,
            arousal: 0.5,
            chemicals: Neuromodulators::new(),
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

        // 3. Update Arousal
        // High stress (negative valence) -> High Arousal
        // Low energy -> Low Arousal (Sleepy)
        // High energy + Novelty (implied) -> High Arousal
        let stress = if self.pleasure_pain < 0.0 { self.pleasure_pain.abs() * 2.0 } else { 0.0 };
        let target_arousal = (self.energy * 0.7) + (stress * 0.3);
        self.arousal = 0.9 * self.arousal + 0.1 * target_arousal;

        // 4. Update Neuromodulators (The Chemical Substrate)
        
        // Dopamine (DA): Reward Prediction Error proxy. 
        // If Valence > Expected (0), DA spikes.
        // DA decays naturally.
        let reward_signal = self.pleasure_pain.max(0.0);
        self.chemicals.dopamine = 0.9 * self.chemicals.dopamine + 0.1 * reward_signal;
        
        // Norepinephrine (NE): Acute Arousal / Shock.
        // Responds to stress (pain) or high intensity.
        let shock = stress + action_intensity;
        self.chemicals.norepinephrine = 0.8 * self.chemicals.norepinephrine + 0.2 * shock;
        
        // Acetylcholine (ACh): Focus / Wakefulness.
        // High Energy = High ACh (External focus).
        // Low Energy = Low ACh (Internal focus / Dream).
        self.chemicals.acetylcholine = self.energy;

        // 5. Integrity impact
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
