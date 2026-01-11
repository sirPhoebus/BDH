use bdh_model::environment::{CartPole, Environment};
use bdh_model::harmonic::{HarmonicBdh, BiologicalConfig};
use std::thread;
use std::time::Duration;

fn main() {
    println!("=== Phase 1: Embodied Consciousness Demo ===");
    println!("Initializing CartPole Environment...");
    
    let mut env = CartPole::default();
    
    let config = BiologicalConfig {
        adaptive_noise_rate: 0.5,
        ..Default::default()
    };
    
    // Brain with 64 neurons, 16 dimensions, 3 layers
    let mut brain = HarmonicBdh::with_config(64, 16, 3, config);
    
    let mut total_survival = 0;
    let episodes = 5;
    
    for episode in 1..=episodes {
        let obs = env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;
        let mut done = false;
        
        // Initial observation
        let mut current_obs = obs;
        
        while !done {
            // Brain decides action based on observation and current valence (reward)
            // For the first step, reward is 0.0
            let valence = if steps > 0 { 1.0 } else { 0.0 };
            
            let action = brain.forward_embodied(&current_obs, valence);
            
            // Environment responds
            let (next_obs, reward, is_done) = env.step(action);
            
            // Visualization
            if episode == episodes { // Only animate the last episode
                let (_drive_name, drive_status) = brain.drives.get_dominant_drive();
                print!("\rEp {} Step {}: Act {} | {} | Pole {:.4}", 
                    episode, steps, action, drive_status, current_obs[2]);
                thread::sleep(Duration::from_millis(50));
            }
            
            current_obs = next_obs;
            total_reward += reward;
            steps += 1;
            done = is_done;
        }
        
        println!("\nEpisode {} finished. Steps: {}, Total Reward: {:.1}", episode, steps, total_reward);
        total_survival += steps;
    }
    
    println!("\nAverage Survival: {:.1} steps", total_survival as f32 / episodes as f32);
    println!("Random baseline is usually ~20-30 steps.");
}
