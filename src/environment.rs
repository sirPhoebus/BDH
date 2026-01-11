use rand::prelude::*;
use std::f32::consts::PI;

/// A simple trait for reinforcement learning environments
pub trait Environment {
    /// Reset the environment to initial state, returning the initial observation
    fn reset(&mut self) -> Vec<f32>;
    
    /// Take an action, returning (observation, reward, done)
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool);
    
    /// Get observation dimension
    fn obs_dim(&self) -> usize;
    
    /// Get action dimension (discrete)
    fn action_dim(&self) -> usize;
}

/// Classic CartPole control problem
/// Based on: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
#[derive(Debug, Clone)]
pub struct CartPole {
    gravity: f32,
    mass_cart: f32,
    mass_pole: f32,
    total_mass: f32,
    length: f32, // actually half the pole's length
    pole_mass_length: f32,
    force_mag: f32,
    tau: f32, // seconds between state updates
    
    // State: [x, x_dot, theta, theta_dot]
    state: [f32; 4],
    
    // Limits
    theta_threshold_radians: f32,
    x_threshold: f32,
    max_steps: usize,
    steps_taken: usize,
}

impl Default for CartPole {
    fn default() -> Self {
        let mass_cart = 1.0;
        let mass_pole = 0.1;
        
        Self {
            gravity: 9.8,
            mass_cart,
            mass_pole,
            total_mass: mass_cart + mass_pole,
            length: 0.5,
            pole_mass_length: mass_pole * 0.5,
            force_mag: 10.0,
            tau: 0.02,
            state: [0.0; 4],
            theta_threshold_radians: 12.0 * 2.0 * PI / 360.0, // 12 degrees
            x_threshold: 2.4,
            max_steps: 500,
            steps_taken: 0,
        }
    }
}

impl Environment for CartPole {
    fn reset(&mut self) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        self.state = [
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
        ];
        self.steps_taken = 0;
        self.state.to_vec()
    }
    
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        let x = self.state[0];
        let x_dot = self.state[1];
        let theta = self.state[2];
        let theta_dot = self.state[3];
        
        // Force direction: 0 = left, 1 = right
        let force = if action == 1 { self.force_mag } else { -self.force_mag };
        
        let costheta = theta.cos();
        let sintheta = theta.sin();
        
        let temp = (force + self.pole_mass_length * theta_dot.powi(2) * sintheta) / self.total_mass;
        let thetaacc = (self.gravity * sintheta - costheta * temp) /
            (self.length * (4.0 / 3.0 - self.mass_pole * costheta.powi(2) / self.total_mass));
        let xacc = temp - self.pole_mass_length * thetaacc * costheta / self.total_mass;
        
        // Euler integration
        let x_next = x + self.tau * x_dot;
        let x_dot_next = x_dot + self.tau * xacc;
        let theta_next = theta + self.tau * theta_dot;
        let theta_dot_next = theta_dot + self.tau * thetaacc;
        
        self.state = [x_next, x_dot_next, theta_next, theta_dot_next];
        self.steps_taken += 1;
        
        // Check termination
        let done = x_next < -self.x_threshold || 
                   x_next > self.x_threshold || 
                   theta_next < -self.theta_threshold_radians || 
                   theta_next > self.theta_threshold_radians ||
                   self.steps_taken >= self.max_steps;
        
        // Reward: +1.0 for every step taken (survival)
        // Optionally add slight penalty for distance/angle to encourage centering
        let reward = if done && self.steps_taken < self.max_steps { 0.0 } else { 1.0 };
        
        (self.state.to_vec(), reward, done)
    }
    
    fn obs_dim(&self) -> usize {
        4
    }
    
    fn action_dim(&self) -> usize {
        2
    }
}


