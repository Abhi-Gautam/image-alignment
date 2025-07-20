pub mod algorithms;
pub mod data;
pub mod analysis;
pub mod visualization;

pub use algorithms::*;
pub use data::*;
pub use analysis::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AlignmentResult {
    pub translation: (f32, f32),
    pub rotation: f32,
    pub scale: f32,
    pub confidence: f32,
    pub processing_time_ms: f32,
    pub algorithm_used: String,
}

impl AlignmentResult {
    pub fn new(algorithm: &str) -> Self {
        Self {
            translation: (0.0, 0.0),
            rotation: 0.0,
            scale: 1.0,
            confidence: 0.0,
            processing_time_ms: 0.0,
            algorithm_used: algorithm.to_string(),
        }
    }
}

pub type Result<T> = anyhow::Result<T>;

#[cfg(test)]
mod tests {
    // No unit tests in lib.rs - all tests are in tests/ directory
}