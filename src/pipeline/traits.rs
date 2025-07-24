use crate::Result;
use opencv::core::Mat;
use serde_json::Value;
use std::collections::HashMap;

/// Enhanced algorithm trait with preprocessing and configuration support
pub trait AlignmentAlgorithm: Send + Sync {
    /// Returns the name of the algorithm
    fn name(&self) -> &str;

    /// Configure the algorithm with provided parameters
    fn configure(&mut self, _config: &AlgorithmConfig) -> Result<()> {
        // Default implementation - algorithms can override if needed
        Ok(())
    }

    /// Preprocess the image before alignment (optional)
    fn preprocess(&self, image: &Mat) -> Result<Mat> {
        // Default implementation returns the image unchanged
        Ok(image.clone())
    }

    /// Perform the alignment of a patch within a search image
    fn align(&self, search_image: &Mat, patch: &Mat) -> Result<AlignmentResult>;

    /// Check if this algorithm supports GPU acceleration
    fn supports_gpu(&self) -> bool {
        false
    }

    /// Get the computational complexity class of this algorithm
    fn estimated_complexity(&self) -> ComplexityClass {
        ComplexityClass::Medium
    }

    /// Get algorithm-specific parameters for tuning
    fn get_parameters(&self) -> HashMap<String, ParameterInfo> {
        HashMap::new()
    }
}

/// Pipeline stage for composable image processing
pub trait PipelineStage: Send + Sync {
    type Input;
    type Output;

    /// Execute this stage of the pipeline
    fn execute(&self, input: Self::Input) -> Result<Self::Output>;

    /// Check if this stage can be parallelized
    fn can_parallelize(&self) -> bool {
        false
    }

    /// Get the name of this stage for logging/debugging
    fn stage_name(&self) -> &str;
}

/// Image augmentation trait for testing robustness
pub trait ImageAugmentation: Send + Sync {
    /// Apply the augmentation to an image
    fn apply(&self, image: &Mat) -> Result<AugmentedImage>;

    /// Get the inverse transform if available (for ground truth)
    fn get_inverse_transform(&self) -> Option<Transform>;

    /// Get a human-readable description of this augmentation
    fn description(&self) -> String;

    /// Get the parameters of this augmentation
    fn get_params(&self) -> HashMap<String, Value>;
}

/// Metric trait for benchmarking
pub trait Metric: Send + Sync {
    /// Compute the metric value given ground truth and predicted results
    fn compute(&self, ground_truth: &AlignmentResult, predicted: &AlignmentResult) -> f64;

    /// Get the name of this metric
    fn name(&self) -> &str;

    /// Check if higher values are better for this metric
    fn higher_is_better(&self) -> bool {
        true
    }
}

/// Complexity classes for algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityClass {
    Low,    // O(n) or O(n log n)
    Medium, // O(n²)
    High,   // O(n³) or worse
}

/// Configuration for algorithms
#[derive(Debug, Clone, Default)]
pub struct AlgorithmConfig {
    pub parameters: HashMap<String, Value>,
}

impl AlgorithmConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_param<K: Into<String>, V: Into<Value>>(mut self, key: K, value: V) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }

    pub fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.parameters
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }
}

/// Information about an algorithm parameter
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    pub name: String,
    pub description: String,
    pub default_value: Value,
    pub value_type: ParameterType,
    pub range: Option<ParameterRange>,
}

#[derive(Debug, Clone)]
pub enum ParameterType {
    Integer,
    Float,
    Boolean,
    String,
    Choice(Vec<String>),
}

#[derive(Debug, Clone)]
pub enum ParameterRange {
    Integer { min: i64, max: i64 },
    Float { min: f64, max: f64 },
}

// Re-export types that will be defined in types.rs
pub use crate::pipeline::types::{AlignmentResult, AugmentedImage, Transform};
