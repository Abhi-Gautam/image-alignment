use opencv::core::{Mat, Rect};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Standardized result for all alignment algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    /// The location of the found patch in search image coordinates
    pub location: SerializableRect,

    /// Alignment quality score (algorithm-specific, typically 0-1)
    pub score: f64,

    /// Confidence level (0-1, where 1 is highest confidence)
    pub confidence: f64,

    /// Processing time in milliseconds
    pub execution_time_ms: f64,

    /// Name of the algorithm used
    pub algorithm_name: String,

    /// Additional algorithm-specific metadata
    pub metadata: HashMap<String, Value>,

    /// Transformation parameters if available
    pub transformation: Option<TransformParams>,
}

/// Serializable rectangle for JSON compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl From<Rect> for SerializableRect {
    fn from(rect: Rect) -> Self {
        Self {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
        }
    }
}

impl From<SerializableRect> for Rect {
    fn from(rect: SerializableRect) -> Self {
        Rect::new(rect.x, rect.y, rect.width, rect.height)
    }
}

/// Transformation parameters for alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformParams {
    pub translation: (f32, f32),
    pub rotation_degrees: f32,
    pub scale: f32,
    pub skew: Option<(f32, f32)>,
}

/// Augmented image with ground truth information
#[derive(Debug, Clone)]
pub struct AugmentedImage {
    /// The augmented image
    pub image: Mat,

    /// Original image before augmentation
    pub original: Mat,

    /// Ground truth transformation
    pub ground_truth: GroundTruth,

    /// List of augmentations applied
    pub augmentations_applied: Vec<String>,

    /// Parameters of each augmentation
    pub augmentation_params: HashMap<String, Value>,
}

/// Ground truth information for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth {
    /// Expected location in the search image
    pub expected_location: SerializableRect,

    /// Transformation that was applied
    pub transformation: TransformParams,

    /// Any additional metadata
    pub metadata: HashMap<String, Value>,
}

/// Generic transformation representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    /// 3x3 transformation matrix
    pub matrix: [[f64; 3]; 3],

    /// Type of transformation
    pub transform_type: TransformType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformType {
    Translation,
    Rotation,
    Scale,
    Affine,
    Perspective,
    Elastic,
    Combined,
}

/// Pipeline execution context
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// Current stage index
    pub stage_index: usize,

    /// Total stages
    pub total_stages: usize,

    /// Timing information
    pub stage_timings: Vec<StageTime>,

    /// Any errors or warnings
    pub messages: Vec<PipelineMessage>,

    /// Shared data between stages
    pub shared_data: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTime {
    pub stage_name: String,
    pub duration_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMessage {
    pub level: MessageLevel,
    pub stage: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageLevel {
    Info,
    Warning,
    Error,
}

/// Test case for benchmarking
#[derive(Debug, Clone)]
pub struct TestCase {
    /// Name of the test case
    pub name: String,

    /// Search image
    pub search_image: Mat,

    /// Patch to find
    pub patch: Mat,

    /// Ground truth result
    pub ground_truth: AlignmentResult,

    /// Augmentations applied
    pub augmentations: Vec<String>,
}

/// Benchmark result for a single algorithm on a test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub algorithm_name: String,
    pub test_case_name: String,
    pub success: bool,
    pub metrics: HashMap<String, f64>,
    pub execution_time_ms: f64,
    pub error_message: Option<String>,
}

impl AlignmentResult {
    pub fn new(algorithm_name: &str) -> Self {
        Self {
            location: SerializableRect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
            score: 0.0,
            confidence: 0.0,
            execution_time_ms: 0.0,
            algorithm_name: algorithm_name.to_string(),
            metadata: HashMap::new(),
            transformation: None,
        }
    }

    pub fn with_location(mut self, rect: Rect) -> Self {
        self.location = rect.into();
        self
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_metadata<K: Into<String>, V: Into<Value>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}
