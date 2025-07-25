use crate::pipeline::builder::{PipelineInput, PipelineOutput};
use crate::pipeline::{AlignmentAlgorithm, ImageAugmentation, PipelineStage};
use crate::Result;
use opencv::core::Mat;
use opencv::prelude::*;

/// Stage that applies an alignment algorithm
pub struct AlignmentStage {
    algorithm: Box<dyn AlignmentAlgorithm>,
}

impl AlignmentStage {
    pub fn new(algorithm: Box<dyn AlignmentAlgorithm>) -> Self {
        Self { algorithm }
    }
}

impl PipelineStage for AlignmentStage {
    type Input = PipelineInput;
    type Output = PipelineOutput;

    fn execute(&self, input: Self::Input) -> Result<Self::Output> {
        match input {
            PipelineInput::ImagePair { search, patch } => {
                let result = self.algorithm.align(&search, &patch)?;
                Ok(PipelineOutput::AlignmentResult(Box::new(result)))
            }
            _ => Err(anyhow::anyhow!("AlignmentStage requires ImagePair input")),
        }
    }

    fn stage_name(&self) -> &str {
        self.algorithm.name()
    }
}

/// Stage that applies image augmentation
pub struct AugmentationStage {
    augmentation: Box<dyn ImageAugmentation>,
}

impl AugmentationStage {
    pub fn new(augmentation: Box<dyn ImageAugmentation>) -> Self {
        Self { augmentation }
    }
}

impl PipelineStage for AugmentationStage {
    type Input = PipelineInput;
    type Output = PipelineOutput;

    fn execute(&self, input: Self::Input) -> Result<Self::Output> {
        match input {
            PipelineInput::Image(image) => {
                let augmented = self.augmentation.apply(&image)?;
                Ok(PipelineOutput::Image(augmented.image))
            }
            _ => Err(anyhow::anyhow!("AugmentationStage requires Image input")),
        }
    }

    fn stage_name(&self) -> &str {
        "Augmentation"
    }
}

/// Stage that preprocesses images
pub struct PreprocessingStage {
    operations: Vec<PreprocessOp>,
}

#[derive(Clone)]
pub enum PreprocessOp {
    GaussianBlur { sigma: f64 },
    HistogramEqualization,
    Normalize,
    Resize { scale: f64 },
}

impl Default for PreprocessingStage {
    fn default() -> Self {
        Self::new()
    }
}

impl PreprocessingStage {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn add_operation(mut self, op: PreprocessOp) -> Self {
        self.operations.push(op);
        self
    }
}

impl PipelineStage for PreprocessingStage {
    type Input = PipelineInput;
    type Output = PipelineOutput;

    fn execute(&self, input: Self::Input) -> Result<Self::Output> {
        match input {
            PipelineInput::Image(mut image) => {
                for op in &self.operations {
                    image = self.apply_operation(image, op)?;
                }
                Ok(PipelineOutput::Image(image))
            }
            _ => Err(anyhow::anyhow!("PreprocessingStage requires Image input")),
        }
    }

    fn stage_name(&self) -> &str {
        "Preprocessing"
    }
}

impl PreprocessingStage {
    fn apply_operation(&self, image: Mat, op: &PreprocessOp) -> Result<Mat> {
        use opencv::{core::Size, imgproc};

        match op {
            PreprocessOp::GaussianBlur { sigma } => {
                let mut blurred = Mat::default();
                let ksize = Size::new(0, 0); // Auto-calculate from sigma
                imgproc::gaussian_blur(
                    &image,
                    &mut blurred,
                    ksize,
                    *sigma,
                    *sigma,
                    opencv::core::BORDER_DEFAULT,
                    opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )?;
                Ok(blurred)
            }
            PreprocessOp::HistogramEqualization => {
                let mut equalized = Mat::default();
                imgproc::equalize_hist(&image, &mut equalized)?;
                Ok(equalized)
            }
            PreprocessOp::Normalize => {
                let mut normalized = Mat::default();
                opencv::core::normalize(
                    &image,
                    &mut normalized,
                    0.0,
                    255.0,
                    opencv::core::NORM_MINMAX,
                    -1,
                    &opencv::core::no_array(),
                )?;
                Ok(normalized)
            }
            PreprocessOp::Resize { scale } => {
                let mut resized = Mat::default();
                let new_size = Size::new(
                    (image.cols() as f64 * scale) as i32,
                    (image.rows() as f64 * scale) as i32,
                );
                imgproc::resize(
                    &image,
                    &mut resized,
                    new_size,
                    0.0,
                    0.0,
                    imgproc::INTER_LINEAR,
                )?;
                Ok(resized)
            }
        }
    }
}

/// Stage that validates alignment results
pub struct ValidationStage {
    min_confidence: f64,
    max_error_pixels: f64,
}

impl Default for ValidationStage {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationStage {
    pub fn new() -> Self {
        Self {
            min_confidence: 0.5,
            max_error_pixels: 10.0,
        }
    }

    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    pub fn with_max_error(mut self, error: f64) -> Self {
        self.max_error_pixels = error;
        self
    }
}

impl PipelineStage for ValidationStage {
    type Input = PipelineInput;
    type Output = PipelineOutput;

    fn execute(&self, input: Self::Input) -> Result<Self::Output> {
        match input {
            PipelineInput::AlignmentResult(result) => {
                if result.confidence < self.min_confidence {
                    return Err(anyhow::anyhow!(
                        "Alignment confidence {} below threshold {}",
                        result.confidence,
                        self.min_confidence
                    ));
                }

                // Additional validation logic here

                Ok(PipelineOutput::AlignmentResult(result))
            }
            _ => Err(anyhow::anyhow!(
                "ValidationStage requires AlignmentResult input"
            )),
        }
    }

    fn stage_name(&self) -> &str {
        "Validation"
    }
}
