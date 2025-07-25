use crate::logging::{PipelineSpan, get_correlation_id, new_correlation_id};
use crate::pipeline::{PipelineContext, PipelineStage, StageTime};
use crate::Result;
use opencv::prelude::MatTraitConst;
use std::time::Instant;
use tracing::{info, debug, error};

/// Builder for creating image processing pipelines
pub struct PipelineBuilder {
    stages: Vec<Box<dyn PipelineStage<Input = PipelineInput, Output = PipelineOutput>>>,
    name: String,
}

/// Input type for pipeline stages
#[derive(Clone)]
pub enum PipelineInput {
    Image(opencv::core::Mat),
    ImagePair {
        search: opencv::core::Mat,
        patch: opencv::core::Mat,
    },
    AlignmentResult(Box<crate::pipeline::AlignmentResult>),
    Multiple(Vec<PipelineInput>),
}

/// Output type for pipeline stages
#[derive(Clone)]
pub enum PipelineOutput {
    Image(opencv::core::Mat),
    AlignmentResult(Box<crate::pipeline::AlignmentResult>),
    Multiple(Vec<PipelineOutput>),
    Report(serde_json::Value),
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            stages: Vec::new(),
            name: name.into(),
        }
    }

    /// Add a stage to the pipeline
    pub fn add_stage<S>(mut self, stage: S) -> Self
    where
        S: PipelineStage<Input = PipelineInput, Output = PipelineOutput> + 'static,
    {
        self.stages.push(Box::new(stage));
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Pipeline {
        Pipeline {
            stages: self.stages,
            name: self.name,
        }
    }
}

/// Executable pipeline
pub struct Pipeline {
    stages: Vec<Box<dyn PipelineStage<Input = PipelineInput, Output = PipelineOutput>>>,
    name: String,
}

impl Pipeline {
    /// Execute the pipeline with the given input
    pub fn execute(&self, input: PipelineInput) -> Result<PipelineOutput> {
        // Ensure we have a correlation ID for this pipeline execution
        let correlation_id = get_correlation_id().unwrap_or_else(|| new_correlation_id());
        
        info!(
            pipeline = %self.name,
            total_stages = self.stages.len(),
            correlation_id = %correlation_id,
            "Starting pipeline execution"
        );

        let mut context = PipelineContext {
            stage_index: 0,
            total_stages: self.stages.len(),
            stage_timings: Vec::new(),
            messages: Vec::new(),
            shared_data: std::collections::HashMap::new(),
        };

        let mut current_input = input;

        for (idx, stage) in self.stages.iter().enumerate() {
            context.stage_index = idx;
            
            let stage_name = stage.stage_name().to_string();
            
            // Create span for this pipeline stage
            let pipeline_span = PipelineSpan::new(&stage_name, Some(correlation_id));
            let _span_guard = pipeline_span.enter();

            let start = Instant::now();

            // Log input metadata
            let input_description = match &current_input {
                PipelineInput::Image(img) => {
                    pipeline_span.record_input("image", Some((img.cols() as u32, img.rows() as u32)));
                    format!("Image({}x{})", img.cols(), img.rows())
                }
                PipelineInput::ImagePair { search, patch } => {
                    pipeline_span.record_input("image_pair", Some((search.cols() as u32, search.rows() as u32)));
                    format!("ImagePair(search:{}x{}, patch:{}x{})", 
                           search.cols(), search.rows(), patch.cols(), patch.rows())
                }
                PipelineInput::AlignmentResult(_) => {
                    pipeline_span.record_input("alignment_result", None);
                    "AlignmentResult".to_string()
                }
                PipelineInput::Multiple(inputs) => {
                    pipeline_span.record_input("multiple", None);
                    format!("Multiple({})", inputs.len())
                }
            };

            debug!(
                stage = %stage_name,
                stage_index = idx,
                input_type = %input_description,
                "Executing pipeline stage"
            );

            // Execute the stage
            match stage.execute(current_input) {
                Ok(output) => {
                    let duration = start.elapsed().as_secs_f64() * 1000.0;
                    
                    context.stage_timings.push(StageTime {
                        stage_name: stage_name.clone(),
                        duration_ms: duration,
                    });

                    // Log output metadata
                    let output_description = match &output {
                        PipelineOutput::Image(img) => {
                            format!("Image({}x{})", img.cols(), img.rows())
                        }
                        PipelineOutput::AlignmentResult(result) => {
                            format!("AlignmentResult(confidence: {:.3})", result.confidence)
                        }
                        PipelineOutput::Multiple(outputs) => {
                            format!("Multiple({})", outputs.len())
                        }
                        PipelineOutput::Report(_) => "Report".to_string(),
                    };

                    pipeline_span.record_completion(&output_description, true);

                    info!(
                        stage = %stage_name,
                        duration_ms = duration,
                        output_type = %output_description,
                        "Pipeline stage completed successfully"
                    );

                    // Convert output to input for next stage
                    current_input = match output {
                        PipelineOutput::Image(img) => PipelineInput::Image(img),
                        PipelineOutput::AlignmentResult(result) => {
                            PipelineInput::AlignmentResult(result)
                        }
                        PipelineOutput::Multiple(outputs) => PipelineInput::Multiple(
                            outputs
                                .into_iter()
                                .map(|o| match o {
                                    PipelineOutput::Image(img) => PipelineInput::Image(img),
                                    PipelineOutput::AlignmentResult(r) => {
                                        PipelineInput::AlignmentResult(r)
                                    }
                                    _ => PipelineInput::Multiple(vec![]),
                                })
                                .collect(),
                        ),
                        PipelineOutput::Report(report) => {
                            // Report is final output, return it
                            info!(
                                pipeline = %self.name,
                                total_duration_ms = context.stage_timings.iter().map(|t| t.duration_ms).sum::<f64>(),
                                "Pipeline execution completed with report"
                            );
                            return Ok(PipelineOutput::Report(report));
                        }
                    };
                }
                Err(e) => {
                    let duration = start.elapsed().as_secs_f64() * 1000.0;
                    
                    pipeline_span.record_completion("error", false);
                    
                    error!(
                        stage = %stage_name,
                        duration_ms = duration,
                        error = %e,
                        "Pipeline stage failed"
                    );
                    
                    context.messages.push(crate::pipeline::PipelineMessage {
                        level: crate::pipeline::MessageLevel::Error,
                        stage: stage_name,
                        message: e.to_string(),
                    });
                    
                    return Err(e);
                }
            }
        }

        // Convert final input back to output
        let final_output = match current_input {
            PipelineInput::Image(img) => PipelineOutput::Image(img),
            PipelineInput::AlignmentResult(result) => PipelineOutput::AlignmentResult(result),
            PipelineInput::Multiple(inputs) => PipelineOutput::Multiple(
                inputs
                    .into_iter()
                    .map(|i| match i {
                        PipelineInput::Image(img) => PipelineOutput::Image(img),
                        PipelineInput::AlignmentResult(r) => PipelineOutput::AlignmentResult(r),
                        _ => PipelineOutput::Multiple(vec![]),
                    })
                    .collect(),
            ),
            _ => {
                return Err(anyhow::anyhow!("Unexpected pipeline state"));
            }
        };

        let total_duration = context.stage_timings.iter().map(|t| t.duration_ms).sum::<f64>();
        
        info!(
            pipeline = %self.name,
            total_duration_ms = total_duration,
            stages_executed = context.stage_timings.len(),
            correlation_id = %correlation_id,
            "Pipeline execution completed successfully"
        );

        Ok(final_output)
    }

    /// Get pipeline name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get number of stages
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }
}

/// Helper macro to create pipeline stages
#[macro_export]
macro_rules! pipeline_stage {
    ($name:ident, $input:ty, $output:ty, $body:expr) => {
        struct $name;

        impl PipelineStage for $name {
            type Input = $input;
            type Output = $output;

            fn execute(&self, input: Self::Input) -> Result<Self::Output> {
                $body(input)
            }

            fn stage_name(&self) -> &str {
                stringify!($name)
            }
        }
    };
}
