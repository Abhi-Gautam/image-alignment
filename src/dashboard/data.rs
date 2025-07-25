use crate::config::DashboardConfig;
use crate::pipeline::AlignmentResult as PipelineAlignmentResult;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Dashboard data structures for organizing test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub test_sessions: Vec<TestSession>,
    pub algorithms: Vec<String>,
    pub patch_sizes: Vec<String>,
    pub transformations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSession {
    pub id: String,
    pub name: String,
    pub sem_image_path: String,
    pub created_at: String,
    pub total_tests: usize,
    pub success_rate: f32,
    pub avg_processing_time: f32,
    pub test_results: Vec<TestResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_id: String,
    pub algorithm_name: String,
    pub original_image_path: String,
    pub patch_info: PatchInfo,
    pub transformation_applied: TransformationInfo,
    pub alignment_result: PipelineAlignmentResult,
    pub visual_outputs: VisualOutputs,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchInfo {
    pub size: (u32, u32),
    pub location: (u32, u32),
    pub patch_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationInfo {
    pub noise_type: String,
    pub noise_parameters: String,
    pub rotation_deg: f32,
    pub translation: (i32, i32),
    pub scale_factor: f32,
    pub transformed_patch_path: String,
}

// AlignmentResult is now imported from pipeline module

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualOutputs {
    pub overlay_result_path: String,
    pub side_by_side_path: String,
    pub error_heatmap_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub translation_error_px: f32,
    pub processing_time_ms: f32,
    pub confidence_score: f32,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmSummary {
    pub name: String,
    pub total_tests: usize,
    pub success_count: usize,
    pub success_rate: f32,
    pub avg_translation_error: f32,
    pub avg_processing_time: f32,
    pub avg_confidence: f32,
}

/// Dashboard data loader that scans results directory
#[derive(Clone)]
pub struct DashboardDataLoader {
    pub results_dir: PathBuf,
    pub config: DashboardConfig,
}

impl DashboardDataLoader {
    pub fn new(results_dir: PathBuf) -> Self {
        Self { 
            results_dir,
            config: DashboardConfig::default(),
        }
    }

    pub fn with_config(results_dir: PathBuf, config: DashboardConfig) -> Self {
        Self { 
            results_dir,
            config,
        }
    }

    /// Helper function to safely calculate averages
    fn safe_average(values: &[f32]) -> f32 {
        if values.is_empty() { 0.0 } else { values.iter().sum::<f32>() / values.len() as f32 }
    }

    /// Helper function to calculate success rate
    fn calculate_success_rate(total: usize, success_count: usize) -> f32 {
        if total > 0 { success_count as f32 / total as f32 * 100.0 } else { 0.0 }
    }

    /// Load all test sessions from results directory
    pub fn load_dashboard_data(&self) -> Result<DashboardData> {
        let mut test_sessions = Vec::new();
        let mut all_algorithms = std::collections::HashSet::new();
        let mut all_patch_sizes = std::collections::HashSet::new();
        let mut all_transformations = std::collections::HashSet::new();

        // Scan for test session directories
        if self.results_dir.exists() {
            for entry in std::fs::read_dir(&self.results_dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_dir() {
                    if let Some(session_name) = path.file_name().and_then(|n| n.to_str()) {
                        if session_name.starts_with("test") {
                            if let Ok(session) = self.load_test_session(&path, session_name) {
                                // Collect unique values
                                for test in &session.test_results {
                                    all_algorithms.insert(test.algorithm_name.clone());
                                    all_patch_sizes.insert(format!(
                                        "{}x{}",
                                        test.patch_info.size.0, test.patch_info.size.1
                                    ));
                                    all_transformations.insert(
                                        test.transformation_applied.noise_parameters.clone(),
                                    );
                                }
                                test_sessions.push(session);
                            }
                        }
                    }
                }
            }
        }

        // Sort sessions by creation time (newest first) and limit if configured
        test_sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        
        // Apply results per page limit if configured
        if self.config.results_per_page > 0 && test_sessions.len() > self.config.results_per_page {
            test_sessions.truncate(self.config.results_per_page);
        }

        Ok(DashboardData {
            test_sessions,
            algorithms: all_algorithms.into_iter().collect(),
            patch_sizes: all_patch_sizes.into_iter().collect(),
            transformations: all_transformations.into_iter().collect(),
        })
    }

    /// Load a single test session from a directory
    fn load_test_session(&self, session_dir: &Path, session_name: &str) -> Result<TestSession> {
        let mut test_results = Vec::new();

        // Find all test_report.json files in subdirectories
        for entry in std::fs::read_dir(session_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let report_path = path.join("test_report.json");
                if report_path.exists() {
                    let content = std::fs::read_to_string(&report_path)?;
                    if let Ok(test_result) = serde_json::from_str::<TestResult>(&content) {
                        test_results.push(test_result);
                    }
                }
            }
        }

        // Calculate session statistics
        let total_tests = test_results.len();
        let success_count = test_results
            .iter()
            .filter(|t| t.performance_metrics.success)
            .count();
        let success_rate = Self::calculate_success_rate(total_tests, success_count);
        let processing_times: Vec<f32> = test_results
            .iter()
            .map(|t| t.alignment_result.execution_time_ms as f32)
            .collect();
        let avg_processing_time = Self::safe_average(&processing_times);

        // Get SEM image path from first test result
        let sem_image_path = test_results
            .first()
            .map(|t| t.original_image_path.clone())
            .unwrap_or_default();

        // Get creation time from directory metadata or use current time
        let created_at = std::fs::metadata(session_dir)
            .and_then(|m| m.created())
            .map(|t| {
                chrono::DateTime::<chrono::Utc>::from(t)
                    .format("%Y-%m-%d %H:%M:%S UTC")
                    .to_string()
            })
            .unwrap_or_else(|_| {
                chrono::Utc::now()
                    .format("%Y-%m-%d %H:%M:%S UTC")
                    .to_string()
            });

        Ok(TestSession {
            id: session_name.to_string(),
            name: session_name.replace("_", " ").to_uppercase(),
            sem_image_path,
            created_at,
            total_tests,
            success_rate,
            avg_processing_time,
            test_results,
        })
    }

    /// Calculate algorithm performance summaries across all sessions
    pub fn calculate_algorithm_summaries(&self, data: &DashboardData) -> Vec<AlgorithmSummary> {
        let mut algorithm_stats: HashMap<String, Vec<&TestResult>> = HashMap::new();

        // Group all test results by algorithm
        for session in &data.test_sessions {
            for test in &session.test_results {
                algorithm_stats
                    .entry(test.algorithm_name.clone())
                    .or_default()
                    .push(test);
            }
        }

        // Calculate summaries with single pass optimization
        algorithm_stats
            .into_iter()
            .map(|(name, tests)| {
                let total_tests = tests.len();
                let success_count = tests
                    .iter()
                    .filter(|t| t.performance_metrics.success)
                    .count();
                let success_rate = Self::calculate_success_rate(total_tests, success_count);

                // Single pass to collect all metrics  
                let mut translation_errors = Vec::with_capacity(total_tests);
                let mut processing_times = Vec::with_capacity(total_tests);
                let mut confidences = Vec::with_capacity(total_tests);
                
                for test in tests {
                    translation_errors.push(test.performance_metrics.translation_error_px);
                    processing_times.push(test.alignment_result.execution_time_ms as f32);
                    confidences.push(test.alignment_result.confidence as f32);
                }

                AlgorithmSummary {
                    name,
                    total_tests,
                    success_count,
                    success_rate,
                    avg_translation_error: Self::safe_average(&translation_errors),
                    avg_processing_time: Self::safe_average(&processing_times),
                    avg_confidence: Self::safe_average(&confidences),
                }
            })
            .collect()
    }
}
