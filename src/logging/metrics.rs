//! Performance metrics collection system
//!
//! Provides lightweight performance metrics collection with statistical analysis
//! and minimal overhead for real-time processing requirements.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Individual performance measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    pub operation: String,
    pub duration_ms: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub correlation_id: Option<Uuid>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Statistical summary of performance measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub operation: String,
    pub count: usize,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub std_dev_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

/// Comprehensive performance metrics for algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMetrics {
    pub algorithm_name: String,
    pub total_executions: usize,
    pub successful_executions: usize,
    pub mean_execution_time_ms: f64,
    pub mean_confidence: f64,
    pub feature_detection_stats: Option<PerformanceStats>,
    pub matching_stats: Option<PerformanceStats>,
    pub ransac_stats: Option<PerformanceStats>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Thread-safe metrics collector
pub struct MetricsCollector {
    pub measurements: Arc<Mutex<Vec<PerformanceMeasurement>>>,
    enabled: bool,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(enabled: bool) -> Self {
        Self {
            measurements: Arc::new(Mutex::new(Vec::new())),
            enabled,
        }
    }

    /// Record a performance measurement
    pub fn record(&self, operation: &str, duration: Duration, correlation_id: Option<Uuid>) {
        if !self.enabled {
            return;
        }

        let measurement = PerformanceMeasurement {
            operation: operation.to_string(),
            duration_ms: duration.as_secs_f64() * 1000.0,
            timestamp: chrono::Utc::now(),
            correlation_id,
            metadata: HashMap::new(),
        };

        if let Ok(mut measurements) = self.measurements.lock() {
            measurements.push(measurement);
            
            // Prevent unbounded growth - keep only last 10000 measurements
            if measurements.len() > 10000 {
                measurements.drain(0..5000);
            }
        }
    }

    /// Record a measurement with additional metadata
    pub fn record_with_metadata(
        &self,
        operation: &str,
        duration: Duration,
        correlation_id: Option<Uuid>,
        metadata: HashMap<String, serde_json::Value>,
    ) {
        if !self.enabled {
            return;
        }

        let measurement = PerformanceMeasurement {
            operation: operation.to_string(),
            duration_ms: duration.as_secs_f64() * 1000.0,
            timestamp: chrono::Utc::now(),
            correlation_id,
            metadata,
        };

        if let Ok(mut measurements) = self.measurements.lock() {
            measurements.push(measurement);
            
            if measurements.len() > 10000 {
                measurements.drain(0..5000);
            }
        }
    }

    /// Get all measurements for a specific operation
    pub fn get_measurements(&self, operation: &str) -> Vec<PerformanceMeasurement> {
        if let Ok(measurements) = self.measurements.lock() {
            measurements
                .iter()
                .filter(|m| m.operation == operation)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get measurements by correlation ID
    pub fn get_measurements_by_correlation(&self, correlation_id: Uuid) -> Vec<PerformanceMeasurement> {
        if let Ok(measurements) = self.measurements.lock() {
            measurements
                .iter()
                .filter(|m| m.correlation_id == Some(correlation_id))
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Calculate performance statistics for an operation
    pub fn calculate_stats(&self, operation: &str) -> Option<PerformanceStats> {
        let measurements = self.get_measurements(operation);
        if measurements.is_empty() {
            return None;
        }

        let mut durations: Vec<f64> = measurements.iter().map(|m| m.duration_ms).collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let count = durations.len();
        let sum: f64 = durations.iter().sum();
        let mean = sum / count as f64;

        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d - mean;
                diff * diff
            })
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let median = if count % 2 == 0 {
            (durations[count / 2 - 1] + durations[count / 2]) / 2.0
        } else {
            durations[count / 2]
        };

        let p95_index = ((count as f64) * 0.95) as usize;
        let p99_index = ((count as f64) * 0.99) as usize;

        Some(PerformanceStats {
            operation: operation.to_string(),
            count,
            mean_ms: mean,
            median_ms: median,
            std_dev_ms: std_dev,
            min_ms: durations[0],
            max_ms: durations[count - 1],
            p95_ms: durations[p95_index.min(count - 1)],
            p99_ms: durations[p99_index.min(count - 1)],
        })
    }

    /// Get comprehensive algorithm metrics
    pub fn get_algorithm_metrics(&self, algorithm_name: &str) -> Option<AlgorithmMetrics> {
        let execution_measurements = self.get_measurements(&format!("{}_execution", algorithm_name));
        if execution_measurements.is_empty() {
            return None;
        }

        let total_executions = execution_measurements.len();
        let successful_executions = execution_measurements
            .iter()
            .filter(|m| {
                m.metadata
                    .get("success")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true)
            })
            .count();

        let mean_execution_time: f64 = execution_measurements
            .iter()
            .map(|m| m.duration_ms)
            .sum::<f64>() / total_executions as f64;

        let mean_confidence: f64 = execution_measurements
            .iter()
            .filter_map(|m| {
                m.metadata
                    .get("confidence")
                    .and_then(|v| v.as_f64())
            })
            .sum::<f64>() / total_executions as f64;

        Some(AlgorithmMetrics {
            algorithm_name: algorithm_name.to_string(),
            total_executions,
            successful_executions,
            mean_execution_time_ms: mean_execution_time,
            mean_confidence,
            feature_detection_stats: self.calculate_stats(&format!("{}_feature_detection", algorithm_name)),
            matching_stats: self.calculate_stats(&format!("{}_matching", algorithm_name)),
            ransac_stats: self.calculate_stats(&format!("{}_ransac", algorithm_name)),
            last_updated: chrono::Utc::now(),
        })
    }

    /// Clear all measurements (for testing or memory management)
    pub fn clear(&self) {
        if let Ok(mut measurements) = self.measurements.lock() {
            measurements.clear();
        }
    }

    /// Get total number of measurements
    pub fn measurement_count(&self) -> usize {
        if let Ok(measurements) = self.measurements.lock() {
            measurements.len()
        } else {
            0
        }
    }

    /// Export measurements to JSON
    pub fn export_to_json(&self) -> Result<String, serde_json::Error> {
        if let Ok(measurements) = self.measurements.lock() {
            serde_json::to_string_pretty(&*measurements)
        } else {
            Ok("[]".to_string())
        }
    }
}

/// High-performance timer for measuring execution time
pub struct Timer {
    start: Instant,
    operation: String,
    correlation_id: Option<Uuid>,
    collector: Option<Arc<MetricsCollector>>,
    metadata: HashMap<String, serde_json::Value>,
}

impl Timer {
    /// Start a new timer
    pub fn start(operation: &str, correlation_id: Option<Uuid>) -> Self {
        Self {
            start: Instant::now(),
            operation: operation.to_string(),
            correlation_id,
            collector: None,
            metadata: HashMap::new(),
        }
    }

    /// Start a timer with metrics collection
    pub fn start_with_collector(
        operation: &str,
        correlation_id: Option<Uuid>,
        collector: Arc<MetricsCollector>,
    ) -> Self {
        Self {
            start: Instant::now(),
            operation: operation.to_string(),
            correlation_id,
            collector: Some(collector),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the timer
    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    /// Stop the timer and record the measurement
    pub fn stop(self) -> Duration {
        let duration = self.start.elapsed();
        
        if let Some(collector) = &self.collector {
            if self.metadata.is_empty() {
                collector.record(&self.operation, duration, self.correlation_id);
            } else {
                collector.record_with_metadata(
                    &self.operation,
                    duration,
                    self.correlation_id,
                    self.metadata,
                );
            }
        }

        tracing::debug!(
            operation = %self.operation,
            duration_ms = duration.as_millis(),
            correlation_id = ?self.correlation_id,
            "Timer completed"
        );

        duration
    }
}

/// Global metrics collector instance
lazy_static::lazy_static! {
    static ref GLOBAL_METRICS: MetricsCollector = MetricsCollector::new(true);
}

/// Get the global metrics collector
pub fn global_metrics() -> &'static MetricsCollector {
    &GLOBAL_METRICS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new(true);
        let correlation_id = Uuid::new_v4();

        // Record some measurements
        collector.record("test_operation", Duration::from_millis(100), Some(correlation_id));
        collector.record("test_operation", Duration::from_millis(150), Some(correlation_id));
        collector.record("test_operation", Duration::from_millis(200), None);

        // Check measurements
        let measurements = collector.get_measurements("test_operation");
        assert_eq!(measurements.len(), 3);

        let by_correlation = collector.get_measurements_by_correlation(correlation_id);
        assert_eq!(by_correlation.len(), 2);

        // Check statistics
        let stats = collector.calculate_stats("test_operation").unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.mean_ms - 150.0).abs() < 1.0);
    }

    #[test]
    fn test_timer() {
        let collector = Arc::new(MetricsCollector::new(true));
        let correlation_id = Uuid::new_v4();

        let timer = Timer::start_with_collector("test_timer", Some(correlation_id), collector.clone())
            .with_metadata("test_param", serde_json::json!("test_value"));

        thread::sleep(Duration::from_millis(10));
        let duration = timer.stop();

        assert!(duration >= Duration::from_millis(10));
        assert_eq!(collector.measurement_count(), 1);

        let measurements = collector.get_measurements("test_timer");
        assert_eq!(measurements.len(), 1);
        assert_eq!(measurements[0].correlation_id, Some(correlation_id));
    }

    #[test]
    fn test_disabled_collector() {
        let collector = MetricsCollector::new(false);
        collector.record("test", Duration::from_millis(100), None);
        assert_eq!(collector.measurement_count(), 0);
    }
}