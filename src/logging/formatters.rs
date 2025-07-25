//! Custom log formatters for different output types
//!
//! Provides specialized formatting for console output, JSON files, and dashboard display
//! with consistent structure and readability optimizations.

use serde::Serialize;
use std::fmt;
use tracing::{Event, Subscriber};
use tracing_subscriber::fmt::{format, FormatEvent, FormatFields, FmtContext};
use tracing_subscriber::registry::LookupSpan;

/// Custom console formatter with enhanced readability
pub struct ConsoleFormatter {
    with_colors: bool,
    with_timestamps: bool,
    compact_mode: bool,
}

impl ConsoleFormatter {
    pub fn new() -> Self {
        Self {
            with_colors: true,
            with_timestamps: true,
            compact_mode: false,
        }
    }

    pub fn with_colors(mut self, enabled: bool) -> Self {
        self.with_colors = enabled;
        self
    }

    pub fn with_timestamps(mut self, enabled: bool) -> Self {
        self.with_timestamps = enabled;
        self
    }

    pub fn compact(mut self) -> Self {
        self.compact_mode = true;
        self
    }
}

impl Default for ConsoleFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, N> FormatEvent<S, N> for ConsoleFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let metadata = event.metadata();
        
        // Timestamp
        if self.with_timestamps {
            let now = chrono::Local::now();
            write!(writer, "{} ", now.format("%H:%M:%S%.3f"))?;
        }

        // Level with colors
        let level_str = if self.with_colors {
            match *metadata.level() {
                tracing::Level::ERROR => "\x1b[31mERROR\x1b[0m",
                tracing::Level::WARN => "\x1b[33mWARN \x1b[0m",
                tracing::Level::INFO => "\x1b[32mINFO \x1b[0m",
                tracing::Level::DEBUG => "\x1b[36mDEBUG\x1b[0m",
                tracing::Level::TRACE => "\x1b[35mTRACE\x1b[0m",
            }
        } else {
            match *metadata.level() {
                tracing::Level::ERROR => "ERROR",
                tracing::Level::WARN => "WARN ",
                tracing::Level::INFO => "INFO ",
                tracing::Level::DEBUG => "DEBUG",
                tracing::Level::TRACE => "TRACE",
            }
        };
        write!(writer, "[{}] ", level_str)?;

        // Target/module
        let target = metadata.target();
        if !self.compact_mode && !target.is_empty() {
            // Shorten the target for readability
            let short_target = if let Some(pos) = target.rfind("::") {
                &target[pos + 2..]
            } else {
                target
            };
            write!(writer, "{}: ", short_target)?;
        }

        // Span context
        if let Some(span) = ctx.lookup_current() {
            let mut span_names = Vec::new();
            let mut current_span = Some(span);
            
            while let Some(span) = current_span {
                span_names.push(span.name());
                current_span = span.parent();
            }
            
            if !span_names.is_empty() {
                span_names.reverse();
                if self.compact_mode {
                    // Only show the most specific span
                    write!(writer, "[{}] ", span_names.last().unwrap())?;
                } else {
                    write!(writer, "[{}] ", span_names.join("::"))?;
                }
            }
        }

        // Event fields and message
        ctx.field_format().format_fields(writer.by_ref(), event)?;

        writeln!(writer)
    }
}

/// JSON formatter for structured logging with additional metadata
#[derive(Serialize)]
pub struct StructuredLogEntry {
    timestamp: chrono::DateTime<chrono::Utc>,
    level: String,
    target: String,
    message: String,
    span: Option<String>,
    correlation_id: Option<String>,
    fields: serde_json::Map<String, serde_json::Value>,
    performance_data: Option<PerformanceData>,
}

#[derive(Serialize)]
pub struct PerformanceData {
    operation: Option<String>,
    duration_ms: Option<f64>,
    algorithm: Option<String>,
    patch_location: Option<(u32, u32)>,
    keypoints: Option<usize>,
    matches: Option<usize>,
}

/// Dashboard-specific formatter for real-time display
pub struct DashboardFormatter {
    include_performance_data: bool,
}

impl DashboardFormatter {
    pub fn new(include_performance_data: bool) -> Self {
        Self {
            include_performance_data,
        }
    }

    /// Format log entry for dashboard consumption
    pub fn format_for_dashboard(&self, entry: &StructuredLogEntry) -> serde_json::Value {
        let mut dashboard_entry = serde_json::json!({
            "timestamp": entry.timestamp,
            "level": entry.level,
            "message": entry.message,
            "span": entry.span,
            "correlation_id": entry.correlation_id
        });

        // Add relevant fields for dashboard display
        if let Some(algorithm) = entry.fields.get("algorithm") {
            dashboard_entry["algorithm"] = algorithm.clone();
        }

        if let Some(execution_time) = entry.fields.get("execution_time_ms") {
            dashboard_entry["execution_time_ms"] = execution_time.clone();
        }

        if let Some(confidence) = entry.fields.get("confidence") {
            dashboard_entry["confidence"] = confidence.clone();
        }

        // Include performance data if enabled
        if self.include_performance_data {
            if let Some(perf_data) = &entry.performance_data {
                dashboard_entry["performance"] = serde_json::to_value(perf_data).unwrap();
            }
        }

        dashboard_entry
    }
}

/// Utility functions for log formatting
pub mod utils {
    use super::*;

    /// Extract algorithm name from span context
    pub fn extract_algorithm_from_span(span_name: &str) -> Option<&str> {
        if span_name.contains("algorithm_execution") {
            // Extract algorithm name from span fields if available
            // This would need to be implemented based on actual span structure
            None
        } else {
            None
        }
    }

    /// Format duration for human readability
    pub fn format_duration(duration_ms: f64) -> String {
        if duration_ms < 1.0 {
            format!("{:.2}μs", duration_ms * 1000.0)
        } else if duration_ms < 1000.0 {
            format!("{:.2}ms", duration_ms)
        } else {
            format!("{:.2}s", duration_ms / 1000.0)
        }
    }

    /// Extract performance metrics from log fields
    pub fn extract_performance_data(
        fields: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<PerformanceData> {
        let has_perf_data = fields.contains_key("execution_time_ms")
            || fields.contains_key("algorithm")
            || fields.contains_key("keypoints_detected");

        if !has_perf_data {
            return None;
        }

        Some(PerformanceData {
            operation: fields
                .get("operation")
                .and_then(|v| v.as_str().map(|s| s.to_string())),
            duration_ms: fields
                .get("execution_time_ms")
                .and_then(|v| v.as_f64()),
            algorithm: fields
                .get("algorithm")
                .and_then(|v| v.as_str().map(|s| s.to_string())),
            patch_location: {
                let x = fields.get("patch_x").and_then(|v| v.as_u64()).map(|v| v as u32);
                let y = fields.get("patch_y").and_then(|v| v.as_u64()).map(|v| v as u32);
                match (x, y) {
                    (Some(x), Some(y)) => Some((x, y)),
                    _ => None,
                }
            },
            keypoints: fields
                .get("keypoints_detected")
                .and_then(|v| v.as_u64().map(|v| v as usize)),
            matches: fields
                .get("filtered_matches")
                .and_then(|v| v.as_u64().map(|v| v as usize)),
        })
    }

    /// Create a compact log message for high-frequency operations
    pub fn create_compact_message(
        operation: &str,
        duration_ms: f64,
        success: bool,
    ) -> String {
        let status = if success { "✓" } else { "✗" };
        format!("{} {} ({})", status, operation, format_duration(duration_ms))
    }

    /// Sanitize log message for safe output
    pub fn sanitize_message(message: &str) -> String {
        message
            .chars()
            .filter(|c| c.is_ascii() && !c.is_control() || c.is_whitespace())
            .collect()
    }
}

/// Log level utilities
pub mod levels {
    use tracing::Level;

    /// Convert string to tracing Level
    pub fn from_string(level_str: &str) -> Option<Level> {
        match level_str.to_lowercase().as_str() {
            "trace" => Some(Level::TRACE),
            "debug" => Some(Level::DEBUG),
            "info" => Some(Level::INFO),
            "warn" => Some(Level::WARN),
            "error" => Some(Level::ERROR),
            _ => None,
        }
    }

    /// Convert Level to string
    pub fn to_string(level: &Level) -> &'static str {
        match *level {
            Level::TRACE => "trace",
            Level::DEBUG => "debug",
            Level::INFO => "info",
            Level::WARN => "warn",
            Level::ERROR => "error",
        }
    }

    /// Check if a level should be filtered based on configuration
    pub fn should_log(current_level: &Level, configured_level: &str) -> bool {
        if let Some(config_level) = from_string(configured_level) {
            current_level <= &config_level
        } else {
            true // Default to logging if invalid level
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duration_formatting() {
        assert_eq!(utils::format_duration(0.5), "500.00μs");
        assert_eq!(utils::format_duration(10.0), "10.00ms");
        assert_eq!(utils::format_duration(1500.0), "1.50s");
    }

    #[test]
    fn test_compact_message() {
        let msg = utils::create_compact_message("orb_execution", 23.45, true);
        assert!(msg.contains("✓"));
        assert!(msg.contains("orb_execution"));
        assert!(msg.contains("23.45ms"));
    }

    #[test]
    fn test_level_conversion() {
        assert_eq!(levels::from_string("debug"), Some(Level::DEBUG));
        assert_eq!(levels::to_string(&Level::INFO), "info");
        assert!(levels::should_log(&Level::WARN, "debug"));
        assert!(!levels::should_log(&Level::TRACE, "warn"));
    }

    #[test]
    fn test_message_sanitization() {
        let dirty = "Test\x00message\nwith\tcontrol\x07chars";
        let clean = utils::sanitize_message(dirty);
        assert!(!clean.contains('\x00'));
        assert!(!clean.contains('\x07'));
        assert!(clean.contains('\n')); // Whitespace should be preserved
    }
}