//! Comprehensive logging framework for image alignment system
//!
//! This module provides structured logging infrastructure with hierarchical spans,
//! performance metrics, and correlation tracking across algorithm execution.

pub mod config;
pub mod spans;
pub mod metrics;
pub mod formatters;
pub mod dashboard;
pub mod rotation;

use anyhow::Result;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};
use uuid::Uuid;

pub use config::LoggingConfig;
pub use spans::{AlgorithmSpan, PipelineSpan, TestSessionSpan};
pub use metrics::{MetricsCollector, PerformanceStats, PerformanceMeasurement};

thread_local! {
    static CORRELATION_ID: std::cell::RefCell<Option<Uuid>> = std::cell::RefCell::new(None);
}

/// Patch context information for spatial logging
#[derive(Debug, Clone)]
pub struct PatchContext {
    pub extracted_from: (u32, u32),
    pub patch_size: (u32, u32),
    pub source_image_size: (u32, u32),
    pub variance: f32,
}

thread_local! {
    static PATCH_CONTEXT: std::cell::RefCell<Option<PatchContext>> = std::cell::RefCell::new(None);
}

/// Initialize the logging system with the provided configuration
pub fn init_logging(config: &LoggingConfig) -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            let level = match config.global_level.as_str() {
                "trace" => "trace",
                "debug" => "debug", 
                "info" => "info",
                "warn" => "warn",
                "error" => "error",
                _ => "info",
            };
            EnvFilter::new(format!("{}={}", env!("CARGO_PKG_NAME").replace('-', "_"), level))
        });

    let mut layers = Vec::new();

    // Console output layer
    if config.console_output {
        let console_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_line_number(true)
            .with_file(config.include_file_location);
        layers.push(console_layer.boxed());
    }

    // File output layer
    if let Some(ref log_dir) = config.log_directory {
        let file_appender = tracing_appender::rolling::daily(log_dir, "alignment.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
        
        let file_layer = fmt::layer()
            .with_writer(non_blocking)
            .with_ansi(false)
            .json();
        layers.push(file_layer.boxed());
    }

    // Initialize the subscriber
    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .init();

    tracing::info!("Logging system initialized with config: {:?}", config);
    Ok(())
}

/// Set a correlation ID for the current thread
pub fn set_correlation_id(id: Uuid) {
    CORRELATION_ID.with(|correlation_id| {
        *correlation_id.borrow_mut() = Some(id);
    });
}

/// Get the current correlation ID for this thread
pub fn get_correlation_id() -> Option<Uuid> {
    CORRELATION_ID.with(|correlation_id| *correlation_id.borrow())
}

/// Generate a new correlation ID and set it for the current thread
pub fn new_correlation_id() -> Uuid {
    let id = Uuid::new_v4();
    set_correlation_id(id);
    id
}

/// Clear the correlation ID for the current thread
pub fn clear_correlation_id() {
    CORRELATION_ID.with(|correlation_id| {
        *correlation_id.borrow_mut() = None;
    });
}

/// Set patch context for the current thread
pub fn set_patch_context(context: PatchContext) {
    PATCH_CONTEXT.with(|patch_context| {
        *patch_context.borrow_mut() = Some(context);
    });
}

/// Get the current patch context for this thread
pub fn get_patch_context() -> Option<PatchContext> {
    PATCH_CONTEXT.with(|patch_context| patch_context.borrow().clone())
}

/// Clear the patch context for the current thread
pub fn clear_patch_context() {
    PATCH_CONTEXT.with(|patch_context| {
        *patch_context.borrow_mut() = None;
    });
}

/// Create a span with correlation ID automatically included
#[macro_export]
macro_rules! correlation_span {
    ($level:expr, $name:expr) => {
        if let Some(correlation_id) = $crate::logging::get_correlation_id() {
            tracing::span!($level, $name, correlation_id = %correlation_id)
        } else {
            tracing::span!($level, $name)
        }
    };
    ($level:expr, $name:expr, $($field:tt)*) => {
        if let Some(correlation_id) = $crate::logging::get_correlation_id() {
            tracing::span!($level, $name, correlation_id = %correlation_id, $($field)*)
        } else {
            tracing::span!($level, $name, $($field)*)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_correlation_id_management() {
        // Initially no correlation ID
        assert!(get_correlation_id().is_none());

        // Set a correlation ID
        let id = new_correlation_id();
        assert_eq!(get_correlation_id(), Some(id));

        // Clear correlation ID
        clear_correlation_id();
        assert!(get_correlation_id().is_none());
    }

    #[test]
    fn test_logging_config_init() {
        let temp_dir = TempDir::new().unwrap();
        let config = LoggingConfig {
            global_level: "info".to_string(),
            console_output: true,
            log_directory: Some(temp_dir.path().to_path_buf()),
            include_file_location: false,
            algorithm_level: "debug".to_string(),
            pipeline_level: "info".to_string(),
            testing_level: "debug".to_string(),
            dashboard_level: "info".to_string(),
        };

        // This should not panic
        assert!(init_logging(&config).is_ok());
    }
}