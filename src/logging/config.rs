//! Logging configuration system
//!
//! Provides comprehensive configuration options for the logging system,
//! including per-component log levels, output destinations, and performance settings.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use super::rotation::RotationConfig;

/// Comprehensive logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Global log level (trace, debug, info, warn, error)
    pub global_level: String,
    
    /// Enable console output
    pub console_output: bool,
    
    /// Directory for log files (None = no file logging)
    pub log_directory: Option<PathBuf>,
    
    /// Include file location in logs (impacts performance)
    pub include_file_location: bool,
    
    /// Algorithm-specific log level
    pub algorithm_level: String,
    
    /// Pipeline stage log level
    pub pipeline_level: String,
    
    /// Testing framework log level
    pub testing_level: String,
    
    /// Dashboard log level
    pub dashboard_level: String,
    
    /// Log rotation configuration
    pub rotation: RotationConfig,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            global_level: "info".to_string(),
            console_output: true,
            log_directory: None,
            include_file_location: false,
            algorithm_level: "info".to_string(),
            pipeline_level: "info".to_string(),
            testing_level: "debug".to_string(),
            dashboard_level: "info".to_string(),
            rotation: RotationConfig::default(),
        }
    }
}

impl LoggingConfig {
    /// Create a development configuration with verbose logging
    pub fn development() -> Self {
        Self {
            global_level: "debug".to_string(),
            console_output: true,
            log_directory: Some(PathBuf::from("logs")),
            include_file_location: true,
            algorithm_level: "trace".to_string(),
            pipeline_level: "debug".to_string(),
            testing_level: "trace".to_string(),
            dashboard_level: "debug".to_string(),
            rotation: RotationConfig::default(),
        }
    }

    /// Create a production configuration with minimal overhead
    pub fn production() -> Self {
        Self {
            global_level: "warn".to_string(),
            console_output: false,
            log_directory: Some(PathBuf::from("/var/log/image-alignment")),
            include_file_location: false,
            algorithm_level: "info".to_string(),
            pipeline_level: "info".to_string(),
            testing_level: "info".to_string(),
            dashboard_level: "warn".to_string(),
            rotation: RotationConfig {
                max_file_size: 500 * 1024 * 1024, // 500MB for production
                max_file_age_hours: 168, // 7 days
                max_files: 30,
                archive_directory: Some(PathBuf::from("/var/log/image-alignment/archive")),
                compress_archives: true,
            },
        }
    }

    /// Create a performance testing configuration
    pub fn performance_testing() -> Self {
        Self {
            global_level: "info".to_string(),
            console_output: true,
            log_directory: Some(PathBuf::from("performance_logs")),
            include_file_location: false,
            algorithm_level: "info".to_string(),
            pipeline_level: "debug".to_string(),
            testing_level: "info".to_string(),
            dashboard_level: "info".to_string(),
            rotation: RotationConfig {
                max_file_size: 50 * 1024 * 1024, // 50MB for performance testing
                max_file_age_hours: 24, // 1 day
                max_files: 5,
                archive_directory: Some(PathBuf::from("performance_logs/archive")),
                compress_archives: true,
            },
        }
    }

    /// Validate the configuration and provide helpful error messages
    pub fn validate(&self) -> Result<(), String> {
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        
        if !valid_levels.contains(&self.global_level.as_str()) {
            return Err(format!("Invalid global_level: {}. Must be one of: {:?}", 
                             self.global_level, valid_levels));
        }
        
        if !valid_levels.contains(&self.algorithm_level.as_str()) {
            return Err(format!("Invalid algorithm_level: {}. Must be one of: {:?}", 
                             self.algorithm_level, valid_levels));
        }
        
        if !valid_levels.contains(&self.pipeline_level.as_str()) {
            return Err(format!("Invalid pipeline_level: {}. Must be one of: {:?}", 
                             self.pipeline_level, valid_levels));
        }
        
        if !valid_levels.contains(&self.testing_level.as_str()) {
            return Err(format!("Invalid testing_level: {}. Must be one of: {:?}", 
                             self.testing_level, valid_levels));
        }
        
        if !valid_levels.contains(&self.dashboard_level.as_str()) {
            return Err(format!("Invalid dashboard_level: {}. Must be one of: {:?}", 
                             self.dashboard_level, valid_levels));
        }

        // Validate log directory if specified
        if let Some(ref log_dir) = self.log_directory {
            if let Some(parent) = log_dir.parent() {
                if !parent.exists() {
                    return Err(format!("Log directory parent does not exist: {:?}", parent));
                }
            }
        }

        Ok(())
    }

    /// Get the effective log level for a specific component
    pub fn get_component_level(&self, component: &str) -> &str {
        match component {
            "algorithm" | "algorithms" => &self.algorithm_level,
            "pipeline" => &self.pipeline_level,
            "testing" | "test" => &self.testing_level,
            "dashboard" => &self.dashboard_level,
            _ => &self.global_level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LoggingConfig::default();
        assert_eq!(config.global_level, "info");
        assert!(config.console_output);
        assert!(config.log_directory.is_none());
        assert!(!config.include_file_location);
    }

    #[test]
    fn test_development_config() {
        let config = LoggingConfig::development();
        assert_eq!(config.global_level, "debug");
        assert_eq!(config.algorithm_level, "trace");
        assert!(config.include_file_location);
        assert!(config.log_directory.is_some());
    }

    #[test]
    fn test_production_config() {
        let config = LoggingConfig::production();
        assert_eq!(config.global_level, "warn");
        assert!(!config.console_output);
        assert!(!config.include_file_location);
    }

    #[test]
    fn test_config_validation() {
        let mut config = LoggingConfig::default();
        assert!(config.validate().is_ok());

        config.global_level = "invalid".to_string();
        assert!(config.validate().is_err());

        config.global_level = "debug".to_string();
        config.algorithm_level = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_component_level_selection() {
        let config = LoggingConfig::development();
        assert_eq!(config.get_component_level("algorithm"), "trace");
        assert_eq!(config.get_component_level("pipeline"), "debug");
        assert_eq!(config.get_component_level("testing"), "trace");
        assert_eq!(config.get_component_level("unknown"), "debug");
    }
}