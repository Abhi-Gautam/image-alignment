//! Log file rotation and retention policies
//!
//! Provides automatic log file rotation based on size and time,
//! with configurable retention policies to manage disk space.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tracing::{info, warn, error};
use serde::{Deserialize, Serialize};

/// Log rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationConfig {
    /// Maximum file size before rotation (in bytes)
    pub max_file_size: u64,
    
    /// Maximum age before rotation (in hours)
    pub max_file_age_hours: u64,
    
    /// Maximum number of log files to keep
    pub max_files: usize,
    
    /// Directory for archived log files
    pub archive_directory: Option<PathBuf>,
    
    /// Enable compression for archived files
    pub compress_archives: bool,
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self {
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_file_age_hours: 24, // 1 day
            max_files: 10,
            archive_directory: None,
            compress_archives: true,
        }
    }
}

/// Log file rotation manager
pub struct LogRotationManager {
    config: RotationConfig,
    log_directory: PathBuf,
    archive_directory: PathBuf,
}

impl LogRotationManager {
    /// Create a new log rotation manager
    pub fn new(log_directory: PathBuf, config: RotationConfig) -> Self {
        let archive_directory = config.archive_directory
            .clone()
            .unwrap_or_else(|| log_directory.join("archive"));
        
        // Ensure directories exist
        if let Err(e) = std::fs::create_dir_all(&log_directory) {
            error!("Failed to create log directory: {}", e);
        }
        
        if let Err(e) = std::fs::create_dir_all(&archive_directory) {
            error!("Failed to create archive directory: {}", e);
        }

        Self {
            config,
            log_directory,
            archive_directory,
        }
    }

    /// Check if a log file needs rotation
    pub fn needs_rotation(&self, log_file: &Path) -> bool {
        if !log_file.exists() {
            return false;
        }

        // Check file size
        if let Ok(metadata) = std::fs::metadata(log_file) {
            if metadata.len() >= self.config.max_file_size {
                info!(
                    file = %log_file.display(),
                    size_mb = metadata.len() / (1024 * 1024),
                    max_size_mb = self.config.max_file_size / (1024 * 1024),
                    "Log file needs rotation due to size"
                );
                return true;
            }

            // Check file age
            if let Ok(created) = metadata.created() {
                let age = SystemTime::now()
                    .duration_since(created)
                    .unwrap_or(Duration::ZERO);
                
                let max_age = Duration::from_secs(self.config.max_file_age_hours * 3600);
                
                if age >= max_age {
                    info!(
                        file = %log_file.display(),
                        age_hours = age.as_secs() / 3600,
                        max_age_hours = self.config.max_file_age_hours,
                        "Log file needs rotation due to age"
                    );
                    return true;
                }
            }
        }

        false
    }

    /// Perform log file rotation
    pub fn rotate_log(&self, log_file: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if !log_file.exists() {
            warn!("Cannot rotate non-existent log file: {}", log_file.display());
            return Ok(());
        }

        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let file_stem = log_file.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("log");
        
        let extension = log_file.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("log");

        let archive_name = if self.config.compress_archives {
            format!("{}_{}.{}.gz", file_stem, timestamp, extension)
        } else {
            format!("{}_{}.{}", file_stem, timestamp, extension)
        };

        let archive_path = self.archive_directory.join(archive_name);

        info!(
            source = %log_file.display(),
            archive = %archive_path.display(),
            compressed = self.config.compress_archives,
            "Rotating log file"
        );

        // Move and optionally compress the log file
        if self.config.compress_archives {
            self.compress_and_archive(log_file, &archive_path)?;
        } else {
            std::fs::rename(log_file, &archive_path)?;
        }

        // Clean up old archives
        self.cleanup_old_archives()?;

        info!("Log rotation completed successfully");
        Ok(())
    }

    /// Compress and archive a log file
    fn compress_and_archive(&self, source: &Path, target: &Path) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::{BufReader, BufWriter};
        use std::fs::File;
        
        let input_file = File::open(source)?;
        let output_file = File::create(target)?;
        
        let mut reader = BufReader::new(input_file);
        let writer = BufWriter::new(output_file);
        
        // Use flate2 for gzip compression
        let mut encoder = flate2::write::GzEncoder::new(writer, flate2::Compression::default());
        std::io::copy(&mut reader, &mut encoder)?;
        encoder.finish()?;

        // Remove original file after successful compression
        std::fs::remove_file(source)?;
        
        Ok(())
    }

    /// Clean up old archived log files
    fn cleanup_old_archives(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut archive_files = Vec::new();

        // Collect all archive files with their metadata
        for entry in std::fs::read_dir(&self.archive_directory)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(created) = metadata.created() {
                        archive_files.push((path, created));
                    }
                }
            }
        }

        // Sort by creation time (oldest first)
        archive_files.sort_by_key(|(_, created)| *created);

        // Remove excess files
        let files_to_remove = archive_files.len().saturating_sub(self.config.max_files);
        
        for (file_path, _) in archive_files.iter().take(files_to_remove) {
            info!(
                file = %file_path.display(),
                "Removing old archive file due to retention policy"
            );
            
            if let Err(e) = std::fs::remove_file(file_path) {
                warn!("Failed to remove archive file {}: {}", file_path.display(), e);
            }
        }

        if files_to_remove > 0 {
            info!("Cleaned up {} old archive files", files_to_remove);
        }

        Ok(())
    }

    /// Perform maintenance on all log files in the directory
    pub fn perform_maintenance(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!(
            log_dir = %self.log_directory.display(),
            "Starting log maintenance"
        );

        for entry in std::fs::read_dir(&self.log_directory)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() && path.extension().map_or(false, |ext| ext == "log") {
                if self.needs_rotation(&path) {
                    if let Err(e) = self.rotate_log(&path) {
                        error!(
                            file = %path.display(),
                            error = %e,
                            "Failed to rotate log file"
                        );
                    }
                }
            }
        }

        // Also clean up archives in case retention policy changed
        self.cleanup_old_archives()?;

        info!("Log maintenance completed");
        Ok(())
    }

    /// Get statistics about log files and archives
    pub fn get_statistics(&self) -> LogStatistics {
        let mut active_files = 0;
        let mut active_size = 0;
        let mut archive_files = 0;
        let mut archive_size = 0;

        // Count active log files
        if let Ok(entries) = std::fs::read_dir(&self.log_directory) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        active_files += 1;
                        active_size += metadata.len();
                    }
                }
            }
        }

        // Count archive files
        if let Ok(entries) = std::fs::read_dir(&self.archive_directory) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        archive_files += 1;
                        archive_size += metadata.len();
                    }
                }
            }
        }

        LogStatistics {
            active_files,
            active_size_bytes: active_size,
            archive_files,
            archive_size_bytes: archive_size,
            log_directory: self.log_directory.clone(),
            archive_directory: self.archive_directory.clone(),
        }
    }
}

/// Statistics about log files and archives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogStatistics {
    pub active_files: usize,
    pub active_size_bytes: u64,
    pub archive_files: usize,
    pub archive_size_bytes: u64,
    pub log_directory: PathBuf,
    pub archive_directory: PathBuf,
}

impl LogStatistics {
    /// Get total size in megabytes
    pub fn total_size_mb(&self) -> f64 {
        (self.active_size_bytes + self.archive_size_bytes) as f64 / (1024.0 * 1024.0)
    }

    /// Get active size in megabytes
    pub fn active_size_mb(&self) -> f64 {
        self.active_size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get archive size in megabytes
    pub fn archive_size_mb(&self) -> f64 {
        self.archive_size_bytes as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_rotation_config_default() {
        let config = RotationConfig::default();
        assert_eq!(config.max_file_size, 100 * 1024 * 1024);
        assert_eq!(config.max_file_age_hours, 24);
        assert_eq!(config.max_files, 10);
        assert!(config.compress_archives);
    }

    #[test]
    fn test_needs_rotation_size() {
        let temp_dir = TempDir::new().unwrap();
        let config = RotationConfig {
            max_file_size: 100, // Very small for testing
            ..Default::default()
        };
        
        let manager = LogRotationManager::new(temp_dir.path().to_path_buf(), config);
        
        // Create a file larger than the limit
        let log_file = temp_dir.path().join("test.log");
        let mut file = File::create(&log_file).unwrap();
        file.write_all(&vec![b'x'; 200]).unwrap(); // 200 bytes
        
        assert!(manager.needs_rotation(&log_file));
    }

    #[test]
    fn test_log_statistics() {
        let temp_dir = TempDir::new().unwrap();
        let config = RotationConfig::default();
        let manager = LogRotationManager::new(temp_dir.path().to_path_buf(), config);
        
        // Create a test log file
        let log_file = temp_dir.path().join("test.log");
        let mut file = File::create(&log_file).unwrap();
        file.write_all(b"test log content").unwrap();
        
        let stats = manager.get_statistics();
        assert_eq!(stats.active_files, 1);
        assert!(stats.active_size_bytes > 0);
        assert!(stats.active_size_mb() > 0.0);
    }
}