//! Dashboard integration for live logging and metrics visualization
//!
//! Provides real-time logging endpoints, WebSocket connections for live updates,
//! and integration with the existing web dashboard for comprehensive monitoring.

use crate::logging::metrics::{MetricsCollector, PerformanceMeasurement};
use axum::{
    extract::{Query, State, WebSocketUpgrade},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;
use tracing::{info, warn};
use uuid::Uuid;

/// Log entry for dashboard display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLogEntry {
    pub id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: String,
    pub message: String,
    pub correlation_id: Option<Uuid>,
    pub algorithm: Option<String>,
    pub execution_time_ms: Option<f64>,
    pub confidence: Option<f64>,
    pub error: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Live metrics snapshot for dashboard
#[derive(Debug, Clone, Serialize)]
pub struct LiveMetricsSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub active_operations: Vec<ActiveOperation>,
    pub recent_performance: Vec<PerformanceMeasurement>,
    pub algorithm_stats: HashMap<String, AlgorithmSummary>,
    pub system_health: SystemHealth,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActiveOperation {
    pub correlation_id: Uuid,
    pub operation_type: String,
    pub algorithm: Option<String>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub current_stage: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AlgorithmSummary {
    pub name: String,
    pub executions_last_hour: usize,
    pub average_execution_time_ms: f64,
    pub success_rate: f64,
    pub current_load: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemHealth {
    pub log_entries_per_second: f64,
    pub active_correlations: usize,
    pub memory_usage_mb: f64,
    pub error_rate_last_hour: f64,
}

/// Query parameters for log filtering
#[derive(Debug, Deserialize)]
pub struct LogQuery {
    pub level: Option<String>,
    pub algorithm: Option<String>,
    pub correlation_id: Option<Uuid>,
    pub start_time: Option<chrono::DateTime<chrono::Utc>>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    pub limit: Option<usize>,
}

/// Dashboard log manager
pub struct DashboardLogManager {
    log_entries: Arc<Mutex<Vec<DashboardLogEntry>>>,
    metrics_collector: Arc<MetricsCollector>,
    active_operations: Arc<Mutex<HashMap<Uuid, ActiveOperation>>>,
    broadcast_sender: broadcast::Sender<DashboardLogEntry>,
    max_entries: usize,
}

impl DashboardLogManager {
    /// Create a new dashboard log manager
    pub fn new(metrics_collector: Arc<MetricsCollector>, max_entries: usize) -> Self {
        let (broadcast_sender, _) = broadcast::channel(1000);
        
        Self {
            log_entries: Arc::new(Mutex::new(Vec::new())),
            metrics_collector,
            active_operations: Arc::new(Mutex::new(HashMap::new())),
            broadcast_sender,
            max_entries,
        }
    }

    /// Add a log entry for dashboard display
    pub fn add_log_entry(&self, entry: DashboardLogEntry) {
        // Store the entry
        if let Ok(mut entries) = self.log_entries.lock() {
            entries.push(entry.clone());
            
            // Maintain size limit
            if entries.len() > self.max_entries {
                let excess = entries.len() - self.max_entries;
                entries.drain(0..excess);
            }
        }

        // Broadcast to live connections
        if let Err(e) = self.broadcast_sender.send(entry) {
            warn!("Failed to broadcast log entry: {}", e);
        }
    }

    /// Track an active operation
    pub fn start_operation(&self, correlation_id: Uuid, operation: ActiveOperation) {
        if let Ok(mut ops) = self.active_operations.lock() {
            ops.insert(correlation_id, operation);
        }
    }

    /// Update operation stage
    pub fn update_operation_stage(&self, correlation_id: Uuid, stage: &str) {
        if let Ok(mut ops) = self.active_operations.lock() {
            if let Some(op) = ops.get_mut(&correlation_id) {
                op.current_stage = stage.to_string();
            }
        }
    }

    /// Complete an operation
    pub fn complete_operation(&self, correlation_id: Uuid) {
        if let Ok(mut ops) = self.active_operations.lock() {
            ops.remove(&correlation_id);
        }
    }

    /// Get filtered log entries
    pub fn get_logs(&self, query: &LogQuery) -> Vec<DashboardLogEntry> {
        if let Ok(entries) = self.log_entries.lock() {
            let mut filtered: Vec<_> = entries.iter().cloned().collect();

            // Apply filters
            if let Some(ref level) = query.level {
                filtered.retain(|e| e.level.eq_ignore_ascii_case(level));
            }

            if let Some(ref algorithm) = query.algorithm {
                filtered.retain(|e| {
                    e.algorithm
                        .as_ref()
                        .map_or(false, |a| a.eq_ignore_ascii_case(algorithm))
                });
            }

            if let Some(correlation_id) = query.correlation_id {
                filtered.retain(|e| e.correlation_id == Some(correlation_id));
            }

            if let Some(start_time) = query.start_time {
                filtered.retain(|e| e.timestamp >= start_time);
            }

            if let Some(end_time) = query.end_time {
                filtered.retain(|e| e.timestamp <= end_time);
            }

            // Sort by timestamp (newest first)
            filtered.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

            // Apply limit
            if let Some(limit) = query.limit {
                filtered.truncate(limit);
            } else {
                filtered.truncate(1000); // Default limit
            }

            filtered
        } else {
            Vec::new()
        }
    }

    /// Generate live metrics snapshot
    pub fn get_live_metrics(&self) -> LiveMetricsSnapshot {
        let active_ops = if let Ok(ops) = self.active_operations.lock() {
            ops.values().cloned().collect()
        } else {
            Vec::new()
        };

        // Get recent performance data (last 5 minutes)
        let five_minutes_ago = chrono::Utc::now() - chrono::Duration::minutes(5);
        let recent_performance: Vec<_> = if let Ok(measurements) = self.metrics_collector.measurements.lock() {
            measurements
                .iter()
                .filter(|m| m.timestamp >= five_minutes_ago)
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        // Calculate algorithm summaries
        let mut algorithm_stats = HashMap::new();
        let algorithms = ["ORB", "SIFT", "AKAZE", "Template", "ECC"];
        
        for algorithm in &algorithms {
            if let Some(metrics) = self.metrics_collector.get_algorithm_metrics(algorithm) {
                let summary = AlgorithmSummary {
                    name: algorithm.to_string(),
                    executions_last_hour: metrics.total_executions,
                    average_execution_time_ms: metrics.mean_execution_time_ms,
                    success_rate: if metrics.total_executions > 0 {
                        (metrics.successful_executions as f64) / (metrics.total_executions as f64) * 100.0
                    } else {
                        0.0
                    },
                    current_load: active_ops
                        .iter()
                        .filter(|op| op.algorithm.as_deref() == Some(algorithm))
                        .count(),
                };
                algorithm_stats.insert(algorithm.to_string(), summary);
            }
        }

        // Calculate system health
        let error_entries = if let Ok(entries) = self.log_entries.lock() {
            let one_hour_ago = chrono::Utc::now() - chrono::Duration::hours(1);
            entries
                .iter()
                .filter(|e| e.timestamp >= one_hour_ago && e.level == "ERROR")
                .count()
        } else {
            0
        };

        let total_entries = if let Ok(entries) = self.log_entries.lock() {
            entries.len()
        } else {
            0
        };

        let system_health = SystemHealth {
            log_entries_per_second: recent_performance.len() as f64 / 300.0, // 5 minutes
            active_correlations: active_ops.len(),
            memory_usage_mb: 0.0, // TODO: Implement actual memory monitoring
            error_rate_last_hour: if total_entries > 0 {
                (error_entries as f64) / (total_entries as f64) * 100.0
            } else {
                0.0
            },
        };

        LiveMetricsSnapshot {
            timestamp: chrono::Utc::now(),
            active_operations: active_ops,
            recent_performance,
            algorithm_stats,
            system_health,
        }
    }

    /// Get broadcast receiver for live updates
    pub fn subscribe(&self) -> broadcast::Receiver<DashboardLogEntry> {
        self.broadcast_sender.subscribe()
    }
}

/// Create dashboard logging routes
pub fn create_dashboard_routes(log_manager: Arc<DashboardLogManager>) -> Router {
    Router::new()
        .route("/api/logs", get(get_logs_handler))
        .route("/api/logs/live", get(websocket_handler))
        .route("/api/metrics/live", get(get_live_metrics_handler))
        .route("/api/operations", get(get_active_operations_handler))
        .route("/api/operations/:correlation_id", post(update_operation_handler))
        .with_state(log_manager)
}

/// Handler for getting filtered logs
async fn get_logs_handler(
    State(log_manager): State<Arc<DashboardLogManager>>,
    Query(query): Query<LogQuery>,
) -> impl IntoResponse {
    let logs = log_manager.get_logs(&query);
    Json(logs)
}

/// WebSocket handler for live log streaming
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(log_manager): State<Arc<DashboardLogManager>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_websocket(socket, log_manager))
}

async fn handle_websocket(
    mut socket: axum::extract::ws::WebSocket,
    log_manager: Arc<DashboardLogManager>,
) {
    let mut receiver = log_manager.subscribe();

    while let Ok(entry) = receiver.recv().await {
        let message = match serde_json::to_string(&entry) {
            Ok(json) => axum::extract::ws::Message::Text(json),
            Err(e) => {
                warn!("Failed to serialize log entry: {}", e);
                continue;
            }
        };

        if let Err(e) = socket.send(message).await {
            warn!("WebSocket send failed: {}", e);
            break;
        }
    }
}

/// Handler for live metrics
async fn get_live_metrics_handler(
    State(log_manager): State<Arc<DashboardLogManager>>,
) -> impl IntoResponse {
    let metrics = log_manager.get_live_metrics();
    Json(metrics)
}

/// Handler for active operations
async fn get_active_operations_handler(
    State(log_manager): State<Arc<DashboardLogManager>>,
) -> impl IntoResponse {
    match log_manager.active_operations.lock() {
        Ok(ops) => {
            let operations: Vec<_> = ops.values().cloned().collect();
            Json(operations).into_response()
        }
        Err(_) => {
            (StatusCode::INTERNAL_SERVER_ERROR, "Failed to get operations").into_response()
        }
    }
}

/// Handler for updating operation status
async fn update_operation_handler() -> impl IntoResponse {
    // TODO: Implement operation update logic
    StatusCode::OK
}

/// Convert tracing events to dashboard log entries
pub fn convert_to_dashboard_entry(
    level: &str,
    message: &str,
    correlation_id: Option<Uuid>,
    fields: HashMap<String, serde_json::Value>,
) -> DashboardLogEntry {
    let algorithm = fields
        .get("algorithm")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let execution_time_ms = fields
        .get("execution_time_ms")
        .and_then(|v| v.as_f64());

    let confidence = fields
        .get("confidence")
        .and_then(|v| v.as_f64());

    let error = if level == "ERROR" {
        Some(message.to_string())
    } else {
        fields
            .get("error")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    };

    DashboardLogEntry {
        id: Uuid::new_v4(),
        timestamp: chrono::Utc::now(),
        level: level.to_string(),
        message: message.to_string(),
        correlation_id,
        algorithm,
        execution_time_ms,
        confidence,
        error,
        metadata: fields,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_dashboard_log_manager() {
        let metrics = Arc::new(MetricsCollector::new(true));
        let manager = DashboardLogManager::new(metrics, 1000);

        let entry = DashboardLogEntry {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            level: "INFO".to_string(),
            message: "Test message".to_string(),
            correlation_id: None,
            algorithm: Some("ORB".to_string()),
            execution_time_ms: Some(25.0),
            confidence: Some(0.85),
            error: None,
            metadata: HashMap::new(),
        };

        manager.add_log_entry(entry.clone());

        let query = LogQuery {
            level: Some("INFO".to_string()),
            algorithm: None,
            correlation_id: None,
            start_time: None,
            end_time: None,
            limit: Some(10),
        };

        let logs = manager.get_logs(&query);
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].message, "Test message");
    }

    #[test]
    fn test_convert_to_dashboard_entry() {
        let mut fields = HashMap::new();
        fields.insert("algorithm".to_string(), serde_json::json!("ORB"));
        fields.insert("execution_time_ms".to_string(), serde_json::json!(23.5));

        let entry = convert_to_dashboard_entry(
            "INFO",
            "Algorithm completed",
            Some(Uuid::new_v4()),
            fields,
        );

        assert_eq!(entry.level, "INFO");
        assert_eq!(entry.message, "Algorithm completed");
        assert_eq!(entry.algorithm, Some("ORB".to_string()));
        assert_eq!(entry.execution_time_ms, Some(23.5));
    }
}