use axum::{
    extract::{Multipart, Path, Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, get_service, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, services::ServeDir};

use crate::config::{Config, DashboardConfig};
use super::data::{DashboardData, DashboardDataLoader, TestSession};

/// Test progress tracking
#[derive(Debug, Clone, Serialize)]
pub struct TestProgress {
    pub status: String, // "running", "completed", "failed"
    pub percentage: u32,
    pub message: String,
    pub error: Option<String>,
}

/// Test execution response
#[derive(Debug, Serialize)]
pub struct TestStartResponse {
    pub status: String,
    pub test_id: String,
}

/// Dashboard server state
#[derive(Clone)]
pub struct DashboardState {
    pub data_loader: DashboardDataLoader,
    pub dashboard_data: Arc<tokio::sync::RwLock<DashboardData>>,
    pub test_progress: Arc<RwLock<HashMap<String, TestProgress>>>,
    pub config: Config,
}

impl DashboardState {
    pub fn new(results_dir: PathBuf) -> anyhow::Result<Self> {
        Self::with_config(results_dir, Config::default())
    }

    pub fn with_config(results_dir: PathBuf, config: Config) -> anyhow::Result<Self> {
        let data_loader = DashboardDataLoader::new(results_dir);
        let dashboard_data = data_loader.load_dashboard_data()?;

        Ok(Self {
            data_loader,
            dashboard_data: Arc::new(tokio::sync::RwLock::new(dashboard_data)),
            test_progress: Arc::new(RwLock::new(HashMap::new())),
            config,
        })
    }

    pub async fn refresh_data(&self) -> anyhow::Result<()> {
        let new_data = self.data_loader.load_dashboard_data()?;
        let mut data = self.dashboard_data.write().await;
        *data = new_data;
        Ok(())
    }
}

/// API query parameters for filtering
#[derive(Debug, Deserialize)]
pub struct FilterParams {
    pub algorithm: Option<String>,
    pub patch_size: Option<String>,
    pub transformation: Option<String>,
    pub success_only: Option<bool>,
}

/// Dashboard server
pub struct DashboardServer {
    state: DashboardState,
    port: u16,
    config: DashboardConfig,
}

impl DashboardServer {
    pub fn new(results_dir: PathBuf, port: u16) -> anyhow::Result<Self> {
        let config = Config::default();
        let state = DashboardState::with_config(results_dir, config.clone())?;
        Ok(Self { 
            state, 
            port: if port == 0 { config.dashboard.default_port } else { port },
            config: config.dashboard,
        })
    }

    pub fn with_config(results_dir: PathBuf, config: Config) -> anyhow::Result<Self> {
        let state = DashboardState::with_config(results_dir, config.clone())?;
        Ok(Self { 
            state, 
            port: config.dashboard.default_port,
            config: config.dashboard,
        })
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let static_dir = std::env::current_dir()?.join("src/dashboard/frontend");

        let app = Router::new()
            // API routes
            .route("/api/data", get(get_dashboard_data))
            .route("/api/sessions", get(get_test_sessions))
            .route("/api/sessions/:id", get(get_test_session))
            .route("/api/algorithms/summary", get(get_algorithm_summary))
            .route("/api/refresh", get(refresh_dashboard_data))
            // Test runner routes
            .route("/api/run-visual-test", post(run_visual_test))
            .route("/api/test-progress/:id", get(get_test_progress))
            // Static file serving for images
            .route("/api/image/*path", get(serve_image))
            // Frontend routes
            .route("/", get(serve_index))
            .route("/dashboard", get(serve_index))
            .route("/session/:id", get(serve_index))
            // Serve static frontend files
            .nest_service("/static", get_service(ServeDir::new(&static_dir)))
            .layer(ServiceBuilder::new().layer(
                if self.config.enable_cors {
                    CorsLayer::permissive()
                } else {
                    CorsLayer::new()
                }
            ))
            .with_state(self.state);

        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", self.port)).await?;
        println!(
            "🚀 Dashboard server running on http://localhost:{}",
            self.port
        );
        println!("📊 Open your browser to view the SEM Image Alignment Dashboard");

        axum::serve(listener, app).await?;
        Ok(())
    }
}

/// API handlers
async fn get_dashboard_data(
    State(state): State<DashboardState>,
    Query(params): Query<FilterParams>,
) -> Result<Json<DashboardData>, StatusCode> {
    let data = state.dashboard_data.read().await;
    let filtered_data = filter_dashboard_data(&data, &params);
    Ok(Json(filtered_data))
}

async fn get_test_sessions(
    State(state): State<DashboardState>,
) -> Result<Json<Vec<TestSession>>, StatusCode> {
    let data = state.dashboard_data.read().await;
    Ok(Json(data.test_sessions.clone()))
}

async fn get_test_session(
    State(state): State<DashboardState>,
    Path(session_id): Path<String>,
) -> Result<Json<TestSession>, StatusCode> {
    let data = state.dashboard_data.read().await;

    if let Some(session) = data.test_sessions.iter().find(|s| s.id == session_id) {
        Ok(Json(session.clone()))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn get_algorithm_summary(
    State(state): State<DashboardState>,
) -> Result<Json<Vec<super::data::AlgorithmSummary>>, StatusCode> {
    let data = state.dashboard_data.read().await;
    let summaries = state.data_loader.calculate_algorithm_summaries(&data);
    Ok(Json(summaries))
}

async fn refresh_dashboard_data(
    State(state): State<DashboardState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.refresh_data().await {
        Ok(()) => Ok(Json(serde_json::json!({"status": "success"}))),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn serve_image(
    State(state): State<DashboardState>,
    Path(image_path): Path<String>,
) -> Result<Response, StatusCode> {
    // Security: ensure path doesn't escape results directory
    let safe_path = image_path.replace("..", "");

    // Handle both absolute paths and relative paths
    let full_path = if safe_path.starts_with("results/") {
        // Path is already relative to project root, use as-is
        std::env::current_dir().unwrap_or_default().join(&safe_path)
    } else {
        // Path is relative to results directory
        state.data_loader.results_dir.join(&safe_path)
    };

    // Security check: ensure path is within allowed directories
    let current_dir = std::env::current_dir().unwrap_or_default();
    let results_dir = current_dir.join("results");
    let datasets_dir = current_dir.join("datasets");

    let path_is_safe = full_path.starts_with(&results_dir)
        || full_path.starts_with(&datasets_dir)
        || full_path.starts_with(&state.data_loader.results_dir);

    if !full_path.exists() || !path_is_safe {
        eprintln!("Image not found or unsafe path: {}", full_path.display());
        return Err(StatusCode::NOT_FOUND);
    }

    match tokio::fs::read(&full_path).await {
        Ok(contents) => {
            let mime_type = mime_guess::from_path(&full_path)
                .first_or_octet_stream()
                .to_string();

            Ok(([(header::CONTENT_TYPE, mime_type)], contents).into_response())
        }
        Err(e) => {
            eprintln!("Failed to read image file {}: {}", full_path.display(), e);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

async fn serve_index() -> Html<&'static str> {
    Html(include_str!("frontend/index.html"))
}

/// Filter dashboard data based on query parameters
fn filter_dashboard_data(data: &DashboardData, params: &FilterParams) -> DashboardData {
    let mut filtered_data = data.clone();

    for session in &mut filtered_data.test_sessions {
        session.test_results.retain(|test| {
            // Filter by algorithm
            if let Some(ref algorithm) = params.algorithm {
                if test.algorithm_name != *algorithm {
                    return false;
                }
            }

            // Filter by patch size
            if let Some(ref patch_size) = params.patch_size {
                let test_patch_size =
                    format!("{}x{}", test.patch_info.size.0, test.patch_info.size.1);
                if test_patch_size != *patch_size {
                    return false;
                }
            }

            // Filter by transformation
            if let Some(ref transformation) = params.transformation {
                if test.transformation_applied.noise_parameters != *transformation {
                    return false;
                }
            }

            // Filter by success only
            if let Some(success_only) = params.success_only {
                if success_only && !test.performance_metrics.success {
                    return false;
                }
            }

            true
        });

        // Recalculate session statistics after filtering
        let total_tests = session.test_results.len();
        let success_count = session
            .test_results
            .iter()
            .filter(|t| t.performance_metrics.success)
            .count();
        session.success_rate = if total_tests > 0 {
            success_count as f32 / total_tests as f32 * 100.0
        } else {
            0.0
        };
        session.avg_processing_time = if total_tests > 0 {
            session
                .test_results
                .iter()
                .map(|t| t.performance_metrics.processing_time_ms)
                .sum::<f32>()
                / total_tests as f32
        } else {
            0.0
        };
        session.total_tests = total_tests;
    }

    filtered_data
}

/// Handler for running visual tests
async fn run_visual_test(
    State(state): State<DashboardState>,
    mut multipart: Multipart,
) -> Result<Json<TestStartResponse>, StatusCode> {
    let mut sem_image_data: Option<Vec<u8>> = None;
    let mut patch_sizes: Option<String> = None;
    let mut scenarios: Option<String> = None;

    // Process multipart form data
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?
    {
        let name = field.name().unwrap_or_default();

        match name {
            "sem_image" => {
                sem_image_data = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|_| StatusCode::BAD_REQUEST)?
                        .to_vec(),
                );
            }
            "patch_sizes" => {
                patch_sizes = Some(field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?);
            }
            "scenarios" => {
                scenarios = Some(field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?);
            }
            _ => {}
        }
    }

    let sem_image_data = sem_image_data.ok_or(StatusCode::BAD_REQUEST)?;
    let patch_sizes = patch_sizes.ok_or(StatusCode::BAD_REQUEST)?;
    let scenarios = scenarios.ok_or(StatusCode::BAD_REQUEST)?;

    // Generate unique test ID
    let test_id = format!("test_{}", chrono::Utc::now().timestamp());

    // Initialize progress tracking
    {
        let mut progress_map = state.test_progress.write().await;
        progress_map.insert(
            test_id.clone(),
            TestProgress {
                status: "running".to_string(),
                percentage: 0,
                message: "Starting test...".to_string(),
                error: None,
            },
        );
    }

    // Spawn background task to run the test
    let state_clone = state.clone();
    let test_id_clone = test_id.clone();
    tokio::spawn(async move {
        if let Err(e) = run_visual_test_background(
            state_clone,
            test_id_clone,
            sem_image_data,
            patch_sizes,
            scenarios,
        )
        .await
        {
            eprintln!("Background test failed: {}", e);
        }
    });

    Ok(Json(TestStartResponse {
        status: "started".to_string(),
        test_id,
    }))
}

/// Handler for getting test progress
async fn get_test_progress(
    State(state): State<DashboardState>,
    Path(test_id): Path<String>,
) -> Result<Json<TestProgress>, StatusCode> {
    let progress_map = state.test_progress.read().await;

    if let Some(progress) = progress_map.get(&test_id) {
        Ok(Json(progress.clone()))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

/// Background task to run visual test
async fn run_visual_test_background(
    state: DashboardState,
    test_id: String,
    sem_image_data: Vec<u8>,
    patch_sizes: String,
    scenarios: String,
) -> anyhow::Result<()> {
    // Update progress: Processing image
    update_test_progress(&state, &test_id, 10, "Processing image...").await;

    // Create a temporary directory for this test
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let output_dir = std::env::current_dir()?
        .join("results")
        .join(format!("test_{}", timestamp));
    std::fs::create_dir_all(&output_dir)?;

    // Save the uploaded image
    let image_path = output_dir.join("input_image.jpg");
    tokio::fs::write(&image_path, &sem_image_data).await?;

    update_test_progress(&state, &test_id, 20, "Preparing visual test command...").await;

    // Get the current executable path
    let current_exe = std::env::current_exe()?;

    // Build the command arguments
    let mut cmd = tokio::process::Command::new(&current_exe);
    cmd.arg("visual-test")
        .arg("--sem-image")
        .arg(&image_path)
        .arg("--output")
        .arg(&output_dir)
        .arg("--patch-sizes")
        .arg(&patch_sizes)
        .arg("--scenarios")
        .arg(&scenarios);

    update_test_progress(&state, &test_id, 30, "Running visual test...").await;

    // Execute the command
    let output = cmd.output().await?;

    if !output.status.success() {
        let error_msg = String::from_utf8_lossy(&output.stderr);
        eprintln!("Visual test command failed: {}", error_msg);

        // Mark as failed
        let mut progress_map = state.test_progress.write().await;
        progress_map.insert(
            test_id,
            TestProgress {
                status: "failed".to_string(),
                percentage: 0,
                message: "Test failed".to_string(),
                error: Some(error_msg.to_string()),
            },
        );
        return Err(anyhow::anyhow!("Visual test command failed: {}", error_msg));
    }

    update_test_progress(&state, &test_id, 90, "Finalizing results...").await;

    // Refresh dashboard data to include new results
    state.refresh_data().await?;

    // Mark as completed
    {
        let mut progress_map = state.test_progress.write().await;
        progress_map.insert(
            test_id,
            TestProgress {
                status: "completed".to_string(),
                percentage: 100,
                message: "Test completed successfully!".to_string(),
                error: None,
            },
        );
    }

    Ok(())
}

/// Helper function to update test progress
async fn update_test_progress(
    state: &DashboardState,
    test_id: &str,
    percentage: u32,
    message: &str,
) {
    let mut progress_map = state.test_progress.write().await;
    progress_map.insert(
        test_id.to_string(),
        TestProgress {
            status: "running".to_string(),
            percentage,
            message: message.to_string(),
            error: None,
        },
    );
}

/// Launch dashboard server
pub async fn start_dashboard_server(results_dir: PathBuf, port: u16) -> anyhow::Result<()> {
    let server = DashboardServer::new(results_dir, port)?;
    server.run().await
}

/// Launch dashboard server with config
pub async fn start_dashboard_server_with_config(results_dir: PathBuf, config: Config) -> anyhow::Result<()> {
    let server = DashboardServer::with_config(results_dir, config)?;
    server.run().await
}
