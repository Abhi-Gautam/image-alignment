use axum::{
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, get_service},
    Json, Router,
};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, services::ServeDir};

use super::data::{DashboardData, DashboardDataLoader, TestSession};

/// Dashboard server state
#[derive(Clone)]
pub struct DashboardState {
    pub data_loader: DashboardDataLoader,
    pub dashboard_data: Arc<tokio::sync::RwLock<DashboardData>>,
}

impl DashboardState {
    pub fn new(results_dir: PathBuf) -> anyhow::Result<Self> {
        let data_loader = DashboardDataLoader::new(results_dir);
        let dashboard_data = data_loader.load_dashboard_data()?;

        Ok(Self {
            data_loader,
            dashboard_data: Arc::new(tokio::sync::RwLock::new(dashboard_data)),
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
}

impl DashboardServer {
    pub fn new(results_dir: PathBuf, port: u16) -> anyhow::Result<Self> {
        let state = DashboardState::new(results_dir)?;
        Ok(Self { state, port })
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
            // Static file serving for images
            .route("/api/image/*path", get(serve_image))
            // Frontend routes
            .route("/", get(serve_index))
            .route("/dashboard", get(serve_index))
            .route("/session/:id", get(serve_index))
            // Serve static frontend files
            .nest_service("/static", get_service(ServeDir::new(&static_dir)))
            .layer(ServiceBuilder::new().layer(CorsLayer::permissive()))
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

/// Launch dashboard server
pub async fn start_dashboard_server(results_dir: PathBuf, port: u16) -> anyhow::Result<()> {
    let server = DashboardServer::new(results_dir, port)?;
    server.run().await
}
