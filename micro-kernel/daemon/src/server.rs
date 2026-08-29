//! The local HTTP and WebSocket server.
//!
//! Everything here binds to loopback. The pairing token exists so that a
//! page served from elsewhere can drive compute on this machine without
//! anything else on the machine being able to; it is not a substitute for
//! the loopback bind, which is the actual boundary.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex};
use tower_http::cors::{Any, CorsLayer};

use crate::graph::{Delta, Kind, Report, Runtime, Value};
use crate::integrations::ollama::{self, Ollama};
use crate::lang::{self, mishima, sangoma, CheckResult};
use crate::studio::{ExportEvent, Studio};

pub struct AppState {
    pub token: String,
    pub runtime: Mutex<Runtime>,
    pub studio: Mutex<Studio>,
    pub ollama: Ollama,
    pub workspace: PathBuf,
    pub tx: broadcast::Sender<ServerMessage>,
    /// Numeric resolution of the analysis path, for floor negotiation.
    pub backend_resolution: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    Hello { version: String, record: u64 },
    Delta { deltas: Vec<Delta> },
    Export { event: Box<ExportEvent> },
    Status { status: StudioStatus },
    Error { message: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioStatus {
    pub fl_watching: Option<String>,
    pub fl_exports_seen: usize,
    pub ollama_endpoint: String,
    pub ollama_model: String,
    pub ollama_models: Vec<String>,
    pub ollama_reachable: bool,
}

pub fn router(state: Arc<AppState>) -> Router {
    // The browser may be served from any origin; the token is what
    // authorises, and the bind address is what confines.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/api/status", get(status))
        .route("/api/check", post(check))
        .route("/api/run", post(run))
        .route("/api/graph", get(graph))
        .route("/api/files", get(list_files))
        .route("/api/file", get(read_file).post(write_file))
        .route("/api/exports", get(exports))
        .route("/api/compose", post(compose))
        .route("/api/accountability", post(accountability))
        .route("/ws", get(ws_upgrade))
        .layer(cors)
        .with_state(state)
}

// ── authorisation ──────────────────────────────────────────────────────

fn authorised(headers: &HeaderMap, token: &str) -> bool {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|v| constant_time_eq(v.trim(), token))
        .unwrap_or(false)
}

/// Compare without leaking the position of the first difference.
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn unauthorised() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({
            "error": "token missing or rejected",
            "remedy": "run `heihachi serve` and paste the token it prints"
        })),
    )
}

// ── handlers ───────────────────────────────────────────────────────────

async fn status(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }
    Json(current_status(&state).await).into_response()
}

pub async fn current_status(state: &AppState) -> StudioStatus {
    let (models, reachable) = match state.ollama.models().await {
        Ok(m) => (m.into_iter().map(|m| m.name).collect(), true),
        Err(_) => (Vec::new(), false),
    };
    let studio = state.studio.lock().await;
    StudioStatus {
        fl_watching: studio.watching().map(|p| p.display().to_string()),
        fl_exports_seen: studio.exports().len(),
        ollama_endpoint: ollama::DEFAULT_ENDPOINT.to_string(),
        ollama_model: state.ollama.model().to_string(),
        ollama_models: models,
        ollama_reachable: reachable,
    }
}

#[derive(Deserialize)]
struct SourceRequest {
    language: String,
    source: String,
}

async fn check(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<SourceRequest>,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }
    Json(check_source(&body.language, &body.source, state.backend_resolution))
        .into_response()
}

fn check_source(language: &str, source: &str, backend: f64) -> CheckResult {
    match language {
        "sangoma" => match sangoma::parse(source) {
            Ok(prog) => sangoma::check(&prog, 0.8, backend),
            Err(d) => CheckResult::from_diagnostics(vec![d], Vec::new()),
        },
        _ => match mishima::parse(source) {
            Ok(prog) => mishima::check(&prog, backend),
            Err(d) => CheckResult::from_diagnostics(vec![d], Vec::new()),
        },
    }
}

#[derive(Serialize)]
struct RunResponse {
    report: Report,
    emissions: Vec<Value>,
    check: CheckResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    decline: Option<Decline>,
}

#[derive(Serialize)]
struct Decline {
    classes: Vec<String>,
    probes_invoked: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    discriminating_probe: Option<String>,
}

async fn run(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<SourceRequest>,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }

    let checked = check_source(&body.language, &body.source, state.backend_resolution);
    if !checked.accepted {
        // A refused program does not run. The diagnostics are the result.
        let rt = state.runtime.lock().await;
        return Json(RunResponse {
            report: rt.report(),
            emissions: Vec::new(),
            check: checked,
            decline: None,
        })
        .into_response();
    }

    let mut rt = state.runtime.lock().await;
    let before = rt.report().emissions;
    let mut decline = None;

    if body.language == "sangoma" {
        if let Ok(prog) = sangoma::parse(&body.source) {
            for c in &prog.constructs {
                let tau = format!("construct.{}", c.name);
                rt.attach_chunk(&tau, "sangoma");
                let power =
                    lang::composite_power(c.ladder.iter().map(|r| r.power));
                let floor = prog.ambient_floor.unwrap_or(0.02);
                if let Ok(v) = Value::reading(
                    format!("{tau}.composite_power"),
                    power,
                    floor,
                    "",
                    "sangoma",
                ) {
                    rt.emit(&tau, v, Some(&tau));
                }
                for t in &c.targets {
                    if let Ok(v) = Value::reading(
                        format!("{tau}.target.{}", t.name),
                        t.magnitude,
                        t.floor,
                        "",
                        "sangoma",
                    ) {
                        rt.emit(&tau, v, Some(&tau));
                    }
                }
            }
        }
    } else if let Ok(prog) = mishima::parse(&body.source) {
        for s in &prog.seeks {
            let tau = format!("seek.{}", s.subject);
            rt.attach_chunk(&tau, "mishima");
            let floor = prog.ambient_floor.unwrap_or(0.02);
            let power = lang::composite_power(s.ladder.iter().map(|r| r.power));
            if let Ok(v) =
                Value::reading(format!("{tau}.composite_power"), power, floor, "", "mishima")
            {
                rt.emit(&tau, v, Some(&tau));
            }

            // Each rung reaches a class. A seek whose rungs reach more than
            // one irreconcilable class terminates contested, and that is a
            // legal outcome carrying what it found -- not an error.
            let classes = reached_classes(&rt, s);
            if classes.len() > 1 {
                if let Ok(v) = Value::new(
                    format!("{tau}.decline"),
                    serde_json::json!(classes.clone()),
                    floor,
                    "",
                    Kind::Decline,
                    "mishima",
                ) {
                    rt.emit(&tau, v, Some(&tau));
                }
                decline = Some(Decline {
                    classes,
                    probes_invoked: s.ladder.iter().map(|r| r.name.clone()).collect(),
                    discriminating_probe: s
                        .ladder
                        .iter()
                        .max_by(|a, b| {
                            a.power.partial_cmp(&b.power).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|r| r.name.clone()),
                });
            }
        }
    }

    let emissions: Vec<Value> = rt
        .nodes()
        .flat_map(|n| n.values.iter())
        .filter(|v| v.record as usize > before)
        .cloned()
        .collect();
    let deltas = rt.drain_deltas();
    let report = rt.report();
    drop(rt);

    if !deltas.is_empty() {
        let _ = state.tx.send(ServerMessage::Delta { deltas });
    }

    Json(RunResponse { report, emissions, check: checked, decline }).into_response()
}

/// Which classes a seek's rungs reach, read off the graph.
///
/// A rung whose name matches a node's subtask reaches that node's class;
/// rungs that reach nothing contribute nothing rather than being counted as
/// agreement.
fn reached_classes(rt: &Runtime, seek: &mishima::Seek) -> Vec<String> {
    let mut classes: Vec<String> = Vec::new();
    for rung in &seek.ladder {
        let hit = rt
            .nodes()
            .find(|n| n.tau.contains(&rung.name))
            .map(|n| n.tau.clone());
        if let Some(class) = hit {
            if !classes.contains(&class) {
                classes.push(class);
            }
        }
    }
    classes
}

async fn graph(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }
    let rt = state.runtime.lock().await;
    Json(serde_json::json!({
        "nodes": rt.nodes().cloned().collect::<Vec<_>>(),
        "edges": rt.edges().cloned().collect::<Vec<_>>(),
        "report": rt.report(),
        "trajectory": rt.trajectory(),
    }))
    .into_response()
}

async fn exports(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }
    let studio = state.studio.lock().await;
    Json(studio.exports().to_vec()).into_response()
}

#[derive(Serialize)]
struct FileEntry {
    name: String,
    path: String,
    language: String,
}

async fn list_files(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }
    let mut out = Vec::new();
    for sub in ["mishima", "sangoma"] {
        let dir = state.workspace.join(sub);
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut names: Vec<_> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file())
                .collect();
            names.sort();
            for path in names {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or_default()
                    .to_string();
                let language = match ext.as_str() {
                    "mma" => "mishima",
                    "sgn" => "sangoma",
                    _ => "other",
                };
                out.push(FileEntry {
                    name: path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or_default()
                        .to_string(),
                    path: format!(
                        "{sub}/{}",
                        path.file_name().and_then(|n| n.to_str()).unwrap_or_default()
                    ),
                    language: language.to_string(),
                });
            }
        }
    }
    Json(out).into_response()
}

#[derive(Deserialize)]
struct PathQuery {
    path: String,
}

/// Resolve a workspace-relative path, refusing anything that escapes it.
fn resolve(workspace: &Path, relative: &str) -> Option<PathBuf> {
    let candidate = workspace.join(relative);
    let allowed: HashSet<PathBuf> = ["mishima", "sangoma"]
        .iter()
        .map(|s| workspace.join(s))
        .collect();
    let parent = candidate.parent()?.to_path_buf();
    if !allowed.contains(&parent) {
        return None;
    }
    if relative.contains("..") {
        return None;
    }
    Some(candidate)
}

async fn read_file(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<PathQuery>,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }
    let Some(path) = resolve(&state.workspace, &q.path) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "path outside the workspace" })),
        )
            .into_response();
    };
    match std::fs::read_to_string(&path) {
        Ok(source) => Json(serde_json::json!({ "path": q.path, "source": source }))
            .into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct WriteRequest {
    path: String,
    source: String,
}

async fn write_file(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<WriteRequest>,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }
    let Some(path) = resolve(&state.workspace, &body.path) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "path outside the workspace" })),
        )
            .into_response();
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&path, body.source) {
        Ok(()) => Json(serde_json::json!({ "ok": true })).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct ComposeRequest {
    language: String,
    request: String,
}

/// Ask the model to write source. It writes; it never answers.
///
/// The result is returned for review, never run: a program the producer has
/// not read is a program nobody authored.
async fn compose(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ComposeRequest>,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }
    let source = crate::compose::compose(&state.ollama, &body.language, &body.request)
        .await;
    Json(source).into_response()
}

#[derive(Deserialize)]
struct AccountabilityRequest {
    outcome: String,
}

async fn accountability(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<AccountabilityRequest>,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }
    let rt = state.runtime.lock().await;
    Json(crate::studio::accountability(&rt, &body.outcome)).into_response()
}

// ── websocket ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct WsQuery {
    token: String,
}

async fn ws_upgrade(
    State(state): State<Arc<AppState>>,
    Query(q): Query<WsQuery>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    if !constant_time_eq(q.token.trim(), &state.token) {
        return unauthorised().into_response();
    }
    ws.on_upgrade(move |socket| ws_loop(socket, state))
}

async fn ws_loop(mut socket: WebSocket, state: Arc<AppState>) {
    let mut rx = state.tx.subscribe();

    let record = state.runtime.lock().await.record();
    let hello = ServerMessage::Hello {
        version: env!("CARGO_PKG_VERSION").to_string(),
        record,
    };
    if send(&mut socket, &hello).await.is_err() {
        return;
    }
    let status = ServerMessage::Status { status: current_status(&state).await };
    if send(&mut socket, &status).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    _ => {}
                }
            }
            broadcast = rx.recv() => {
                match broadcast {
                    Ok(message) => {
                        if send(&mut socket, &message).await.is_err() {
                            break;
                        }
                    }
                    // A slow client that fell behind is told so rather than
                    // silently shown a gap in the record.
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        let warn = ServerMessage::Error {
                            message: format!(
                                "this view fell {n} messages behind and reloaded"
                            ),
                        };
                        if send(&mut socket, &warn).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }
}

async fn send(socket: &mut WebSocket, message: &ServerMessage) -> Result<(), ()> {
    let text = serde_json::to_string(message).map_err(|_| ())?;
    socket.send(Message::Text(text)).await.map_err(|_| ())
}

pub async fn bind(
    state: Arc<AppState>,
    addr: SocketAddr,
) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router(state)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_comparison_rejects_wrong_and_short_tokens() {
        assert!(constant_time_eq("abc", "abc"));
        assert!(!constant_time_eq("abc", "abd"));
        assert!(!constant_time_eq("abc", "ab"));
        assert!(!constant_time_eq("", "a"));
    }

    #[test]
    fn workspace_paths_that_escape_are_refused() {
        let ws = PathBuf::from("/workspace");
        assert!(resolve(&ws, "mishima/recall.mma").is_some());
        assert!(resolve(&ws, "sangoma/reese.sgn").is_some());
        assert!(resolve(&ws, "../secrets.txt").is_none());
        assert!(resolve(&ws, "mishima/../../etc/passwd").is_none());
        assert!(resolve(&ws, "elsewhere/file.mma").is_none());
    }

    #[test]
    fn a_refused_program_reports_diagnostics_and_does_not_run() {
        let result = check_source("mishima", "floor 0.02\nseek x toward { y }", 0.001);
        assert!(!result.accepted);
        assert!(result.diagnostics.iter().all(|d| !d.remedy.is_empty()));
    }
}
