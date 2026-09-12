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

use crate::graph::{self, Delta, Kind, Report, Runtime, Value};
use crate::integrations::clap_host;
use crate::integrations::fl_control::{Command, FlLink};
use crate::integrations::ollama::{self, Ollama};
use crate::lang::{self, mishima, sangoma, CheckResult};
use crate::studio::{self, ExportEvent, Studio};

pub struct AppState {
    pub token: String,
    pub runtime: Mutex<Runtime>,
    pub studio: Mutex<Studio>,
    pub ollama: Ollama,
    pub workspace: PathBuf,
    pub tx: broadcast::Sender<ServerMessage>,
    /// Numeric resolution of the analysis path, for floor negotiation.
    pub backend_resolution: f64,
    /// The socket an FL MIDI Controller Script dials into, for the narrow
    /// transport/mixer/pattern acts FL's scripting API exposes.
    pub fl_link: Arc<FlLink>,
    /// Directories scanned for `.clap` bundles.
    pub plugin_dirs: Vec<PathBuf>,
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
        .route("/api/act", post(act))
        .route("/api/fl_link/status", get(fl_link_status))
        .route("/api/plugins", get(plugins))
        .route("/api/render", post(render_construct))
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

            // Each rung reaches a class, determined by walking the ladder to
            // closure over the live graph (`graph::reach::seek_to_closure`).
            // A seek whose rungs reach more than one irreconcilable class
            // terminates contested, and that is a legal outcome carrying
            // what it found -- not an error.
            let det = graph::reach::seek_to_closure(&rt, s, 0.99);
            if det.status == graph::reach::DeterminationStatus::Declined {
                if let Ok(v) = Value::new(
                    format!("{tau}.decline"),
                    serde_json::json!(det.classes.clone()),
                    floor,
                    "",
                    Kind::Decline,
                    "mishima",
                ) {
                    rt.emit(&tau, v, Some(&tau));
                }
                decline = Some(Decline {
                    classes: det.classes,
                    probes_invoked: det.probes_invoked,
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
    // Persistence failure must not turn a successful run into a failed
    // response -- log and continue, matching the "never fails" discipline
    // used throughout the analysis path.
    if let Err(e) = crate::persist::save(&state.workspace, &rt.snapshot()) {
        tracing::warn!("could not persist runtime: {e}");
    }
    drop(rt);

    if !deltas.is_empty() {
        let _ = state.tx.send(ServerMessage::Delta { deltas });
    }

    Json(RunResponse { report, emissions, check: checked, decline }).into_response()
}

#[derive(Deserialize)]
struct ActRequest {
    command: Command,
}

#[derive(Serialize)]
struct ActResponse {
    dispatched: bool,
}

/// Dispatch a transport/mixer/pattern command to the connected FL script.
///
/// This does not wait for FL's state to actually change: it attaches a
/// chunk, emits a value recording what was requested, and returns. A
/// dispatch failure (nothing connected) is recorded as an anomaly value,
/// not an HTTP error -- the runtime holds no expectation to compare against,
/// so an unreachable script is a fact about the run, not a failure of it.
async fn act(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ActRequest>,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }

    let tau = format!("act.{}", body.command.kind_name());
    let dispatch = state.fl_link.send(&body.command).await;

    let deltas = {
        let mut rt = state.runtime.lock().await;
        rt.attach_chunk(&tau, "fl_control");
        let value = match &dispatch {
            Ok(()) => Value::new(
                format!("{tau}.dispatched"),
                serde_json::json!(body.command),
                1.0,
                "",
                Kind::Reading,
                "fl_control",
            ),
            Err(e) => Value::new(
                format!("{tau}.anomaly"),
                serde_json::json!(e.to_string()),
                f64::MIN_POSITIVE,
                "",
                Kind::Anomaly,
                "fl_control",
            ),
        };
        if let Ok(v) = value {
            rt.emit(&tau, v, Some(&tau));
        }
        let deltas = rt.drain_deltas();
        if let Err(e) = crate::persist::save(&state.workspace, &rt.snapshot()) {
            tracing::warn!("could not persist runtime: {e}");
        }
        deltas
    };

    if !deltas.is_empty() {
        let _ = state.tx.send(ServerMessage::Delta { deltas });
    }

    Json(ActResponse { dispatched: dispatch.is_ok() }).into_response()
}

async fn fl_link_status(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }
    Json(serde_json::json!({ "connected": state.fl_link.is_connected().await })).into_response()
}

async fn plugins(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }
    let dirs = state.plugin_dirs.clone();
    let found = tokio::task::spawn_blocking(move || clap_host::scan(&dirs))
        .await
        .unwrap_or_default();
    Json(found).into_response()
}

#[derive(Deserialize)]
struct RenderRequest {
    source: String,
    construct: String,
    #[serde(default = "default_sample_rate")]
    sample_rate: u32,
    #[serde(default = "default_frames")]
    frames: usize,
}

fn default_sample_rate() -> u32 {
    48_000
}

fn default_frames() -> usize {
    48_000 * 2 // 2 seconds
}

#[derive(Serialize)]
struct RenderResponse {
    rendered: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Render one `sangoma` construct through its bound CLAP plugin chain and
/// commit the measured result. A construct with no `clap_id` bound on any
/// stage, or a stage whose plugin cannot be found, does not fail the
/// request -- it reports what happened as the response and as an anomaly
/// value on the construct's node, consistent with the runtime holding no
/// verdict to compare against.
async fn render_construct(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<RenderRequest>,
) -> impl IntoResponse {
    if !authorised(&headers, &state.token) {
        return unauthorised().into_response();
    }

    let prog = match sangoma::parse(&body.source) {
        Ok(p) => p,
        Err(d) => {
            return Json(RenderResponse { rendered: false, error: Some(d.message) })
                .into_response()
        }
    };
    let Some(construct) = prog.constructs.iter().find(|c| c.name == body.construct) else {
        return Json(RenderResponse {
            rendered: false,
            error: Some(format!("no construct named {:?}", body.construct)),
        })
        .into_response();
    };

    let dirs = state.plugin_dirs.clone();
    let construct = construct.clone();
    let frames = body.frames;
    let sample_rate = body.sample_rate;

    let result = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, String> {
        let available = clap_host::scan(&dirs);
        let mut buffer: Option<Vec<f32>> = None;
        for stage in &construct.stages {
            let Some(clap_id) = &stage.clap_id else {
                return Err(format!("stage {:?} has no clap binding", stage.name));
            };
            let descriptor = available
                .iter()
                .find(|d| &d.id == clap_id)
                .ok_or_else(|| format!("plugin {clap_id:?} not found for stage {:?}", stage.name))?;
            let mut plugin = clap_host::HostedPlugin::load(descriptor)
                .map_err(|e| format!("stage {:?}: {e}", stage.name))?;
            if let Some(gain) = stage.gain_db {
                // Stage 0's declared gain parameter, if the plugin exposes
                // one at that id -- a coarse first mapping, not a general
                // parameter scheme.
                plugin.set_param(0, gain);
            }
            let rendered = plugin
                .render(frames, sample_rate)
                .map_err(|e| format!("stage {:?}: {e}", stage.name))?;
            buffer = Some(rendered);
        }
        buffer.ok_or_else(|| "construct has no stages".to_string())
    })
    .await
    .unwrap_or_else(|e| Err(format!("render task panicked: {e}")));

    let (rendered, error, summary) = match result {
        Ok(samples) => {
            let summary = crate::integrations::analysis::write_and_measure(&samples, sample_rate, 2);
            (true, None, Some(summary))
        }
        Err(e) => (false, Some(e), None),
    };

    let deltas = {
        let mut rt = state.runtime.lock().await;
        if let Some(summary) = &summary {
            studio::commit_render_construct(&mut rt, &body.construct, summary);
        } else if let Some(e) = &error {
            let tau = format!("construct.{}", body.construct);
            rt.attach_chunk(&tau, "clap_render");
            if let Ok(v) = Value::new(
                format!("{tau}.note"),
                serde_json::json!(e),
                f64::MIN_POSITIVE,
                "",
                Kind::Anomaly,
                "clap_render",
            ) {
                rt.emit(&tau, v, Some(&tau));
            }
        }
        let deltas = rt.drain_deltas();
        if let Err(e) = crate::persist::save(&state.workspace, &rt.snapshot()) {
            tracing::warn!("could not persist runtime: {e}");
        }
        deltas
    };
    if !deltas.is_empty() {
        let _ = state.tx.send(ServerMessage::Delta { deltas });
    }

    Json(RenderResponse { rendered, error }).into_response()
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

/// Take the port, or explain why it could not be taken.
///
/// Separated from `serve_on` so the caller can acquire the socket *before*
/// announcing an address. Printing "listening" and then failing to bind
/// reports a success that did not happen, and sends the reader looking for
/// a bug in the wrong program -- the address in the banner is answering,
/// just not from here.
pub async fn listener(addr: SocketAddr) -> anyhow::Result<tokio::net::TcpListener> {
    match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => Ok(l),
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => Err(anyhow::anyhow!(
            "port {} is already held by another process.\n  Something else is listening \
             there -- possibly another heihachi, possibly an unrelated program.\n  \
             Pass a different port: --addr 127.0.0.1:{}",
            addr.port(),
            addr.port().wrapping_add(1)
        )),
        Err(e) => Err(e.into()),
    }
}

pub async fn serve_on(
    state: Arc<AppState>,
    listener: tokio::net::TcpListener,
) -> anyhow::Result<()> {
    axum::serve(listener, router(state)).await?;
    Ok(())
}

pub async fn bind(
    state: Arc<AppState>,
    addr: SocketAddr,
) -> anyhow::Result<()> {
    serve_on(state, listener(addr).await?).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_held_port_is_refused_with_a_remedy_not_a_bare_os_error() {
        // Hold a port, then ask for the same one. The second attempt must
        // fail, and must say what to do about it -- an unexplained bind
        // failure after a printed banner is how a port conflict gets
        // mistaken for a bug in this program.
        let first = listener("127.0.0.1:0".parse().unwrap()).await.unwrap();
        let addr = first.local_addr().unwrap();

        let err = listener(addr).await.expect_err("second bind must fail");
        let text = err.to_string();
        assert!(text.contains("already held"), "message was: {text}");
        assert!(text.contains("--addr"), "message must name the remedy: {text}");
    }

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
