//! The heihachi CLI.
//!
//!   heihachi serve --watch <dir>   run the daemon and print a pairing token
//!   heihachi check <file>          check a .mma or .sgn without running it
//!   heihachi observe <render>      measure one render and print the values
//!
//! `serve` binds to loopback. The pairing token lets a page served from
//! anywhere drive compute here; the bind address is what keeps anything
//! else from reaching it.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use rand::Rng;
use tokio::sync::{broadcast, Mutex};

use heihachi_daemon::graph::Runtime;
use heihachi_daemon::integrations::ollama::{Ollama, DEFAULT_ENDPOINT};
use heihachi_daemon::lang::{mishima, sangoma, Severity};
use heihachi_daemon::server::{self, AppState, ServerMessage};
use heihachi_daemon::studio::{self, Studio};

#[derive(Parser)]
#[command(name = "heihachi", version, about = "The heihachi micro-kernel daemon")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the local daemon and print a pairing token.
    Serve {
        /// Address to bind. Loopback by default, and you should leave it there.
        #[arg(long, default_value = "127.0.0.1:7749")]
        addr: String,
        /// Folder to watch for renders. Point this at your FL export folder.
        #[arg(long)]
        watch: Option<PathBuf>,
        /// Where .mma and .sgn files live.
        #[arg(long, default_value = "workspace")]
        workspace: PathBuf,
        /// Ollama endpoint.
        #[arg(long, default_value = DEFAULT_ENDPOINT)]
        ollama: String,
        /// Model to use as the constructor rung.
        #[arg(long, default_value = "llama3.2")]
        model: String,
        /// Use this token instead of generating one. For scripts.
        #[arg(long)]
        token: Option<String>,
    },
    /// Check a program without running it.
    Check {
        file: PathBuf,
        /// The resolution your analysis path delivers.
        #[arg(long, default_value = "0.001")]
        backend_resolution: f64,
    },
    /// Measure one render and print what was recovered.
    Observe { render: PathBuf },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "heihachi_daemon=info,warn".into()),
        )
        .with_target(false)
        .init();

    match Cli::parse().command {
        Command::Serve { addr, watch, workspace, ollama, model, token } => {
            serve(addr, watch, workspace, ollama, model, token).await
        }
        Command::Check { file, backend_resolution } => check(file, backend_resolution),
        Command::Observe { render } => observe(render),
    }
}

fn generate_token() -> String {
    const ALPHABET: &[u8] = b"abcdefghijkmnopqrstuvwxyz23456789";
    let mut rng = rand::thread_rng();
    let body: String = (0..24)
        .map(|_| ALPHABET[rng.gen_range(0..ALPHABET.len())] as char)
        .collect();
    format!("hk_{body}")
}

async fn serve(
    addr: String,
    watch: Option<PathBuf>,
    workspace: PathBuf,
    ollama_endpoint: String,
    model: String,
    token: Option<String>,
) -> anyhow::Result<()> {
    let addr: SocketAddr = addr.parse()?;
    let token = token.unwrap_or_else(generate_token);

    // make the workspace so the file tree is never empty on first run
    for sub in ["mishima", "sangoma"] {
        std::fs::create_dir_all(workspace.join(sub)).ok();
    }
    seed_workspace(&workspace);

    let (tx, _rx) = broadcast::channel(256);
    let mut studio = Studio::new();
    if let Some(dir) = &watch {
        studio.watch(dir.clone());
    }

    let state = Arc::new(AppState {
        token: token.clone(),
        runtime: Mutex::new(Runtime::new()),
        studio: Mutex::new(studio),
        ollama: Ollama::new(ollama_endpoint.clone(), model.clone()),
        workspace: workspace.clone(),
        tx: tx.clone(),
        // the analysis path is f64 throughout, so its error is far below
        // any floor an audio measurement will declare
        backend_resolution: 1e-9,
    });

    if let Some(dir) = watch.clone() {
        spawn_watcher(dir, Arc::clone(&state));
    }

    let reachable = state.ollama.models().await.is_ok();

    println!();
    println!("  heihachi daemon");
    println!("  ───────────────────────────────────────────────");
    println!("  listening   http://{addr}");
    println!("  token       {token}");
    println!("  workspace   {}", workspace.display());
    match &watch {
        Some(d) => println!("  watching    {}", d.display()),
        None => println!("  watching    (nothing -- pass --watch <render folder>)"),
    }
    println!(
        "  ollama      {ollama_endpoint} [{model}] {}",
        if reachable { "reachable" } else { "UNREACHABLE" }
    );
    println!("  ───────────────────────────────────────────────");
    println!("  paste the token into the web tool to pair.");
    println!("  everything stays on this machine.");
    println!();

    server::bind(state, addr).await
}

/// Watch a render folder and commit what appears.
///
/// A debounce is used because a DAW writes a render in several bursts; the
/// file is not measured until it has stopped changing.
fn spawn_watcher(dir: PathBuf, state: Arc<AppState>) {
    use notify::{RecursiveMode, Watcher};
    use notify_debouncer_full::new_debouncer;

    // Capture the handle here, on the runtime thread: the watcher runs on a
    // plain OS thread and cannot ask for a reactor it is not inside.
    let runtime = tokio::runtime::Handle::current();

    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut debouncer = match new_debouncer(Duration::from_secs(2), None, tx) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!("cannot watch {}: {e}", dir.display());
                return;
            }
        };
        if let Err(e) = debouncer.watcher().watch(&dir, RecursiveMode::Recursive) {
            tracing::error!("cannot watch {}: {e}", dir.display());
            return;
        }
        tracing::info!("watching {}", dir.display());

        for result in rx {
            let Ok(events) = result else { continue };
            for event in events {
                for path in &event.paths {
                    if !heihachi_daemon::integrations::fl::is_audio(path) {
                        continue;
                    }
                    let path = path.clone();
                    let state = Arc::clone(&state);
                    runtime.spawn(async move {
                        commit_render(path, state).await;
                    });
                }
            }
        }
    });
}

async fn commit_render(path: PathBuf, state: Arc<AppState>) {
    {
        let studio = state.studio.lock().await;
        if studio.already_seen(&path) {
            return;
        }
    }

    let mut event = studio::observe_export(&path);

    // The model rung runs last and may decline to run at all; a rung that
    // did not run drew no distinctions, and says so.
    let described = studio::describe(&event);
    if !described.is_empty() {
        let term_map = state.ollama.draw_distinctions(&described).await;
        event.distinctions = term_map.distinctions;
        if let Some(note) = term_map.note {
            event.notes.push(format!("model: {note}"));
        }
    }

    let deltas = {
        let mut rt = state.runtime.lock().await;
        studio::commit_export(&mut rt, &event);
        rt.drain_deltas()
    };

    tracing::info!(
        "committed {} ({} measurements, {} distinctions)",
        event.stem,
        event.audio.measurements.len(),
        event.distinctions.len()
    );

    {
        let mut studio = state.studio.lock().await;
        studio.record_export(path, event.clone());
    }

    let _ = state.tx.send(ServerMessage::Export { event: Box::new(event) });
    if !deltas.is_empty() {
        let _ = state.tx.send(ServerMessage::Delta { deltas });
    }
}

fn check(file: PathBuf, backend_resolution: f64) -> anyhow::Result<()> {
    let source = std::fs::read_to_string(&file)?;
    let is_sangoma = file
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("sgn"))
        .unwrap_or(false);

    let result = if is_sangoma {
        match sangoma::parse(&source) {
            Ok(prog) => sangoma::check(&prog, 0.8, backend_resolution),
            Err(d) => heihachi_daemon::lang::CheckResult::from_diagnostics(
                vec![d],
                Vec::new(),
            ),
        }
    } else {
        match mishima::parse(&source) {
            Ok(prog) => mishima::check(&prog, backend_resolution),
            Err(d) => heihachi_daemon::lang::CheckResult::from_diagnostics(
                vec![d],
                Vec::new(),
            ),
        }
    };

    for d in &result.diagnostics {
        let mark = match d.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        println!(
            "{}:{}: {mark} [{}]\n  {}\n  remedy: {}",
            file.display(),
            d.line,
            d.rule,
            d.message,
            d.remedy
        );
    }
    if result.diagnostics.is_empty() {
        println!("{}: no diagnostics", file.display());
    }
    // A refused program exits non-zero so a shell can act on it. This is a
    // fact about the check, not a verdict from the runtime.
    if result.accepted {
        Ok(())
    } else {
        std::process::exit(1)
    }
}

fn observe(render: PathBuf) -> anyhow::Result<()> {
    let event = studio::observe_export(&render);
    println!("{}", event.stem);
    if event.audio.measurements.is_empty() {
        println!("  (no measurements)");
    }
    for m in &event.audio.measurements {
        println!(
            "  {:<22} {:>10.3} {:<6} resolved to {:.5}",
            m.channel, m.value, m.unit, m.floor
        );
    }
    if let Some(chain) = &event.project {
        if !chain.devices.is_empty() {
            println!("  devices: {}", chain.devices.join(", "));
        }
        if let Some(tempo) = chain.tempo {
            println!("  tempo:   {tempo:.3} bpm");
        }
    }
    for note in &event.notes {
        println!("  note: {note}");
    }
    Ok(())
}

/// Write the worked examples from the papers, once, so a first run has
/// something to open.
fn seed_workspace(workspace: &PathBuf) {
    let recall = workspace.join("mishima").join("recall.mma");
    if !recall.exists() {
        let _ = std::fs::write(
            &recall,
            include_str!("../examples/recall.mma"),
        );
    }
    let reese = workspace.join("sangoma").join("reese.sgn");
    if !reese.exists() {
        let _ = std::fs::write(
            &reese,
            include_str!("../examples/reese.sgn"),
        );
    }
}
