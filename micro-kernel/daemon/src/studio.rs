//! The studio: what the daemon notices about work in progress.
//!
//! Watching a render folder is the whole FL integration. You render as
//! normal; the daemon sees the file, measures it, reads the project beside
//! it, asks the model for distinctions, and commits a node. Nothing is
//! installed into FL and nothing is controlled remotely, because FL exposes
//! no interface for that and pretending otherwise would be a worse tool.
//!
//! Every rung here degrades to a note. A render that will not decode, a
//! project that will not parse and a model that will not answer are three
//! different explanations, and the run report is worth nothing if they are
//! collapsed into a blank.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::graph::cut::{ContactGraph, MEDIUM};
use crate::graph::{Kind, Runtime, Value};
use crate::integrations::analysis::{self, AudioSummary};
use crate::integrations::fl::{self, ProjectChain};

/// A render the daemon noticed, with what each rung recovered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportEvent {
    pub path: String,
    pub stem: String,
    pub bytes: u64,
    pub audio: AudioSummary,
    pub project: Option<ProjectChain>,
    /// Distinctions drawn by the model rung, when it ran.
    pub distinctions: Vec<String>,
    /// One line per rung that could not run, and why.
    pub notes: Vec<String>,
}

#[derive(Debug, Default)]
pub struct Studio {
    watching: Option<PathBuf>,
    exports: Vec<ExportEvent>,
    seen: BTreeSet<String>,
}

impl Studio {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn watch(&mut self, dir: PathBuf) {
        self.watching = Some(dir);
    }

    pub fn watching(&self) -> Option<&Path> {
        self.watching.as_deref()
    }

    pub fn exports(&self) -> &[ExportEvent] {
        &self.exports
    }

    /// Whether this exact version of this render has already been committed.
    ///
    /// Keyed on path *and* modification time, for two reasons. A single
    /// write raises more than one filesystem event, and committing on each
    /// would record the same render twice. But re-rendering to the same
    /// filename is the normal way to work -- a producer bounces
    /// `bass_v1.wav`, changes something, and bounces it again -- and that
    /// must commit, or the record misses exactly the revisions it exists
    /// to hold.
    pub fn already_seen(&self, path: &Path) -> bool {
        self.seen.contains(&Self::key(path))
    }

    fn key(path: &Path) -> String {
        let stamp = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis())
            .unwrap_or(0);
        format!("{}|{stamp}", path.display())
    }

    /// Claim a render for commitment, returning false if it is already
    /// claimed.
    ///
    /// Check and mark happen together, under one lock, because measuring a
    /// render and asking the model about it takes seconds -- long enough
    /// for the second filesystem event of the same write to arrive and pass
    /// a check that had not yet recorded anything.
    pub fn claim(&mut self, path: &Path) -> bool {
        self.seen.insert(Self::key(path))
    }

    pub fn record_export(&mut self, path: PathBuf, event: ExportEvent) {
        self.seen.insert(Self::key(&path));
        self.exports.push(event);
        // keep the live list bounded; the graph holds the durable record
        if self.exports.len() > 500 {
            self.exports.remove(0);
        }
    }
}

/// Measure a render and everything around it. Never fails.
pub fn observe_export(path: &Path) -> ExportEvent {
    let mut notes = Vec::new();

    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled")
        .to_string();

    let audio = analysis::analyse(path);
    if let Some(note) = &audio.note {
        notes.push(format!("measurement: {note}"));
    }

    let project = fl::project_beside(path).map(|p| fl::read_project(&p));
    if let Some(chain) = &project {
        if let Some(note) = &chain.note {
            notes.push(format!("project: {note}"));
        }
    } else {
        notes.push("project: no .flp found beside this render".into());
    }

    ExportEvent {
        path: path.display().to_string(),
        stem,
        bytes,
        audio,
        project,
        distinctions: Vec::new(),
        notes,
    }
}

/// Commit an export to the runtime graph.
///
/// One node per render, bearing its measurements. Each measurement carries
/// the floor at which it was obtained, so nothing downstream can claim a
/// distinction the instrument did not make.
pub fn commit_export(rt: &mut Runtime, event: &ExportEvent) {
    let tau = format!("render.{}", event.stem);
    rt.attach_chunk(&tau, "measurement");

    for m in &event.audio.measurements {
        if let Ok(value) =
            Value::reading(m.channel.clone(), m.value, m.floor, m.unit.clone(), "measurement")
        {
            rt.emit(&tau, value, Some(&tau));
        }
    }

    // The chain that produced it, when the project could be read.
    if let Some(chain) = &event.project {
        if !chain.devices.is_empty() {
            rt.attach_chunk(&tau, "project");
            if let Ok(value) = Value::new(
                format!("{tau}.devices"),
                serde_json::json!(chain.devices),
                1.0,
                "",
                Kind::Finding,
                "project",
            ) {
                rt.emit(&tau, value, Some(&tau));
            }
        }
        if let Some(tempo) = chain.tempo {
            // FL stores tempo in millibeats, so the floor is one thousandth
            if let Ok(value) =
                Value::reading(format!("{tau}.tempo"), tempo, 0.001, "bpm", "project")
            {
                rt.emit(&tau, value, Some(&tau));
            }
        }
    }

    if !event.distinctions.is_empty() {
        rt.attach_chunk(&tau, "model");
        if let Ok(value) = Value::new(
            format!("{tau}.distinctions"),
            serde_json::json!(event.distinctions),
            1.0,
            "",
            Kind::Finding,
            "model",
        ) {
            rt.emit(&tau, value, Some(&tau));
        }
    }

    // A rung that could not run contributes its explanation, not a blank.
    for note in &event.notes {
        if let Ok(value) = Value::new(
            format!("{tau}.note"),
            serde_json::json!(note),
            f64::MIN_POSITIVE,
            "",
            Kind::Anomaly,
            "studio",
        ) {
            rt.emit(&tau, value, Some(&tau));
        }
    }
}

/// Commit a rendered construct's measurements to the runtime graph.
///
/// Uses the same `tau = "construct.{name}"` that the `/api/run` handler
/// already emits reachability-check values under, so a rendered construct's
/// measurements land on the same node as its static check -- two different
/// rungs (the check, the render) meeting on one subtask rather than each
/// claiming a node of its own.
pub fn commit_render_construct(rt: &mut Runtime, construct_name: &str, summary: &AudioSummary) {
    let tau = format!("construct.{construct_name}");
    rt.attach_chunk(&tau, "clap_render");

    for m in &summary.measurements {
        if let Ok(value) =
            Value::reading(m.channel.clone(), m.value, m.floor, m.unit.clone(), "clap_render")
        {
            rt.emit(&tau, value, Some(&tau));
        }
    }

    if let Some(note) = &summary.note {
        if let Ok(value) = Value::new(
            format!("{tau}.note"),
            serde_json::json!(note),
            f64::MIN_POSITIVE,
            "",
            Kind::Anomaly,
            "clap_render",
        ) {
            rt.emit(&tau, value, Some(&tau));
        }
    }
}

/// A one-line description of the material, for the model prompt.
pub fn describe(event: &ExportEvent) -> String {
    let mut lines = Vec::new();
    for m in &event.audio.measurements {
        lines.push(format!(
            "{} = {:.3} {} (resolved to {:.4})",
            m.channel, m.value, m.unit, m.floor
        ));
    }
    if let Some(chain) = &event.project {
        if !chain.devices.is_empty() {
            lines.push(format!("devices in the chain: {}", chain.devices.join(", ")));
        }
    }
    lines.join("\n")
}

#[derive(Debug, Clone, Serialize)]
pub struct AccountabilityReport {
    pub outcome: String,
    pub weight: f64,
    pub contributors: Vec<Contributor>,
    /// Set when the outcome is not in the graph, or nothing reaches it.
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Contributor {
    pub from: String,
    pub to: String,
    pub weight: f64,
}

/// Which emissions were jointly necessary for an outcome.
///
/// This is a post-hoc query over the provenance graph, not a verdict: it
/// says what would have had to be different, and says nothing about whether
/// the outcome was good.
pub fn accountability(rt: &Runtime, outcome: &str) -> AccountabilityReport {
    let mut graph = match ContactGraph::new(0.02) {
        Ok(g) => g,
        Err(e) => {
            return AccountabilityReport {
                outcome: outcome.to_string(),
                weight: 0.0,
                contributors: Vec::new(),
                note: Some(e.to_string()),
            }
        }
    };

    // Weight each induced edge by how many values passed along it, floored
    // so that every committed contact is at least the floor.
    let mut any = false;
    for (from, to) in rt.edges() {
        let count = rt.read(to).len().max(1) as f64;
        if graph.link(from, to, 0.02 * count).is_ok() {
            any = true;
        }
    }
    for node in rt.nodes() {
        let _ = graph.attach(&node.tau, 0.02);
    }

    if !any {
        return AccountabilityReport {
            outcome: outcome.to_string(),
            weight: 0.0,
            contributors: Vec::new(),
            note: Some(
                "no causal edges have been induced yet: run something that reads \
                 what another step emitted"
                    .into(),
            ),
        };
    }
    if !rt.nodes().any(|n| n.tau == outcome) {
        return AccountabilityReport {
            outcome: outcome.to_string(),
            weight: 0.0,
            contributors: Vec::new(),
            note: Some(format!("no node named {outcome:?} is in the graph")),
        };
    }

    let acc = graph.accountable_set(MEDIUM, outcome);
    AccountabilityReport {
        outcome: outcome.to_string(),
        weight: acc.weight,
        contributors: acc
            .edges
            .into_iter()
            .map(|e| Contributor { from: e.from, to: e.to, weight: e.weight })
            .collect(),
        note: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observing_a_missing_render_yields_notes_not_a_panic() {
        let event = observe_export(Path::new("nowhere/absent.wav"));
        assert!(!event.notes.is_empty());
        assert!(event.audio.measurements.is_empty());
    }

    #[test]
    fn committing_an_export_with_notes_records_them_as_values() {
        let mut rt = Runtime::new();
        let event = observe_export(Path::new("nowhere/absent.wav"));
        commit_export(&mut rt, &event);
        // the notes became anomaly values rather than being dropped
        assert!(!rt.anomalies().is_empty());
        assert_eq!(rt.report().anomalies, event.notes.len());
    }

    #[test]
    fn accountability_on_an_empty_graph_explains_itself() {
        let rt = Runtime::new();
        let report = accountability(&rt, "render.anything");
        assert!(report.note.is_some());
        assert!(report.contributors.is_empty());
    }

    #[test]
    fn a_studio_does_not_commit_the_same_render_twice() {
        let mut studio = Studio::new();
        let path = PathBuf::from("bounce.wav");
        assert!(studio.claim(&path), "first claim succeeds");
        assert!(!studio.claim(&path), "second claim of the same version is refused");
        studio.record_export(path.clone(), observe_export(&path));
        assert!(studio.already_seen(&path));
    }
}
