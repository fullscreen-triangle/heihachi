//! The runtime graph.
//!
//! This is Paper I made executable. The kernel's entire semantic content is
//! [`Runtime::run`]: execute every chunk in a node's bag and increment the
//! record once per emission. It does not select among chunks, does not order
//! the graph, and does not inspect the values that result.
//!
//! Two invariants are enforced by construction rather than by discipline:
//!
//!   * a [`Value`] of non-positive floor cannot be built ([`Value::new`]
//!     returns `Err`), so a claim of exact measurement is not expressible; and
//!   * [`Runtime::report`] exposes no field carrying a verdict, because the
//!     graph stores no expectation against which one could be computed.

pub mod cut;
pub mod reach;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Why a value was emitted. Anomalies are values like any other: nothing in
/// the kernel branches on this field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Reading,
    Anomaly,
    Finding,
    Decline,
}

/// A datum attached to a node.
///
/// `floor` is never optional and never zero. A value with no stated
/// resolution would be a claim of exact measurement, which the language
/// refuses to express.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Value {
    pub channel: String,
    pub magnitude: serde_json::Value,
    pub floor: f64,
    pub unit: String,
    pub kind: Kind,
    pub record: u64,
    pub origin: String,
    /// Channels this emission was computed from, for provenance.
    pub reads: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ValueError {
    #[error("a value with floor {0} is not constructible: \
             a claim of exact measurement is not expressible")]
    NonPositiveFloor(f64),
}

impl Value {
    pub fn new(
        channel: impl Into<String>,
        magnitude: serde_json::Value,
        floor: f64,
        unit: impl Into<String>,
        kind: Kind,
        origin: impl Into<String>,
    ) -> Result<Self, ValueError> {
        if !(floor > 0.0) {
            return Err(ValueError::NonPositiveFloor(floor));
        }
        Ok(Self {
            channel: channel.into(),
            magnitude,
            floor,
            unit: unit.into(),
            kind,
            record: 0,
            origin: origin.into(),
            reads: Vec::new(),
        })
    }

    /// A numeric reading, the common case.
    pub fn reading(
        channel: impl Into<String>,
        magnitude: f64,
        floor: f64,
        unit: impl Into<String>,
        origin: impl Into<String>,
    ) -> Result<Self, ValueError> {
        Self::new(
            channel,
            serde_json::json!(magnitude),
            floor,
            unit,
            Kind::Reading,
            origin,
        )
    }
}

/// A node: a subtask identity, a bag of chunks, and the values that accreted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// The subtask identity. This alone individuates the node.
    pub tau: String,
    /// Names of the chunks in the bag. The bodies live in the executor.
    pub chunks: Vec<String>,
    pub values: Vec<Value>,
}

impl Node {
    fn new(tau: String) -> Self {
        Self { tau, chunks: Vec::new(), values: Vec::new() }
    }
}

/// The result of a run. Deliberately carries no verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub record: u64,
    pub nodes_executed: usize,
    pub emissions: usize,
    pub anomalies: usize,
    pub induced_edges: usize,
}

/// One change to the graph, for streaming to a connected interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Delta {
    NodeRaised { tau: String },
    ChunkAttached { tau: String, chunk: String },
    ValueEmitted { tau: String, value: Box<Value> },
    EdgeInduced { from: String, to: String },
}

/// The kernel.
#[derive(Debug, Default)]
pub struct Runtime {
    nodes: BTreeMap<String, Node>,
    record: u64,
    log: Vec<Value>,
    edges: BTreeSet<(String, String)>,
    executed: Vec<String>,
    /// Deltas since the last drain, for the live view.
    pending: Vec<Delta>,
}

impl Runtime {
    pub fn new() -> Self {
        Self::default()
    }

    // ── the graph vocabulary: identify, read, transform, emit ───────────
    //
    // There is no fifth verb. In particular there is no `is_correct` and no
    // `expected`: the graph stores what was emitted, never what should have
    // been.

    /// Locate the node bearing a subtask identity, raising it if new.
    ///
    /// Convergence rather than creation: raising a subtask that already
    /// exists returns the existing node, so two modules that decompose
    /// different problems and arrive at the same subtask meet on one node.
    pub fn identify(&mut self, tau: &str) -> &mut Node {
        if !self.nodes.contains_key(tau) {
            self.nodes.insert(tau.to_string(), Node::new(tau.to_string()));
            self.pending.push(Delta::NodeRaised { tau: tau.to_string() });
        }
        self.nodes.get_mut(tau).expect("just inserted")
    }

    pub fn read(&self, tau: &str) -> &[Value] {
        self.nodes.get(tau).map(|n| n.values.as_slice()).unwrap_or(&[])
    }

    pub fn attach_chunk(&mut self, tau: &str, chunk: impl Into<String>) {
        let chunk = chunk.into();
        self.identify(tau).chunks.push(chunk.clone());
        self.pending.push(Delta::ChunkAttached { tau: tau.to_string(), chunk });
    }

    /// Attach a value to a node and advance the record.
    ///
    /// The record is incremented once per emission and never decremented;
    /// re-measuring is a new commitment at a strictly higher record, never a
    /// cached recomputation.
    pub fn emit(&mut self, tau: &str, mut value: Value, origin_tau: Option<&str>) {
        self.record += 1;
        value.record = self.record;

        if let Some(from) = origin_tau {
            if from != tau {
                let edge = (from.to_string(), tau.to_string());
                if self.edges.insert(edge.clone()) {
                    self.pending.push(Delta::EdgeInduced { from: edge.0, to: edge.1 });
                }
            }
        }

        self.identify(tau).values.push(value.clone());
        self.log.push(value.clone());
        self.pending.push(Delta::ValueEmitted {
            tau: tau.to_string(),
            value: Box::new(value),
        });
    }

    // ── execution ───────────────────────────────────────────────────────

    /// Execute every chunk in the node's bag.
    ///
    /// No chunk is skipped on the strength of what another produced. A chunk
    /// that fails contributes an anomaly value rather than halting the run:
    /// nothing here branches on emission content, so an error cannot alter
    /// control flow.
    pub fn run<F>(&mut self, tau: &str, mut execute: F)
    where
        F: FnMut(&str, &str) -> Result<Vec<Value>, String>,
    {
        let chunks = self.identify(tau).chunks.clone();
        self.executed.push(tau.to_string());

        for chunk in chunks {
            let emissions = match execute(tau, &chunk) {
                Ok(vs) => vs,
                Err(message) => vec![Value::new(
                    format!("{tau}.anomaly"),
                    serde_json::json!(message),
                    f64::MIN_POSITIVE,
                    "",
                    Kind::Anomaly,
                    chunk.clone(),
                )
                .expect("MIN_POSITIVE is positive")],
            };
            for value in emissions {
                self.emit(tau, value, Some(tau));
            }
        }
    }

    // ── provenance and reporting ────────────────────────────────────────

    /// The nodes that carried propagated information.
    ///
    /// A node whose emissions nothing read is absent from this set, and no
    /// actor performed a rejection to exclude it.
    pub fn trajectory(&self) -> Vec<String> {
        let mut seen = BTreeSet::new();
        for (u, v) in &self.edges {
            seen.insert(u.clone());
            seen.insert(v.clone());
        }
        seen.into_iter().collect()
    }

    pub fn anomalies(&self) -> Vec<&Value> {
        self.log.iter().filter(|v| v.kind == Kind::Anomaly).collect()
    }

    /// The runtime's output. Not an exit code.
    ///
    /// No field here carries a verdict, because a verdict presupposes a
    /// comparison against an expectation and the graph stores none.
    pub fn report(&self) -> Report {
        Report {
            record: self.record,
            nodes_executed: self.executed.len(),
            emissions: self.log.len(),
            anomalies: self.anomalies().len(),
            induced_edges: self.edges.len(),
        }
    }

    pub fn record(&self) -> u64 {
        self.record
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edges(&self) -> impl Iterator<Item = &(String, String)> {
        self.edges.iter()
    }

    /// Take the deltas accumulated since the last call.
    pub fn drain_deltas(&mut self) -> Vec<Delta> {
        std::mem::take(&mut self.pending)
    }

    // ── persistence ─────────────────────────────────────────────────────

    /// Capture the durable parts of the record.
    ///
    /// `log` and `pending` are excluded: `log` is redundant with the union of
    /// `node.values` (reconstructed on restore), and `pending` is a live-view
    /// cursor with nothing to catch up to before a first client connects.
    pub fn snapshot(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            nodes: self.nodes.clone(),
            record: self.record,
            edges: self.edges.iter().cloned().collect(),
            executed: self.executed.clone(),
        }
    }

    /// Rebuild a runtime from a snapshot. `log` is reconstructed from node
    /// values sorted by record, so `report()`/`anomalies()` behave exactly as
    /// they would for a runtime built by replaying the original `emit` calls.
    pub fn restore(snapshot: RuntimeSnapshot) -> Self {
        let mut log: Vec<Value> = snapshot
            .nodes
            .values()
            .flat_map(|n| n.values.iter().cloned())
            .collect();
        log.sort_by_key(|v| v.record);

        Self {
            nodes: snapshot.nodes,
            record: snapshot.record,
            log,
            edges: snapshot.edges.into_iter().collect(),
            executed: snapshot.executed,
            pending: Vec::new(),
        }
    }
}

/// The durable subset of [`Runtime`]'s state, for persistence across restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub nodes: BTreeMap<String, Node>,
    pub record: u64,
    pub edges: Vec<(String, String)>,
    pub executed: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_chunk(_tau: &str, _chunk: &str) -> Result<Vec<Value>, String> {
        Ok(vec![Value::reading("a.level", -6.0, 0.1, "dB", "t").unwrap()])
    }

    #[test]
    fn zero_floor_value_is_not_constructible() {
        for bad in [0.0, -1.0, -1e-9] {
            assert!(Value::reading("x", 1.0, bad, "", "t").is_err());
        }
        assert!(Value::reading("x", 1.0, 1e-12, "", "t").is_ok());
    }

    #[test]
    fn report_carries_no_verdict() {
        let mut rt = Runtime::new();
        rt.attach_chunk("a", "ok");
        rt.attach_chunk("a", "bad");
        rt.run("a", |_, chunk| {
            if chunk == "bad" {
                Err("clipped".into())
            } else {
                ok_chunk("a", chunk)
            }
        });

        let json = serde_json::to_value(rt.report()).unwrap();
        let keys: Vec<_> = json.as_object().unwrap().keys().cloned().collect();
        for forbidden in ["success", "failure", "exit_code", "status", "ok", "passed"] {
            assert!(!keys.contains(&forbidden.to_string()));
        }
        // the anomaly was recorded, and the sibling chunk still ran
        assert_eq!(rt.report().anomalies, 1);
        assert_eq!(rt.report().emissions, 2);
    }

    #[test]
    fn every_chunk_runs_even_after_one_fails() {
        let mut rt = Runtime::new();
        for name in ["first", "raises", "third", "fourth"] {
            rt.attach_chunk("n", name);
        }
        let mut fired = Vec::new();
        rt.run("n", |_, chunk| {
            fired.push(chunk.to_string());
            if chunk == "raises" {
                Err("boom".into())
            } else {
                ok_chunk("n", chunk)
            }
        });
        assert_eq!(fired.len(), 4);
        assert_eq!(fired.last().unwrap(), "fourth");
    }

    #[test]
    fn record_is_monotone_and_never_memoised() {
        let mut rt = Runtime::new();
        rt.attach_chunk("a", "m");
        let mut seq = Vec::new();
        for _ in 0..400 {
            rt.run("a", ok_chunk);
            seq.push(rt.record());
        }
        assert!(seq.windows(2).all(|w| w[1] > w[0]));

        let mut repeats = Vec::new();
        for _ in 0..3 {
            rt.run("a", ok_chunk);
            repeats.push(rt.record());
        }
        assert_eq!(repeats, vec![401, 402, 403]);
    }

    #[test]
    fn two_modules_converge_on_one_node() {
        let mut rt = Runtime::new();
        rt.attach_chunk("bass.transient", "spectral");
        rt.attach_chunk("bass.transient", "annotation");
        rt.run("bass.transient", ok_chunk);
        assert_eq!(rt.node_count(), 1);
        assert_eq!(rt.read("bass.transient").len(), 2);
    }

    #[test]
    fn snapshot_then_restore_preserves_record_and_nodes() {
        let mut rt = Runtime::new();
        rt.attach_chunk("a", "m");
        rt.run("a", ok_chunk);
        rt.emit(
            "b",
            Value::reading("b.v", 1.0, 0.1, "dB", "m").unwrap(),
            Some("a"),
        );

        let snap = rt.snapshot();
        let restored = Runtime::restore(snap);

        assert_eq!(restored.record(), rt.record());
        assert_eq!(restored.node_count(), rt.node_count());
        assert_eq!(restored.read("a").len(), rt.read("a").len());
        assert_eq!(restored.read("b").len(), rt.read("b").len());
        assert_eq!(restored.report().emissions, rt.report().emissions);
        assert_eq!(
            restored.edges().collect::<Vec<_>>(),
            rt.edges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn restored_runtime_continues_the_monotone_sequence() {
        let mut rt = Runtime::new();
        rt.attach_chunk("a", "m");
        rt.run("a", ok_chunk);
        let before = rt.record();

        let mut restored = Runtime::restore(rt.snapshot());
        restored.attach_chunk("a", "m");
        restored.run("a", ok_chunk);

        assert!(restored.record() > before);
    }

    #[test]
    fn unread_emission_leaves_no_edge_and_needs_no_rejection() {
        let mut rt = Runtime::new();
        rt.attach_chunk("heard", "h");
        rt.attach_chunk("unheard", "u");
        rt.run("heard", ok_chunk);
        rt.run("unheard", ok_chunk);
        rt.emit(
            "downstream",
            Value::reading("d.v", 2.0, 0.1, "dB", "m").unwrap(),
            Some("heard"),
        );
        let traj = rt.trajectory();
        assert!(traj.contains(&"heard".to_string()));
        assert!(!traj.contains(&"unheard".to_string()));
    }
}
