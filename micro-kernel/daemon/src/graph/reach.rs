//! `nec ∘ seek`: reachability-then-necessity closure over the live graph.
//!
//! Ports `seek_to_closure` from `validation/hkcore.py`, which the Python
//! validation suite already checks against synthetic registries
//! (`exp_mishima.py`'s closure-vs-threshold experiment, Paper II Thm 6.3).
//! The closure algorithm itself is pure arithmetic and translates directly;
//! what has no Python analogue is [`class_reached`], which must read a rung's
//! landing class off the real runtime graph instead of a hand-set fixture.

use crate::graph::Runtime;
use crate::lang::{composite_power, mishima};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DeterminationStatus {
    Converged,
    Declined,
}

#[derive(Debug, Clone, Serialize)]
pub struct Determination {
    pub status: DeterminationStatus,
    pub classes: Vec<String>,
    pub probes_invoked: Vec<String>,
}

/// Which class, if any, a rung reaches -- read off the graph rather than
/// declared on a fixture.
///
/// A rung reaches a node's class when some value the node carries was
/// emitted on a channel exactly equal to the rung's name, or nested under it
/// (`"{rung_name}."` prefix), matching the `{tau}.composite_power` /
/// `{tau}.target.{name}` naming convention already used in `server.rs`. This
/// is deliberately stricter than substring-anywhere: a rung `spectral` must
/// not match a node whose channel merely contains that text, such as
/// `render.non_spectral_thing`.
pub fn class_reached(rt: &Runtime, rung_name: &str) -> Option<String> {
    let prefix = format!("{rung_name}.");
    rt.nodes()
        .find(|n| {
            n.values
                .iter()
                .any(|v| v.channel == rung_name || v.channel.starts_with(&prefix))
        })
        .map(|n| n.tau.clone())
}

/// Invoke a seek's ladder until no remaining rung reaches a new class.
///
/// A rung that reaches nothing contributes nothing -- it is neither counted
/// toward agreement nor forced into a decline. Contested closure (more than
/// one class reached) is `Declined`, and that is data describing what the
/// ladder found, never an error.
pub fn seek_to_closure(rt: &Runtime, seek: &mishima::Seek, threshold: f64) -> Determination {
    let mut reached: Vec<String> = Vec::new();
    let mut invoked: Vec<String> = Vec::new();
    let mut running = 0.0;

    for rung in &seek.ladder {
        invoked.push(rung.name.clone());
        running = composite_power([running, rung.power]);

        if let Some(class) = class_reached(rt, &rung.name) {
            if !reached.contains(&class) {
                reached.push(class);
            }
        }
        let _ = threshold; // reserved for a future threshold-vs-closure report

        let remaining = &seek.ladder[invoked.len()..];
        let closed = remaining.iter().all(|r| match class_reached(rt, &r.name) {
            Some(class) => reached.contains(&class),
            None => true,
        });
        if closed {
            break;
        }
    }

    let status = if reached.len() <= 1 {
        DeterminationStatus::Converged
    } else {
        DeterminationStatus::Declined
    };

    Determination { status, classes: reached, probes_invoked: invoked }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Value;
    use crate::lang::mishima::{Rung, Seek};

    fn seek_with(rungs: &[(&str, f64)]) -> Seek {
        Seek {
            subject: "test".into(),
            exclusions: vec!["x".into()],
            toward: "y".into(),
            ladder: rungs.iter().map(|(n, p)| Rung { name: n.to_string(), power: *p }).collect(),
            admit: "closure".into(),
            otherwise_decline: true,
            result: "r".into(),
            line: 1,
        }
    }

    #[test]
    fn a_single_landing_class_converges() {
        let mut rt = Runtime::new();
        rt.emit("dub", Value::reading("spectral", 1.0, 0.1, "", "t").unwrap(), None);
        rt.emit("dub", Value::reading("annotation", 1.0, 0.1, "", "t").unwrap(), Some("dub"));

        let seek = seek_with(&[("spectral", 0.6), ("annotation", 0.5)]);
        let det = seek_to_closure(&rt, &seek, 0.99);

        assert_eq!(det.status, DeterminationStatus::Converged);
        assert_eq!(det.classes, vec!["dub".to_string()]);
    }

    #[test]
    fn two_irreconcilable_landing_classes_decline() {
        let mut rt = Runtime::new();
        rt.emit("dub", Value::reading("spectral", 1.0, 0.1, "", "t").unwrap(), None);
        rt.emit("vip", Value::reading("annotation", 1.0, 0.1, "", "t").unwrap(), Some("dub"));

        let seek = seek_with(&[("spectral", 0.6), ("annotation", 0.5)]);
        let det = seek_to_closure(&rt, &seek, 0.99);

        assert_eq!(det.status, DeterminationStatus::Declined);
        assert_eq!(det.classes.len(), 2);
    }

    #[test]
    fn a_rung_that_reaches_nothing_is_excluded_not_forced() {
        let mut rt = Runtime::new();
        rt.emit("dub", Value::reading("spectral", 1.0, 0.1, "", "t").unwrap(), None);

        let seek = seek_with(&[("spectral", 0.6), ("nonexistent_rung", 0.5)]);
        let det = seek_to_closure(&rt, &seek, 0.99);

        assert_eq!(det.status, DeterminationStatus::Converged);
        assert_eq!(det.classes, vec!["dub".to_string()]);
    }

    #[test]
    fn a_rung_name_that_is_a_substring_of_an_unrelated_node_does_not_falsely_reach_it() {
        // Regression test for the gap this phase closes: the old
        // `reached_classes` used `tau.contains(&rung.name)`, so a rung named
        // "spectral" would wrongly match a node "render.non_spectral_thing".
        let mut rt = Runtime::new();
        rt.emit(
            "render.non_spectral_thing",
            Value::reading("level", 1.0, 0.1, "", "t").unwrap(),
            None,
        );
        rt.emit(
            "downstream",
            Value::reading("x", 1.0, 0.1, "", "t").unwrap(),
            Some("render.non_spectral_thing"),
        );

        assert_eq!(class_reached(&rt, "spectral"), None);
    }
}
