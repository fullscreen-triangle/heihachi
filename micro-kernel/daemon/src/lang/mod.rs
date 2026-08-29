//! The two language front ends.
//!
//! `mishima` (`.mma`) computes over the accumulated record; `sangoma`
//! (`.sgn`) constructs new material against declared targets. They share a
//! lexer, a diagnostic shape, and the rule that no literal of zero
//! resolution is writable.

pub mod lexer;
pub mod mishima;
pub mod sangoma;

use serde::{Deserialize, Serialize};

/// A refusal or a warning, from either language, in one shape.
///
/// `remedy` is never empty. A refusal with no way forward is a dead end,
/// and the checkers emit none.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    /// `error` stops the program from running; `warning` does not.
    pub severity: Severity,
    /// The rule that fired, e.g. `rule:mandatory-not`.
    pub rule: String,
    /// What is wrong.
    pub message: String,
    /// What to do about it.
    pub remedy: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
}

impl Diagnostic {
    pub fn error(
        rule: impl Into<String>,
        message: impl Into<String>,
        remedy: impl Into<String>,
        line: usize,
    ) -> Self {
        Self {
            severity: Severity::Error,
            rule: rule.into(),
            message: message.into(),
            remedy: remedy.into(),
            line,
            column: 1,
        }
    }

    pub fn warning(
        rule: impl Into<String>,
        message: impl Into<String>,
        remedy: impl Into<String>,
        line: usize,
    ) -> Self {
        Self {
            severity: Severity::Warning,
            rule: rule.into(),
            message: message.into(),
            remedy: remedy.into(),
            line,
            column: 1,
        }
    }
}

/// The result of checking a program. Never a thrown error: a refusal is
/// data the interface can render next to the source line it concerns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Whether the program would run. False if any diagnostic is an error.
    pub accepted: bool,
    pub diagnostics: Vec<Diagnostic>,
    pub declarations: Vec<Declaration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Declaration {
    pub kind: String,
    pub name: String,
    /// Parameters the writer did not set, and so should review first.
    pub defaulted: Vec<String>,
}

impl CheckResult {
    pub fn from_diagnostics(
        diagnostics: Vec<Diagnostic>,
        declarations: Vec<Declaration>,
    ) -> Self {
        let accepted = !diagnostics.iter().any(|d| d.severity == Severity::Error);
        Self { accepted, diagnostics, declarations }
    }
}

/// Composite power of rungs in series: `1 - prod(1 - k_i)`.
pub fn composite_power(powers: impl IntoIterator<Item = f64>) -> f64 {
    let remaining: f64 = powers.into_iter().map(|k| 1.0 - k).product();
    1.0 - remaining
}

/// Depth needed to reach a target composite with the strongest rung.
///
/// Returns `None` when the target is unreachable at any finite depth.
pub fn rungs_required(target: f64, best: f64) -> Option<usize> {
    if !(0.0 < best && best < 1.0) {
        return None;
    }
    if target >= 1.0 {
        return None;
    }
    if target <= 0.0 {
        return Some(0);
    }
    Some(((1.0 - target).ln() / (1.0 - best).ln()).ceil() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composition_matches_the_direct_product() {
        let got = composite_power([0.45, 0.30, 0.55]);
        let want = 1.0 - (1.0 - 0.45) * (1.0 - 0.30) * (1.0 - 0.55);
        assert!((got - want).abs() < 1e-12);
        assert!((got - 0.82675).abs() < 1e-9);
    }

    #[test]
    fn depth_agrees_with_incrementing_until_the_target_clears() {
        for target in [0.5, 0.7, 0.9, 0.95, 0.99] {
            for power in [0.15, 0.25, 0.35, 0.5] {
                let closed = rungs_required(target, power).unwrap();
                let mut n = 0usize;
                let mut comp = 0.0;
                while comp < target && n < 10_000 {
                    comp = composite_power([comp, power]);
                    n += 1;
                }
                assert_eq!(closed, n, "target {target}, power {power}");
            }
        }
    }
}
