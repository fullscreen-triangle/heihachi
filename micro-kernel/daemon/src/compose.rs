//! Turning a request in English into source the producer can read.
//!
//! The model writes; it never answers. What comes back is source for review
//! in the editor, and it is deliberately not run: a program the producer has
//! not read is a program nobody authored, and the whole point of the record
//! is that its commitments were made by someone.
//!
//! The composed source is checked before it is returned, so a model that
//! produced something ill-formed is caught here rather than at the point the
//! producer presses run.

use serde::Serialize;

use crate::integrations::ollama::Ollama;
use crate::lang::{mishima, sangoma, CheckResult};

#[derive(Debug, Clone, Serialize)]
pub struct Composed {
    pub source: String,
    pub check: CheckResult,
    /// Set when the model could not be reached or produced nothing usable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

const MISHIMA_PROMPT: &str = r#"Write one mishima program. Output ONLY the program, no prose, no code fences.

Grammar, exactly:

floor <positive number>

seek <identifier>
  not    { <thing>, <thing>, <thing> }
  toward { region(<identifier>) }
  via    { rung <name> at <0..1>
        >> rung <name> at <0..1>
        >> rung <name> at <0..1> }
  until  closure
  otherwise decline
  yield  <identifier>

Hard rules:
- The `not` clause is MANDATORY and must name at least one excluded property.
  A search that does not say what it excludes is refused by the parser.
- Use at least THREE rungs of differing kind, or the checker refuses the
  program: fewer than three cannot survive the loss of one.
- Rung names should be drawn from: spectral, annotation, model, transient,
  loudness, stereo, chain.
- Powers are between 0 and 1 and should differ from one another.

The request:
"#;

const SANGOMA_PROMPT: &str = r#"Write one sangoma program. Output ONLY the program, no prose, no code fences.

Grammar, exactly:

floor <positive number>

medium air { ceiling: -1.0 }

species <name> {
  source { <field>: <number>, <field>: <number> }
  motion { <field>: <number>, <field>: <number> }
}

construct <name> {
  stage <name>
  stage <name>
  stage <name>
  target {
    <property> >= <number>#<resolution>
    <property> <= <number>#<resolution>
  }
  via { rung <name> at <0..1>
     >> rung <name> at <0..1>
     >> rung <name> at <0..1> }
}

Hard rules:
- EVERY target magnitude MUST carry a resolution after `#`, e.g. 6.0#0.5.
  A target with no resolution is refused by the parser.
- A resolution finer than the ambient floor is refused as an over-claim, so
  keep resolutions at or above the declared floor.
- Target properties should be drawn from: crest, midrange, width, peak, rms,
  centroid, low_ratio.
- Use at least three rungs with differing powers.

The request:
"#;

pub async fn compose(ollama: &Ollama, language: &str, request: &str) -> Composed {
    let is_sangoma = language == "sangoma";
    let prompt = if is_sangoma { SANGOMA_PROMPT } else { MISHIMA_PROMPT };
    let raw = ollama.write_source(&format!("{prompt}{request}\n")).await;

    let (source, note) = match raw {
        Ok(text) => (strip_fences(&text), None),
        Err(e) => (
            fallback(is_sangoma, request),
            Some(format!(
                "model unavailable ({e}); this is a skeleton to edit, not a composition"
            )),
        ),
    };

    let check = if is_sangoma {
        match sangoma::parse(&source) {
            Ok(prog) => sangoma::check(&prog, 0.8, 0.001),
            Err(d) => CheckResult::from_diagnostics(vec![d], Vec::new()),
        }
    } else {
        match mishima::parse(&source) {
            Ok(prog) => mishima::check(&prog, 0.001),
            Err(d) => CheckResult::from_diagnostics(vec![d], Vec::new()),
        }
    };

    Composed { source, check, note }
}

/// Models like to wrap output in fences whatever the instruction says.
fn strip_fences(raw: &str) -> String {
    let trimmed = raw.trim();
    let without = trimmed
        .strip_prefix("```")
        .map(|rest| {
            // drop an optional language tag on the opening fence
            let rest = rest.split_once('\n').map(|(_, r)| r).unwrap_or(rest);
            rest.strip_suffix("```").unwrap_or(rest)
        })
        .unwrap_or(trimmed);
    without.trim().to_string()
}

/// What to hand back when the model is not available.
///
/// A skeleton the producer edits is more useful than an error, and it is
/// labelled as a skeleton so it is not mistaken for a composition.
fn fallback(is_sangoma: bool, request: &str) -> String {
    let slug: String = request
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .split('_')
        .filter(|s| !s.is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join("_");
    let name = if slug.is_empty() { "untitled".into() } else { slug };

    if is_sangoma {
        format!(
            "-- {name}.sgn -- skeleton: the model was unavailable\nfloor 0.02\n\n\
             medium air {{ ceiling: -1.0 }}\n\n\
             construct {name} {{\n  stage source\n  stage shape\n  stage finish\n\n  \
             target {{\n    crest    >= 6.0#0.5\n    midrange >= 0.40#0.05\n  }}\n\n  \
             via {{ rung source at 0.40\n     >> rung shape  at 0.35\n     \
             >> rung finish at 0.55 }}\n}}\n"
        )
    } else {
        format!(
            "-- {name}.mma -- skeleton: the model was unavailable\nfloor 0.02\n\n\
             seek {name}\n  not    {{ thin, undistorted, mono }}\n  \
             toward {{ region({name}) }}\n  \
             via    {{ rung spectral   at 0.45\n        \
             >> rung annotation at 0.30\n        \
             >> rung model      at 0.55 }}\n  \
             until  closure\n  otherwise decline\n  yield  found\n"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fences_are_stripped() {
        assert_eq!(strip_fences("```mishima\nfloor 0.02\n```"), "floor 0.02");
        assert_eq!(strip_fences("```\nfloor 0.02\n```"), "floor 0.02");
        assert_eq!(strip_fences("  floor 0.02  "), "floor 0.02");
    }

    #[test]
    fn the_mishima_fallback_is_a_program_that_checks() {
        let source = fallback(false, "that growl from 2019");
        let prog = mishima::parse(&source).expect("fallback must parse");
        let result = mishima::check(&prog, 0.001);
        assert!(result.accepted, "{:?}", result.diagnostics);
    }

    #[test]
    fn the_sangoma_fallback_is_a_program_that_checks() {
        let source = fallback(true, "a heavy reese bass");
        let prog = sangoma::parse(&source).expect("fallback must parse");
        let result = sangoma::check(&prog, 0.8, 0.001);
        assert!(result.accepted, "{:?}", result.diagnostics);
    }

    #[test]
    fn the_fallback_names_itself_a_skeleton() {
        assert!(fallback(false, "x").contains("skeleton"));
        assert!(fallback(true, "x").contains("skeleton"));
    }
}
