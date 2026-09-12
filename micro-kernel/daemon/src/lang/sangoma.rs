//! sangoma (`.sgn`): constructing material against declared targets.
//!
//! A program declares what the finished sound must satisfy and the
//! resolution at which each requirement is asserted. The checker answers a
//! question a procedure never poses: is this reachable at this resolution?
//! If not, the refusal names the depth that would be required.
//!
//! The rules enforced here are physical -- headroom, latency, resolution,
//! reachability. Everything above them is composable and unjudged: a
//! compiler that refused on taste would enforce its author's present
//! opinions and obstruct the misuse that makes new sounds.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::{composite_power, rungs_required, CheckResult, Declaration, Diagnostic};
use super::lexer::{lex, TokKind, Token};

pub const KEYWORDS: &[&str] = &[
    "floor", "medium", "species", "source", "motion", "construct", "stage",
    "chain", "assemble", "from", "target", "via", "rung", "at", "observe",
    "assert", "emit", "let", "as", "ceiling", "gain", "latency",
];

pub fn keywords() -> HashSet<&'static str> {
    KEYWORDS.iter().copied().collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetItem {
    pub name: String,
    pub relation: String,
    pub magnitude: f64,
    pub floor: f64,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    pub name: String,
    /// Declared gain in dB, when stated.
    pub gain_db: Option<f64>,
    /// Declared latency in samples, when stated.
    pub latency: Option<u32>,
    /// The CLAP plugin id this stage renders through, when bound. A stage
    /// with no binding is reachability-checked but not renderable.
    pub clap_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rung {
    pub name: String,
    pub power: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Construct {
    pub name: String,
    pub stages: Vec<Stage>,
    pub targets: Vec<TargetItem>,
    pub ladder: Vec<Rung>,
    pub line: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Program {
    pub ambient_floor: Option<f64>,
    pub species: HashMap<String, HashMap<String, f64>>,
    /// Ceiling declared on the medium, in dBFS.
    pub ceiling_db: Option<f64>,
    pub constructs: Vec<Construct>,
}

struct Parser {
    toks: Vec<Token>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.toks[self.i]
    }
    fn next(&mut self) -> Token {
        let t = self.toks[self.i].clone();
        if self.i + 1 < self.toks.len() {
            self.i += 1;
        }
        t
    }
    fn accept(&mut self, text: &str) -> bool {
        if self.peek().text == text {
            self.next();
            true
        } else {
            false
        }
    }
    fn expect(&mut self, text: &str) -> Result<Token, Diagnostic> {
        if self.peek().text == text {
            Ok(self.next())
        } else {
            let t = self.peek().clone();
            Err(Diagnostic::error(
                "syntax",
                format!("expected {:?}, found {:?}", text, t.text),
                format!("insert {text:?}"),
                t.line,
            ))
        }
    }
}

pub fn parse(source: &str) -> Result<Program, Diagnostic> {
    let toks = lex(source, &keywords()).map_err(|e| {
        Diagnostic::error(
            "rule:floor-positivity",
            e.message.clone(),
            "state a strictly positive resolution",
            e.line,
        )
    })?;
    let mut p = Parser { toks, i: 0 };
    let mut prog = Program::default();

    while p.peek().kind != TokKind::Eof {
        let t = p.peek().clone();
        match t.text.as_str() {
            "floor" => {
                p.next();
                let num = p.next();
                let v = num.value.unwrap_or(0.0);
                if !(v > 0.0) {
                    return Err(Diagnostic::error(
                        "rule:floor-positivity",
                        "the ambient floor must be strictly positive",
                        "declare the resolution your render path delivers",
                        t.line,
                    ));
                }
                prog.ambient_floor = Some(v);
            }
            "species" | "medium" => {
                let is_medium = t.text == "medium";
                p.next();
                let name = p.next().text;
                let mut fields = HashMap::new();
                p.expect("{")?;
                let mut depth = 1;
                while depth > 0 && p.peek().kind != TokKind::Eof {
                    let tk = p.next();
                    match tk.text.as_str() {
                        "{" => depth += 1,
                        "}" => depth -= 1,
                        _ => {
                            if p.peek().text == ":" {
                                p.next();
                                let val = p.next();
                                if let Some(v) = val.value {
                                    fields.insert(tk.text.clone(), v);
                                    if is_medium && tk.text == "ceiling" {
                                        prog.ceiling_db = Some(v);
                                    }
                                }
                            }
                        }
                    }
                }
                prog.species.insert(name, fields);
            }
            "construct" => prog.constructs.push(parse_construct(&mut p)?),
            _ => {
                p.next();
            }
        }
    }
    Ok(prog)
}

fn parse_construct(p: &mut Parser) -> Result<Construct, Diagnostic> {
    let line = p.peek().line;
    p.expect("construct")?;
    let name = p.next().text;
    p.expect("{")?;
    let mut stages = Vec::new();
    let mut targets = Vec::new();
    let mut ladder = Vec::new();

    while !p.accept("}") {
        if p.peek().kind == TokKind::Eof {
            return Err(Diagnostic::error(
                "syntax",
                format!("unterminated construct {name:?}"),
                "close the construct with `}`",
                line,
            ));
        }
        let t = p.peek().clone();
        match t.text.as_str() {
            "stage" => {
                p.next();
                let sname = p.next().text;
                let mut gain_db = None;
                let mut latency = None;
                let mut clap_id = None;
                // optional trailing declarations: gain <db>, latency <samples>, clap <plugin-id>
                loop {
                    if p.accept("gain") {
                        gain_db = p.next().value;
                    } else if p.accept("latency") {
                        latency = p.next().value.map(|v| v as u32);
                    } else if p.accept("clap") {
                        clap_id = Some(p.next().text);
                    } else {
                        break;
                    }
                }
                stages.push(Stage { name: sname, gain_db, latency, clap_id });
            }
            "target" => {
                p.next();
                p.expect("{")?;
                while !p.accept("}") {
                    if p.peek().kind == TokKind::Eof {
                        return Err(Diagnostic::error(
                            "syntax",
                            "unterminated target block",
                            "close the block with `}`",
                            line,
                        ));
                    }
                    let item = p.next();
                    let rel = p.next().text;
                    let num = p.next();
                    if num.kind != TokKind::Num {
                        return Err(Diagnostic::error(
                            "syntax",
                            "a target needs a magnitude",
                            "write e.g. crest >= 6.0#0.5",
                            item.line,
                        ));
                    }
                    let Some(floor) = num.floor else {
                        return Err(Diagnostic::error(
                            "rule:floor-on-target",
                            format!("target `{}` states no resolution", item.text),
                            "annotate the magnitude, e.g. 6.0#0.5, so the claim is \
                             checkable",
                            item.line,
                        ));
                    };
                    targets.push(TargetItem {
                        name: item.text,
                        relation: rel,
                        magnitude: num.value.unwrap_or(0.0),
                        floor,
                        line: item.line,
                    });
                }
            }
            "via" => {
                p.next();
                p.expect("{")?;
                while !p.accept("}") {
                    if p.peek().kind == TokKind::Eof {
                        return Err(Diagnostic::error(
                            "syntax",
                            "unterminated via clause",
                            "close the clause with `}`",
                            line,
                        ));
                    }
                    if p.accept("rung") {
                        let rname = p.next().text;
                        let mut power = 0.0;
                        if p.accept("at") {
                            power = p.next().value.unwrap_or(0.0);
                        }
                        ladder.push(Rung { name: rname, power });
                    } else {
                        p.next();
                    }
                }
            }
            _ => {
                p.next();
            }
        }
    }
    Ok(Construct { name, stages, targets, ladder, line })
}

/// Peak level after a chain of declared gains, and whether it clears.
pub fn headroom(gains_db: &[f64], ceiling_db: f64) -> (f64, bool) {
    let peak: f64 = gains_db.iter().sum();
    (peak, peak <= ceiling_db)
}

/// Detectability: a stage is observable within an assembly iff tau < 1.
pub fn detectability(thickness: f64, extent: f64) -> f64 {
    if extent <= 0.0 {
        f64::INFINITY
    } else {
        thickness / extent
    }
}

pub fn check(
    prog: &Program,
    required_power: f64,
    backend_resolution: f64,
) -> CheckResult {
    let mut diags = Vec::new();
    let mut decls = Vec::new();

    match prog.ambient_floor {
        None => diags.push(Diagnostic::error(
            "rule:floor-declared",
            "no ambient floor was declared",
            "add a `floor` declaration",
            1,
        )),
        Some(floor) => {
            if backend_resolution > floor {
                diags.push(Diagnostic::error(
                    "rule:floor-negotiation",
                    format!(
                        "declared floor {floor} is finer than the render path can \
                         resolve ({backend_resolution})"
                    ),
                    "coarsen the floor, or render at higher resolution",
                    1,
                ));
            }
        }
    }

    for c in &prog.constructs {
        decls.push(Declaration {
            kind: "construct".into(),
            name: c.name.clone(),
            defaulted: if c.ladder.is_empty() {
                vec!["via".into()]
            } else {
                Vec::new()
            },
        });

        if c.targets.is_empty() {
            diags.push(Diagnostic::error(
                "rule:target-required",
                format!("construct `{}` declares no target", c.name),
                "state what the finished sound must satisfy",
                c.line,
            ));
        }

        // A target may not claim a resolution finer than the instruments
        // support: that would assert a distinction nothing delivers.
        if let Some(floor) = prog.ambient_floor {
            for t in &c.targets {
                if t.floor < floor {
                    diags.push(Diagnostic::error(
                        "rule:over-claim",
                        format!(
                            "target `{}` claims resolution {}, finer than the \
                             ambient floor {}",
                            t.name, t.floor, floor
                        ),
                        "coarsen the target, or declare a finer floor your tools support",
                        t.line,
                    ));
                }
            }
        }

        // Reachability, with the required depth named on failure.
        if !c.ladder.is_empty() {
            let attainable = composite_power(c.ladder.iter().map(|r| r.power));
            if attainable < required_power {
                let best = c.ladder.iter().map(|r| r.power).fold(0.0_f64, f64::max);
                let remedy = match rungs_required(required_power, best) {
                    Some(n) => format!(
                        "{n} rungs at the strongest available power ({best}) would \
                         reach it"
                    ),
                    None => "no finite depth reaches this target with these stages"
                        .to_string(),
                };
                diags.push(Diagnostic::error(
                    "rule:reachability",
                    format!(
                        "construct `{}` attains {attainable:.4} of a required \
                         {required_power}",
                        c.name
                    ),
                    remedy,
                    c.line,
                ));
            }
        }

        // Headroom: a chain whose declared gains breach the ceiling clips.
        let gains: Vec<f64> = c.stages.iter().filter_map(|s| s.gain_db).collect();
        if !gains.is_empty() {
            let ceiling = prog.ceiling_db.unwrap_or(-1.0);
            let (peak, clears) = headroom(&gains, ceiling);
            if !clears {
                diags.push(Diagnostic::error(
                    "rule:headroom",
                    format!(
                        "declared gains sum to {peak:+.2} dB against a ceiling of \
                         {ceiling:+.2} dB"
                    ),
                    format!(
                        "reduce total gain by at least {:.2} dB, or raise the ceiling",
                        peak - ceiling
                    ),
                    c.line,
                ));
            }
        }

        // Latency is reported rather than refused: it must be compensated,
        // not avoided.
        let latency: u32 = c.stages.iter().filter_map(|s| s.latency).sum();
        if latency > 0 {
            diags.push(Diagnostic::warning(
                "rule:latency",
                format!("chain introduces {latency} samples of latency"),
                "compensate downstream, or account for it in the arrangement",
                c.line,
            ));
        }
    }

    CheckResult::from_diagnostics(diags, decls)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"
floor 0.02
medium air { ceiling: -1.0 }
species reese {
  source { operators: 2, ratio: 1.0, index: 3.5 }
  motion { rate: 0.3, depth: 0.8 }
}
construct bass {
  stage fm_source
  stage resample
  stage saturate
  target {
    crest    >= 6.0#0.5
    midrange >= 0.40#0.05
    width    <= 0.85#0.02
  }
  via { rung fm_source at 0.40
     >> rung resample  at 0.35
     >> rung saturate  at 0.55 }
}
"#;

    #[test]
    fn worked_example_parses_with_the_expected_power() {
        let prog = parse(GOOD).unwrap();
        assert_eq!(prog.ambient_floor, Some(0.02));
        assert_eq!(prog.ceiling_db, Some(-1.0));
        let c = &prog.constructs[0];
        assert_eq!(c.stages.len(), 3);
        assert_eq!(c.targets.len(), 3);
        let power = composite_power(c.ladder.iter().map(|r| r.power));
        assert!((power - 0.8245).abs() < 1e-9);
        assert!(check(&prog, 0.8, 0.001).accepted);
    }

    #[test]
    fn target_without_resolution_is_refused() {
        let err = parse("floor 0.02\nconstruct b { target { crest >= 6.0 } }")
            .unwrap_err();
        assert_eq!(err.rule, "rule:floor-on-target");
        assert!(!err.remedy.is_empty());
    }

    #[test]
    fn over_claiming_resolution_is_refused() {
        let prog =
            parse("floor 0.05\nconstruct b { target { crest >= 6.0#0.001 } }").unwrap();
        let res = check(&prog, 0.8, 0.001);
        assert!(res.diagnostics.iter().any(|d| d.rule == "rule:over-claim"));
    }

    #[test]
    fn unreachable_target_names_the_required_depth() {
        let prog = parse(
            "floor 0.02\nconstruct b { target { crest >= 6.0#0.5 } \
             via { rung a at 0.2 >> rung b at 0.2 } }",
        )
        .unwrap();
        let res = check(&prog, 0.9, 0.001);
        let d = res
            .diagnostics
            .iter()
            .find(|d| d.rule == "rule:reachability")
            .expect("reachability diagnostic");
        assert!(d.remedy.contains("rungs"), "remedy was {:?}", d.remedy);
    }

    #[test]
    fn a_chain_that_would_clip_is_refused() {
        let prog = parse(
            "floor 0.02\nmedium air { ceiling: -1.0 }\n\
             construct b { stage a gain 3.0 stage c gain 2.0 \
             target { crest >= 6.0#0.5 } }",
        )
        .unwrap();
        let res = check(&prog, 0.0, 0.001);
        assert!(res.diagnostics.iter().any(|d| d.rule == "rule:headroom"));
        assert!(!res.accepted);
    }

    #[test]
    fn detectability_boundary_sits_at_one() {
        assert!(detectability(0.1, 1.0) < 1.0);
        assert!(detectability(0.5, 1.0) < 1.0);
        assert!(detectability(1.0, 1.0) >= 1.0);
        assert!(detectability(2.0, 1.0) >= 1.0);
    }

    #[test]
    fn every_diagnostic_carries_a_remedy() {
        let sources = [
            "floor 0.02\nconstruct b { stage a }",
            "floor 0.05\nconstruct b { target { crest >= 6.0#0.001 } }",
            "floor 0.02\nconstruct b { target { crest >= 6.0#0.5 } via { rung a at 0.1 } }",
        ];
        let mut total = 0;
        for src in sources {
            let prog = parse(src).unwrap();
            for d in check(&prog, 0.8, 0.001).diagnostics {
                total += 1;
                assert!(!d.remedy.trim().is_empty(), "{} had no remedy", d.rule);
            }
        }
        assert!(total > 0);
    }
}
