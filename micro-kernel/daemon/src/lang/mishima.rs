//! mishima (`.mma`): computing over the accumulated record.
//!
//! The central syntactic rule is that a `seek` must state what it excludes.
//! A region individuated against nothing is not a region, so the absence of
//! a `not` clause is a parse error rather than a type error: the producer
//! meets it while writing rather than after running.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::{composite_power, rungs_required, CheckResult, Declaration, Diagnostic};
use super::lexer::{lex, TokKind, Token};

pub const KEYWORDS: &[&str] = &[
    "floor", "module", "probe", "commit", "seek", "not", "toward", "via",
    "until", "closure", "converge", "otherwise", "decline", "yield", "ladder",
    "rung", "at", "observe", "assert", "emit", "let", "as", "when",
];

pub fn keywords() -> HashSet<&'static str> {
    KEYWORDS.iter().copied().collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rung {
    pub name: String,
    pub power: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Seek {
    pub subject: String,
    pub exclusions: Vec<String>,
    pub toward: String,
    pub ladder: Vec<Rung>,
    pub admit: String,
    pub otherwise_decline: bool,
    pub result: String,
    pub line: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Program {
    pub ambient_floor: Option<f64>,
    pub seeks: Vec<Seek>,
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
                format!(
                    "expected {:?}, found {:?}",
                    text,
                    if t.text.is_empty() { "end of input" } else { &t.text }
                ),
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
                if num.kind != TokKind::Num {
                    return Err(Diagnostic::error(
                        "syntax",
                        "floor needs a number",
                        "write e.g. floor 0.02",
                        t.line,
                    ));
                }
                let v = num.value.unwrap_or(0.0);
                if !(v > 0.0) {
                    return Err(Diagnostic::error(
                        "rule:floor-positivity",
                        "the ambient floor must be strictly positive",
                        "declare the resolution your instruments deliver",
                        t.line,
                    ));
                }
                prog.ambient_floor = Some(v);
            }
            "module" => {
                p.next();
                p.next();
                p.expect("{")?;
                let mut depth = 1;
                while depth > 0 && p.peek().kind != TokKind::Eof {
                    let tk = p.next();
                    match tk.text.as_str() {
                        "{" => depth += 1,
                        "}" => depth -= 1,
                        _ => {}
                    }
                }
            }
            "seek" => prog.seeks.push(parse_seek(&mut p)?),
            _ => {
                p.next();
            }
        }
    }
    Ok(prog)
}

fn parse_seek(p: &mut Parser) -> Result<Seek, Diagnostic> {
    let line = p.peek().line;
    p.expect("seek")?;
    let subject = p.next().text;

    // The `not` clause is mandatory. Its absence is refused here, at parse
    // time, because a search that does not say what it excludes has not
    // specified a region at all.
    if p.peek().text != "not" {
        return Err(Diagnostic::error(
            "rule:mandatory-not",
            "a seek without a `not` clause does not specify a region",
            "state what the search excludes, e.g. not { thin, undistorted }",
            p.peek().line,
        ));
    }
    p.next();
    p.expect("{")?;
    let mut exclusions = Vec::new();
    while !p.accept("}") {
        if p.peek().kind == TokKind::Eof {
            return Err(Diagnostic::error(
                "syntax",
                "unterminated exclusion list",
                "close the list with `}`",
                line,
            ));
        }
        let tok = p.next();
        if tok.text != "," {
            exclusions.push(tok.text);
        }
    }
    if exclusions.is_empty() {
        return Err(Diagnostic::error(
            "rule:mandatory-not",
            "the exclusion list is empty",
            "name at least one region the target is not",
            line,
        ));
    }

    p.expect("toward")?;
    p.expect("{")?;
    let mut toward_parts = Vec::new();
    while !p.accept("}") {
        if p.peek().kind == TokKind::Eof {
            return Err(Diagnostic::error(
                "syntax",
                "unterminated toward clause",
                "close the clause with `}`",
                line,
            ));
        }
        toward_parts.push(p.next().text);
    }

    let mut ladder = Vec::new();
    if p.accept("via") {
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
                let name = p.next().text;
                let mut power = 0.0;
                if p.accept("at") {
                    power = p.next().value.unwrap_or(0.0);
                }
                ladder.push(Rung { name, power });
            } else {
                p.next();
            }
        }
    }

    p.expect("until")?;
    let admit = p.next().text;
    let otherwise_decline = if p.accept("otherwise") {
        p.expect("decline")?;
        true
    } else {
        false
    };
    p.expect("yield")?;
    let result = p.next().text;

    Ok(Seek {
        subject,
        exclusions,
        toward: toward_parts.join(" "),
        ladder,
        admit,
        otherwise_decline,
        result,
        line,
    })
}

/// Static checks. Every diagnostic carries a remedy.
pub fn check(prog: &Program, backend_resolution: f64) -> CheckResult {
    let mut diags = Vec::new();
    let mut decls = Vec::new();

    match prog.ambient_floor {
        None => diags.push(Diagnostic::error(
            "rule:floor-declared",
            "no ambient floor was declared",
            "add a `floor` declaration naming your resolution",
            1,
        )),
        Some(floor) => {
            if backend_resolution > floor {
                diags.push(Diagnostic::error(
                    "rule:floor-negotiation",
                    format!(
                        "declared floor {floor} is finer than the back end can \
                         resolve ({backend_resolution}); a reported distinction \
                         may be an artefact"
                    ),
                    "coarsen the declared floor, or analyse at higher resolution",
                    1,
                ));
            }
        }
    }

    for s in &prog.seeks {
        decls.push(Declaration {
            kind: "seek".into(),
            name: s.subject.clone(),
            defaulted: if s.ladder.is_empty() {
                vec!["via".into()]
            } else {
                Vec::new()
            },
        });

        // A ladder of fewer than three rungs cannot contain a support cycle
        // of length three, so it is not robust to the loss of any one.
        if s.admit == "closure" && !s.ladder.is_empty() && s.ladder.len() < 3 {
            diags.push(Diagnostic::error(
                "rule:coherence",
                format!(
                    "ladder of {} rung(s) cannot close: a support structure of \
                     fewer than three is not robust to the loss of one",
                    s.ladder.len()
                ),
                "add rungs of differing kind until at least three support the result",
                s.line,
            ));
        }

        if !s.ladder.is_empty() {
            let attainable = composite_power(s.ladder.iter().map(|r| r.power));
            if attainable < 0.5 {
                let best = s
                    .ladder
                    .iter()
                    .map(|r| r.power)
                    .fold(0.0_f64, f64::max);
                let needed = rungs_required(0.5, best)
                    .map(|n| format!("{n} of the strongest would reach 0.5"))
                    .unwrap_or_else(|| "no finite depth reaches 0.5".into());
                diags.push(Diagnostic::error(
                    "rule:saturation",
                    format!("ladder attains composite power {attainable:.4}"),
                    format!("add rungs; {needed}"),
                    s.line,
                ));
            }
        }

        if !s.otherwise_decline && s.admit == "closure" {
            diags.push(Diagnostic::warning(
                "rule:undiscriminated",
                "a closure that may contest has no declared alternative",
                "add `otherwise decline` and handle the contested outcome",
                s.line,
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
seek reese_growl
  not    { thin, undistorted, mono }
  toward { region(that_2019_growl) }
  via    { rung spectral   at 0.45
        >> rung annotation at 0.30
        >> rung model      at 0.55 }
  until  closure
  otherwise decline
  yield  found
"#;

    #[test]
    fn worked_example_parses_with_the_expected_power() {
        let prog = parse(GOOD).unwrap();
        assert_eq!(prog.ambient_floor, Some(0.02));
        let s = &prog.seeks[0];
        assert_eq!(s.exclusions.len(), 3);
        assert_eq!(s.ladder.len(), 3);
        let power = composite_power(s.ladder.iter().map(|r| r.power));
        assert!((power - 0.82675).abs() < 1e-9);
        assert!(check(&prog, 0.001).accepted);
    }

    #[test]
    fn seek_without_not_is_a_parse_error() {
        let err = parse("floor 0.02\nseek x toward { y } until closure yield f")
            .unwrap_err();
        assert_eq!(err.rule, "rule:mandatory-not");
        assert!(!err.remedy.is_empty());
    }

    #[test]
    fn empty_exclusion_list_is_refused() {
        let err = parse("floor 0.02\nseek x not { } toward { y } until closure yield f")
            .unwrap_err();
        assert_eq!(err.rule, "rule:mandatory-not");
    }

    #[test]
    fn short_ladders_are_refused_under_closure() {
        for n in 1..=3 {
            let rungs: Vec<String> =
                (0..n).map(|i| format!("rung r{i} at 0.4")).collect();
            let src = format!(
                "floor 0.02\nseek x not {{ y }} toward {{ z }} via {{ {} }} \
                 until closure otherwise decline yield f",
                rungs.join(" >> ")
            );
            let prog = parse(&src).unwrap();
            let fired: Vec<_> = check(&prog, 0.001)
                .diagnostics
                .into_iter()
                .map(|d| d.rule)
                .collect();
            if n < 3 {
                assert!(fired.contains(&"rule:coherence".to_string()), "n={n}");
            } else {
                assert!(!fired.contains(&"rule:coherence".to_string()), "n={n}");
            }
        }
    }

    #[test]
    fn a_floor_finer_than_the_back_end_is_refused() {
        let prog = parse(GOOD).unwrap();
        let coarse = check(&prog, 0.5);
        assert!(!coarse.accepted);
        assert!(coarse
            .diagnostics
            .iter()
            .any(|d| d.rule == "rule:floor-negotiation"));
        assert!(coarse.diagnostics.iter().all(|d| !d.remedy.is_empty()));
    }
}
