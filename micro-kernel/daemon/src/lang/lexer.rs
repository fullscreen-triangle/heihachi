//! One lexer for both languages; they differ only in their keywords.
//!
//! A numeric literal may carry its resolution by the suffix `value#floor`.
//! A non-positive floor suffix is rejected here, not later: there is no way
//! to write a literal of zero resolution, because that would be a claim of
//! exact measurement.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TokKind {
    Ident,
    Keyword,
    Num,
    Str,
    Op,
    Eof,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub kind: TokKind,
    pub text: String,
    pub line: usize,
    pub column: usize,
    pub value: Option<f64>,
    /// The resolution suffix, when the literal carried one.
    pub floor: Option<f64>,
}

#[derive(Debug, thiserror::Error)]
#[error("line {line}: {message}")]
pub struct LexError {
    pub line: usize,
    pub message: String,
}

const OPS: &[&str] = &[
    ">>", ":=", "<=", ">=", "==", "!=", "||", "{", "}", "(", ")", "[", "]", ",",
    ":", ".", "<", ">", "#",
];

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Scan a number starting at `i`, returning (text, value, end).
fn scan_number(src: &[char], i: usize) -> Option<(String, f64, usize)> {
    let mut j = i;
    if j < src.len() && src[j] == '-' {
        j += 1;
    }
    let digits_start = j;
    while j < src.len() && src[j].is_ascii_digit() {
        j += 1;
    }
    if j == digits_start {
        return None;
    }
    if j < src.len() && src[j] == '.' && j + 1 < src.len() && src[j + 1].is_ascii_digit() {
        j += 1;
        while j < src.len() && src[j].is_ascii_digit() {
            j += 1;
        }
    }
    if j < src.len() && (src[j] == 'e' || src[j] == 'E') {
        let mut k = j + 1;
        if k < src.len() && (src[k] == '+' || src[k] == '-') {
            k += 1;
        }
        if k < src.len() && src[k].is_ascii_digit() {
            while k < src.len() && src[k].is_ascii_digit() {
                k += 1;
            }
            j = k;
        }
    }
    let text: String = src[i..j].iter().collect();
    text.parse().ok().map(|v| (text, v, j))
}

pub fn lex(source: &str, keywords: &HashSet<&str>) -> Result<Vec<Token>, LexError> {
    let src: Vec<char> = source.chars().collect();
    let mut toks = Vec::new();
    let (mut i, mut line, mut col) = (0usize, 1usize, 1usize);

    while i < src.len() {
        let c = src[i];
        if c == '\n' {
            line += 1;
            col = 1;
            i += 1;
            continue;
        }
        if c == ' ' || c == '\t' || c == '\r' {
            i += 1;
            col += 1;
            continue;
        }
        // comments run to end of line
        if c == '-' && i + 1 < src.len() && src[i + 1] == '-' {
            while i < src.len() && src[i] != '\n' {
                i += 1;
            }
            continue;
        }
        // a number, possibly negative, possibly with a resolution suffix
        if c.is_ascii_digit()
            || (c == '-' && i + 1 < src.len() && src[i + 1].is_ascii_digit())
        {
            if let Some((mut text, value, mut j)) = scan_number(&src, i) {
                let mut floor = None;
                if j < src.len() && src[j] == '#' {
                    match scan_number(&src, j + 1) {
                        Some((ftext, fvalue, fend)) => {
                            if !(fvalue > 0.0) {
                                return Err(LexError {
                                    line,
                                    message: format!(
                                        "floor suffix must be strictly positive, got {ftext}: \
                                         a claim of exact measurement is not writable"
                                    ),
                                });
                            }
                            floor = Some(fvalue);
                            text.push('#');
                            text.push_str(&ftext);
                            j = fend;
                        }
                        None => {
                            return Err(LexError {
                                line,
                                message: "'#' must be followed by a resolution".into(),
                            })
                        }
                    }
                }
                toks.push(Token {
                    kind: TokKind::Num,
                    text,
                    line,
                    column: col,
                    value: Some(value),
                    floor,
                });
                col += j - i;
                i = j;
                continue;
            }
        }
        if is_ident_start(c) {
            let start = i;
            while i < src.len() && is_ident_continue(src[i]) {
                i += 1;
            }
            let text: String = src[start..i].iter().collect();
            let kind = if keywords.contains(text.as_str()) {
                TokKind::Keyword
            } else {
                TokKind::Ident
            };
            toks.push(Token { kind, text, line, column: col, value: None, floor: None });
            col += i - start;
            continue;
        }
        if c == '"' {
            let start = i + 1;
            let mut j = start;
            while j < src.len() && src[j] != '"' {
                j += 1;
            }
            if j >= src.len() {
                return Err(LexError { line, message: "unterminated string".into() });
            }
            let text: String = src[start..j].iter().collect();
            toks.push(Token {
                kind: TokKind::Str,
                text,
                line,
                column: col,
                value: None,
                floor: None,
            });
            col += j - i + 1;
            i = j + 1;
            continue;
        }
        let rest: String = src[i..].iter().take(2).collect();
        let matched = OPS.iter().find(|op| rest.starts_with(**op));
        match matched {
            Some(op) => {
                toks.push(Token {
                    kind: TokKind::Op,
                    text: (*op).to_string(),
                    line,
                    column: col,
                    value: None,
                    floor: None,
                });
                i += op.len();
                col += op.len();
            }
            None => {
                return Err(LexError {
                    line,
                    message: format!("unexpected character {c:?}"),
                })
            }
        }
    }

    toks.push(Token {
        kind: TokKind::Eof,
        text: String::new(),
        line,
        column: col,
        value: None,
        floor: None,
    });
    Ok(toks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kw() -> HashSet<&'static str> {
        HashSet::from(["floor", "seek"])
    }

    #[test]
    fn zero_resolution_literal_is_refused() {
        for bad in ["floor 1.0#0", "floor 1.0#0.0", "floor 2.0#-1"] {
            assert!(lex(bad, &kw()).is_err(), "{bad} should be refused");
        }
        assert!(lex("floor 0.02", &kw()).is_ok());
        assert!(lex("floor 1.0#0.001", &kw()).is_ok());
    }

    #[test]
    fn resolution_suffix_is_carried() {
        let toks = lex("6.0#0.5", &kw()).unwrap();
        assert_eq!(toks[0].value, Some(6.0));
        assert_eq!(toks[0].floor, Some(0.5));
    }

    #[test]
    fn comments_and_negatives_coexist() {
        let toks = lex("-1.0 -- a comment\nfloor", &kw()).unwrap();
        assert_eq!(toks[0].value, Some(-1.0));
        assert_eq!(toks[1].kind, TokKind::Keyword);
    }
}
