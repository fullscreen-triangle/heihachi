//! Ollama as a constructor rung.
//!
//! The model's job here is narrow and deliberately so: it draws
//! distinctions. Given the values on a node it returns the finite set of
//! distinctions the material draws, and nothing else. It does not rank,
//! score, or decide, and there is no path by which its output overrides
//! another rung -- a model that disagrees with the spectral rung produces a
//! contested closure naming the disagreement, which is the correct report.
//!
//! Two consequences follow from that confinement, and they are why it is
//! safe to point a small local model at this at all:
//!
//!   * a hallucinated distinction adds a contact to the graph, so the cut
//!     computed over it is still correct and the cell merely comes out
//!     coarser. Extraction error coarsens; it does not corrupt.
//!   * nothing downstream inspects which rung produced a distinction, so
//!     the model is interchangeable with a measurement of equal power.

use std::time::Duration;

use serde::{Deserialize, Serialize};

pub const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:11434";

/// What the model returns: the distinctions a source draws.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TermMap {
    pub distinctions: Vec<String>,
    /// Set when the model was unreachable or its reply unusable. A term map
    /// with a note is still a legal value; it simply drew no distinctions.
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: u64,
}

#[derive(Debug, Clone)]
pub struct Ollama {
    endpoint: String,
    model: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    /// Omitted entirely for free text; `json` when a schema is wanted.
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'a str>,
    options: Options,
}

#[derive(Serialize)]
struct Options {
    temperature: f64,
    num_predict: i32,
}

#[derive(Deserialize)]
struct GenerateResponse {
    #[serde(default)]
    response: String,
}

#[derive(Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagEntry>,
}

#[derive(Deserialize)]
struct TagEntry {
    name: String,
    #[serde(default)]
    size: u64,
}

/// The one prompt. It asks for distinctions and forbids judgement, because
/// a model that returned a verdict would be answering a question the system
/// has no vocabulary to represent.
const PROMPT: &str = "\
You are naming the distinctions a piece of audio draws. You are NOT judging it.

Rules:
- Return ONLY a JSON object of the form {\"distinctions\": [\"...\", \"...\"]}.
- Each distinction is a short lowercase snake_case phrase naming a property the
  material either has or lacks, e.g. sub_heavy, mid_scooped, short_decay,
  wide_stereo, gritty_saturation.
- Name at most 8. Fewer is better than invented ones.
- NEVER include quality words: good, bad, better, professional, muddy, clean.
- NEVER rank, score, or recommend anything.

Measurements taken of the material:
";

impl Ollama {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap_or_default(),
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Which models the local server holds. Used to report availability
    /// rather than to choose among them.
    pub async fn models(&self) -> Result<Vec<ModelInfo>, String> {
        let url = format!("{}/api/tags", self.endpoint);
        let resp = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| format!("ollama unreachable at {}: {e}", self.endpoint))?;
        let tags: TagsResponse = resp
            .json()
            .await
            .map_err(|e| format!("ollama returned an unreadable model list: {e}"))?;
        Ok(tags
            .models
            .into_iter()
            .map(|m| ModelInfo { name: m.name, size: m.size })
            .collect())
    }

    /// Ask the model for free text, used to compose source for review.
    ///
    /// Unlike `draw_distinctions` this reports failure, because a compose
    /// that silently returned nothing would look like the model having
    /// written an empty program.
    pub async fn write_source(&self, prompt: &str) -> Result<String, String> {
        let body = GenerateRequest {
            model: &self.model,
            prompt,
            stream: false,
            // free text, not JSON: the reply is source in another language
            format: None,
            options: Options { temperature: 0.2, num_predict: 512 },
        };
        let url = format!("{}/api/generate", self.endpoint);
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("{e}"))?;
        let parsed: GenerateResponse =
            resp.json().await.map_err(|e| format!("unreadable reply: {e}"))?;
        Ok(parsed.response)
    }

    /// Draw distinctions from a set of measurements.
    ///
    /// Never returns `Err` for an unreachable model: a rung that could not
    /// run drew no distinctions, which is a fact about the run and is
    /// carried as a note rather than raised.
    pub async fn draw_distinctions(&self, measurements: &str) -> TermMap {
        let prompt = format!("{PROMPT}{measurements}\n");
        let body = GenerateRequest {
            model: &self.model,
            prompt: &prompt,
            stream: false,
            format: Some("json"),
            options: Options { temperature: 0.1, num_predict: 256 },
        };

        let url = format!("{}/api/generate", self.endpoint);
        let resp = match self.client.post(&url).json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                return TermMap {
                    note: Some(format!("model unreachable: {e}")),
                    ..Default::default()
                }
            }
        };
        let parsed: GenerateResponse = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                return TermMap {
                    note: Some(format!("model reply unreadable: {e}")),
                    ..Default::default()
                }
            }
        };
        parse_term_map(&parsed.response)
    }
}

/// Recover a term map from a model reply, discarding anything that is not a
/// well-formed distinction.
pub fn parse_term_map(reply: &str) -> TermMap {
    let Some(start) = reply.find('{') else {
        return TermMap {
            note: Some("model returned no JSON object".into()),
            ..Default::default()
        };
    };
    let Some(end) = reply.rfind('}') else {
        return TermMap {
            note: Some("model returned an unterminated JSON object".into()),
            ..Default::default()
        };
    };
    let slice = &reply[start..=end];

    let value: serde_json::Value = match serde_json::from_str(slice) {
        Ok(v) => v,
        Err(e) => {
            return TermMap {
                note: Some(format!("model reply was not valid JSON: {e}")),
                ..Default::default()
            }
        }
    };

    let raw = value
        .get("distinctions")
        .and_then(|d| d.as_array())
        .cloned()
        .unwrap_or_default();

    let mut distinctions: Vec<String> = raw
        .into_iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .filter_map(|s| normalise(&s))
        .collect();
    distinctions.sort();
    distinctions.dedup();
    distinctions.truncate(8);

    if distinctions.is_empty() {
        return TermMap {
            distinctions,
            note: Some("model drew no usable distinctions".into()),
        };
    }
    TermMap { distinctions, note: None }
}

/// Quality words are dropped rather than trusted: the model is not
/// permitted to judge, and a reply that judges anyway is filtered at the
/// boundary rather than argued with.
const JUDGEMENTS: &[&str] = &[
    "good", "bad", "better", "worse", "best", "worst", "professional",
    "amateur", "muddy", "clean", "nice", "poor", "excellent", "weak",
    "strong", "quality", "pleasing", "harsh",
];

fn normalise(raw: &str) -> Option<String> {
    let cleaned: String = raw
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let cleaned = cleaned.trim_matches('_').to_string();
    if cleaned.is_empty() || cleaned.len() > 48 {
        return None;
    }
    if cleaned.split('_').any(|w| JUDGEMENTS.contains(&w)) {
        return None;
    }
    Some(cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_well_formed_reply_yields_distinctions() {
        let tm = parse_term_map(
            r#"{"distinctions": ["sub_heavy", "mid_scooped", "short_decay"]}"#,
        );
        assert_eq!(tm.distinctions.len(), 3);
        assert!(tm.note.is_none());
    }

    #[test]
    fn judgement_words_are_dropped() {
        let tm = parse_term_map(
            r#"{"distinctions": ["sub_heavy", "good_mix", "muddy", "wide_stereo"]}"#,
        );
        assert_eq!(tm.distinctions, vec!["sub_heavy", "wide_stereo"]);
    }

    #[test]
    fn prose_around_the_object_is_tolerated() {
        let tm = parse_term_map(
            "Here you go:\n{\"distinctions\": [\"gritty_saturation\"]}\nHope that helps.",
        );
        assert_eq!(tm.distinctions, vec!["gritty_saturation"]);
    }

    #[test]
    fn a_broken_reply_is_a_note_not_a_failure() {
        for bad in ["", "no json here", "{not json}", r#"{"other": 1}"#] {
            let tm = parse_term_map(bad);
            assert!(tm.distinctions.is_empty());
            assert!(tm.note.is_some(), "{bad:?} should carry a note");
        }
    }

    #[test]
    fn distinctions_are_capped_and_deduplicated() {
        let many: Vec<String> = (0..20).map(|i| format!("\"d_{i}\"")).collect();
        let tm = parse_term_map(&format!(
            "{{\"distinctions\": [{}, \"d_0\"]}}",
            many.join(",")
        ));
        assert_eq!(tm.distinctions.len(), 8);
        let mut sorted = tm.distinctions.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), tm.distinctions.len());
    }
}
