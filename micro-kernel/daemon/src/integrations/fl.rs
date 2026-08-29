//! FL Studio integration by watching exports and reading projects.
//!
//! FL exposes no general remote-control API, so this is deliberately
//! read-only and requires nothing to be installed into FL. You render as
//! normal; the daemon notices the file, analyses it, and commits a node to
//! the runtime graph. A `.flp` beside the render is parsed for the device
//! chain that produced it.
//!
//! The `.flp` format is undocumented. What is parsed here is the outer
//! event stream, which is stable enough to recover plugin and channel
//! names; anything not recognised is skipped rather than guessed at, and
//! an unreadable project yields an anomaly value rather than a failure.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A render the daemon noticed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Export {
    pub path: PathBuf,
    pub stem: String,
    pub bytes: u64,
    /// The project sitting beside it, if one was found.
    pub project: Option<PathBuf>,
}

/// What a project file says about how a sound was made.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectChain {
    pub title: Option<String>,
    pub tempo: Option<f64>,
    /// Generator and effect names, in the order they appear.
    pub devices: Vec<String>,
    pub channels: Vec<String>,
    /// Set when the file could not be read as a project.
    pub note: Option<String>,
}

const AUDIO_EXTENSIONS: &[&str] = &["wav", "flac", "aiff", "aif", "mp3", "ogg"];

pub fn is_audio(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| AUDIO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Look for a project file beside a render, or one directory up.
pub fn project_beside(render: &Path) -> Option<PathBuf> {
    let stem = render.file_stem()?.to_str()?.to_string();
    let dirs = [render.parent()?, render.parent()?.parent()?];
    for dir in dirs {
        let direct = dir.join(format!("{stem}.flp"));
        if direct.is_file() {
            return Some(direct);
        }
        // otherwise take any single project in the directory
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut found: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase)
                        == Some("flp".into())
                })
                .collect();
            if found.len() == 1 {
                return found.pop();
            }
        }
    }
    None
}

/// Parse the outer event stream of a `.flp` for names and tempo.
///
/// This reads the container rather than interpreting it: FLhd/FLdt chunks,
/// then a byte-oriented event stream in which events below 0x40 carry one
/// byte, below 0x80 two, below 0xC0 four, and above that a length-prefixed
/// payload. Text events are recovered; everything else is skipped.
pub fn read_project(path: &Path) -> ProjectChain {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            return ProjectChain {
                note: Some(format!("project unreadable: {e}")),
                ..Default::default()
            }
        }
    };
    if bytes.len() < 8 || &bytes[0..4] != b"FLhd" {
        return ProjectChain {
            note: Some("not an FL project (no FLhd header)".into()),
            ..Default::default()
        };
    }

    // find the FLdt data chunk
    let Some(data_at) = find(&bytes, b"FLdt") else {
        return ProjectChain {
            note: Some("project has no FLdt data chunk".into()),
            ..Default::default()
        };
    };
    let mut i = data_at + 8;

    let mut chain = ProjectChain::default();
    let mut texts: Vec<(u8, String)> = Vec::new();

    while i < bytes.len() {
        let event = bytes[i];
        i += 1;
        if event < 0x40 {
            i += 1;
        } else if event < 0x80 {
            i += 2;
        } else if event < 0xC0 {
            if event == 0x9C && i + 4 <= bytes.len() {
                // tempo, in millibeats per minute
                let raw = u32::from_le_bytes([
                    bytes[i],
                    bytes[i + 1],
                    bytes[i + 2],
                    bytes[i + 3],
                ]);
                if raw > 0 {
                    chain.tempo = Some(f64::from(raw) / 1000.0);
                }
            }
            i += 4;
        } else {
            // variable-length: a 7-bit encoded size, then the payload
            let mut size = 0usize;
            let mut shift = 0u32;
            while i < bytes.len() {
                let b = bytes[i];
                i += 1;
                size |= ((b & 0x7F) as usize) << shift;
                if b & 0x80 == 0 {
                    break;
                }
                shift += 7;
                if shift > 28 {
                    break;
                }
            }
            let end = i.saturating_add(size).min(bytes.len());
            let payload = &bytes[i..end];
            if let Some(text) = utf16_text(payload) {
                if !text.trim().is_empty() {
                    texts.push((event, text));
                }
            }
            i = end;
        }
    }

    // 0xC9 is the generator/plugin name; 0xC3 the channel name; 0xC1 the title
    for (event, text) in texts {
        match event {
            0xC9 | 0xD5 => chain.devices.push(text),
            0xC3 | 0xCB => chain.channels.push(text),
            0xC1 => chain.title = Some(text),
            _ => {}
        }
    }
    chain.devices.dedup();
    chain.channels.dedup();
    chain
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Decode a UTF-16LE payload, which is how FL stores names.
fn utf16_text(payload: &[u8]) -> Option<String> {
    if payload.len() < 4 || payload.len() % 2 != 0 {
        return None;
    }
    let units: Vec<u16> = payload
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .take_while(|u| *u != 0)
        .collect();
    if units.len() < 2 {
        return None;
    }
    let text = String::from_utf16(&units).ok()?;
    // reject payloads that decoded to control characters: those were not text
    if text.chars().any(|c| c.is_control()) {
        return None;
    }
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_extensions_are_recognised() {
        assert!(is_audio(Path::new("bounce.wav")));
        assert!(is_audio(Path::new("bounce.FLAC")));
        assert!(!is_audio(Path::new("project.flp")));
        assert!(!is_audio(Path::new("notes.txt")));
    }

    #[test]
    fn a_non_project_yields_a_note_rather_than_a_panic() {
        let dir = std::env::temp_dir().join("hk_fl_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("not_a_project.flp");
        std::fs::write(&p, b"this is not an FL project").unwrap();
        let chain = read_project(&p);
        assert!(chain.note.is_some());
        assert!(chain.devices.is_empty());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_missing_project_yields_a_note() {
        let chain = read_project(Path::new("does_not_exist_anywhere.flp"));
        assert!(chain.note.unwrap().contains("unreadable"));
    }

    #[test]
    fn utf16_decoding_rejects_binary() {
        assert!(utf16_text(&[0x41, 0x00, 0x42, 0x00]).is_some());
        assert!(utf16_text(&[0x01, 0x00, 0x02, 0x00]).is_none());
        assert!(utf16_text(&[0x41]).is_none());
    }
}
