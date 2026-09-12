//! Runtime persistence: the record survives a restart.
//!
//! Workspace-relative JSON, written atomically (temp file + rename) so a
//! crash mid-write never leaves a truncated file behind. Load never fails
//! outward: a missing or corrupt snapshot yields `None` and the caller
//! starts from a fresh runtime, matching the "never fails" discipline used
//! throughout `integrations::analysis`.

use std::path::{Path, PathBuf};

use crate::graph::RuntimeSnapshot;

fn snapshot_path(workspace: &Path) -> PathBuf {
    workspace.join(".heihachi").join("runtime.json")
}

/// Write the snapshot atomically: a temp file, then a rename over the target.
pub fn save(workspace: &Path, snapshot: &RuntimeSnapshot) -> std::io::Result<()> {
    let path = snapshot_path(workspace);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    let body = serde_json::to_vec_pretty(snapshot)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Load a snapshot, if one exists and parses. Never propagates an error --
/// a missing or corrupt file just means there is nothing to restore.
pub fn load(workspace: &Path) -> Option<RuntimeSnapshot> {
    let path = snapshot_path(workspace);
    let body = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            tracing::warn!("cannot read {}: {e}", path.display());
            return None;
        }
    };
    match serde_json::from_slice(&body) {
        Ok(snap) => Some(snap),
        Err(e) => {
            tracing::warn!("cannot parse {}: {e}", path.display());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Runtime;

    #[test]
    fn round_trips_through_save_and_load() {
        let dir = std::env::temp_dir().join(format!("heihachi-persist-test-{}", rand_suffix()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut rt = Runtime::new();
        rt.attach_chunk("a", "m");
        rt.run("a", |_, _| {
            Ok(vec![crate::graph::Value::reading("a.level", -6.0, 0.1, "dB", "t").unwrap()])
        });

        save(&dir, &rt.snapshot()).unwrap();
        let loaded = load(&dir).expect("snapshot should load");
        assert_eq!(loaded.record, rt.record());
        assert_eq!(loaded.nodes.len(), rt.node_count());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_file_loads_as_none() {
        let dir = std::env::temp_dir().join(format!("heihachi-persist-missing-{}", rand_suffix()));
        assert!(load(&dir).is_none());
    }

    #[test]
    fn corrupt_file_loads_as_none_not_a_panic() {
        let dir = std::env::temp_dir().join(format!("heihachi-persist-corrupt-{}", rand_suffix()));
        std::fs::create_dir_all(dir.join(".heihachi")).unwrap();
        std::fs::write(dir.join(".heihachi").join("runtime.json"), b"{ not json").unwrap();

        assert!(load(&dir).is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    fn rand_suffix() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
    }
}
