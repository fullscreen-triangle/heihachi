//! The one real two-way channel into FL Studio: an FL MIDI Controller
//! Script, dialing into this daemon over a loopback socket.
//!
//! FL exposes no general remote-control API. Its Python scripting surface
//! (`transport`, `mixer`, `patterns`, `channels`) is the only thing that can
//! be driven from outside, and it is narrow: transport, mixer track
//! volume/pan, and pattern/channel selection. Nothing here inserts plugins,
//! edits the piano roll, writes automation, or renders -- FL's API does not
//! expose those, and no protocol on this side changes that.
//!
//! Acts are not commands-and-wait-for-exit-code. Dispatching a command
//! attaches a chunk and emits a value recording what was *requested*; it
//! does not block on FL's state actually changing, and an unreachable
//! script yields an anomaly value rather than a hung request or an error
//! response. A later acknowledgement from the script, if one arrives, is a
//! separate emitted value, not a reply to the first.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportAction {
    Start,
    Stop,
    Record,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MixerParam {
    Volume,
    Pan,
}

/// A command dispatched to the connected FL script. Tagged the same way as
/// [`crate::graph::Delta`], for the same reason: one wire shape per variant,
/// legible without a lookup table.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Command {
    Transport { action: TransportAction },
    MixerSet { track: u32, param: MixerParam, value: f64 },
    PatternJump { index: u32 },
}

impl Command {
    /// A short name for the command's kind, used to build a `tau` for the
    /// act it produces -- not the wire tag (which serde owns) but stable
    /// enough to be a subtask identity.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Command::Transport { .. } => "transport",
            Command::MixerSet { .. } => "mixer_set",
            Command::PatternJump { .. } => "pattern_jump",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FlLinkError {
    #[error("no FL script is connected")]
    NotConnected,
    #[error("could not write to the connected script: {0}")]
    Io(#[from] std::io::Error),
    #[error("could not encode the command: {0}")]
    Encode(#[from] serde_json::Error),
}

/// The daemon's side of the socket the FL script dials into.
///
/// Binds loopback only, matching the boundary the HTTP daemon already
/// holds. A script that hasn't connected yet, or has dropped, leaves `send`
/// returning `Err(NotConnected)` rather than blocking -- the caller turns
/// that into an anomaly value, never a hang.
pub struct FlLink {
    peer: Mutex<Option<TcpStream>>,
}

impl FlLink {
    /// Bind the listener and spawn the accept loop. Returns immediately;
    /// the returned `FlLink` is empty until a script connects.
    pub fn listen(addr: std::net::SocketAddr) -> (Arc<Self>, tokio::task::JoinHandle<()>) {
        let link = Arc::new(Self { peer: Mutex::new(None) });
        let accepting = Arc::clone(&link);
        let handle = tokio::spawn(async move {
            let listener = match TcpListener::bind(addr).await {
                Ok(l) => l,
                Err(e) => {
                    tracing::error!("fl_control: cannot bind {addr}: {e}");
                    return;
                }
            };
            tracing::info!("fl_control listening on {addr}");
            loop {
                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        tracing::info!("fl_control: script connected from {peer_addr}");
                        *accepting.peer.lock().await = Some(stream);
                    }
                    Err(e) => {
                        tracing::warn!("fl_control: accept failed: {e}");
                    }
                }
            }
        });
        (link, handle)
    }

    /// An `FlLink` with no listener at all, for contexts (tests, CLI
    /// subcommands) that never expect a script to connect.
    pub fn disconnected() -> Arc<Self> {
        Arc::new(Self { peer: Mutex::new(None) })
    }

    /// Dispatch one command as a line-delimited JSON frame.
    ///
    /// Never blocks waiting for FL's state to change and never retries; a
    /// write failure drops the peer so the next `send` reports
    /// `NotConnected` instead of writing into a dead socket.
    pub async fn send(&self, cmd: &Command) -> Result<(), FlLinkError> {
        let mut line = serde_json::to_string(cmd)?;
        line.push('\n');

        let mut guard = self.peer.lock().await;
        let Some(stream) = guard.as_mut() else {
            return Err(FlLinkError::NotConnected);
        };
        if let Err(e) = stream.write_all(line.as_bytes()).await {
            *guard = None;
            return Err(FlLinkError::Io(e));
        }
        Ok(())
    }

    pub async fn is_connected(&self) -> bool {
        self.peer.lock().await.is_some()
    }
}

/// Read one acknowledgement line from a peer stream, if any is pending.
/// Exposed separately from the accept loop so the background reader task in
/// `main.rs` can be written without duplicating the framing logic.
pub async fn read_ack_line(stream: &mut TcpStream) -> Option<String> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    match reader.read_line(&mut line).await {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line.trim_end().to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_wire_shape_matches_the_documented_protocol() {
        let cmd = Command::Transport { action: TransportAction::Start };
        let json = serde_json::to_value(&cmd).unwrap();
        assert_eq!(json, serde_json::json!({"cmd": "transport", "action": "start"}));

        let cmd = Command::MixerSet { track: 3, param: MixerParam::Volume, value: 0.8 };
        let json = serde_json::to_value(&cmd).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"cmd": "mixer_set", "track": 3, "param": "volume", "value": 0.8})
        );

        let cmd = Command::PatternJump { index: 5 };
        let json = serde_json::to_value(&cmd).unwrap();
        assert_eq!(json, serde_json::json!({"cmd": "pattern_jump", "index": 5}));
    }

    #[test]
    fn command_round_trips_through_json() {
        let cmd = Command::MixerSet { track: 1, param: MixerParam::Pan, value: -0.5 };
        let s = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&s).unwrap();
        assert_eq!(back.kind_name(), "mixer_set");
    }

    #[tokio::test]
    async fn sending_with_nothing_connected_is_an_error_not_a_hang() {
        let link = FlLink::disconnected();
        let result = link.send(&Command::Transport { action: TransportAction::Start }).await;
        assert!(matches!(result, Err(FlLinkError::NotConnected)));
    }
}
