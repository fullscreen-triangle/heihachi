//! CLAP plugin hosting: the difference between checking whether a `sangoma`
//! chain *could* reach a target and actually rendering one.
//!
//! Built against `clack-host` 0.1.1 / `clack-extensions` 0.1.1, verified
//! against their source rather than guessed: the host implements
//! [`HostHandlers`] (a trait with lifetime-generic associated types for the
//! main-thread, audio-thread, and thread-safe handler halves CLAP's spec
//! requires), loads a plugin through [`PluginEntry::load`], instantiates it
//! with [`PluginInstance::new`], activates it into a
//! [`StoppedPluginAudioProcessor`], and renders by calling
//! [`StartedPluginAudioProcessor::process`] in a plain loop over blocks of
//! silence -- there is no separate "offline render" entry point; offline
//! rendering *is* driving `process()` outside a realtime callback.
//!
//! Parameters are not set with a setter: CLAP communicates parameter changes
//! as a [`ParamValueEvent`] injected into the same input event buffer used
//! for `process()`. `HostedPlugin::set_param` stages a value and injects it
//! at the next render call.
//!
//! # Safety
//!
//! Loading a `.clap` file executes third-party native code. `clack-host`'s
//! own documentation is explicit that this can trigger Undefined Behavior
//! for a non-compliant plugin regardless of how carefully the host is
//! written; nothing here adds process isolation. A malfunctioning plugin can
//! crash the daemon, not just this rendering path.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use clack_extensions::params::{HostParams, HostParamsImplMainThread, HostParamsImplShared, PluginParams};
use clack_host::events::event_types::ParamValueEvent;
use clack_host::events::io::{EventBuffer, InputEvents, OutputEvents};
use clack_host::events::Pckn;
use clack_host::host::{HostExtensions, HostHandlers, HostInfo, MainThreadHandler, SharedHandler};
use clack_host::plugin::{InitializingPluginHandle, PluginInstance, PluginInstanceError};
use clack_host::process::audio_buffers::{AudioPortBuffer, AudioPortBufferType, AudioPorts, InputChannel};
use clack_host::process::{PluginAudioConfiguration, ProcessStatus};

/// What a scan found, before anything is loaded. Enumerating a `.clap`
/// bundle's descriptors never fails outward -- an unreadable or
/// non-compliant bundle is skipped with a note, matching the "never fails"
/// discipline used throughout `integrations::analysis`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PluginDescriptor {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
}

/// Standard CLAP plugin search directories for the current platform. CLAP
/// bundles are conventionally installed system- or user-wide; this is a
/// starting point, not exhaustive -- callers may add configured directories
/// on top.
pub fn standard_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    #[cfg(target_os = "windows")]
    {
        if let Ok(common) = std::env::var("COMMONPROGRAMFILES") {
            dirs.push(PathBuf::from(common).join("CLAP"));
        }
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            dirs.push(PathBuf::from(local).join("Programs").join("Common").join("CLAP"));
        }
    }
    #[cfg(target_os = "macos")]
    {
        dirs.push(PathBuf::from("/Library/Audio/Plug-Ins/CLAP"));
        if let Some(home) = dirs::home_dir_fallback() {
            dirs.push(home.join("Library/Audio/Plug-Ins/CLAP"));
        }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        dirs.push(PathBuf::from("/usr/lib/clap"));
        if let Ok(home) = std::env::var("HOME") {
            dirs.push(PathBuf::from(home).join(".clap"));
        }
    }
    dirs
}

#[cfg(target_os = "macos")]
mod dirs {
    pub fn home_dir_fallback() -> Option<std::path::PathBuf> {
        std::env::var("HOME").ok().map(std::path::PathBuf::from)
    }
}

/// Walk the given directories for `.clap` bundles and enumerate the plugins
/// each one declares. Never panics: a bundle that fails to load, or an
/// unreadable directory, is skipped rather than surfaced as an error --
/// there is no way for a caller to act on "this one third-party file is
/// broken" other than not listing it.
pub fn scan(dirs: &[PathBuf]) -> Vec<PluginDescriptor> {
    let mut found = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("clap") {
                continue;
            }
            found.extend(scan_bundle(&path));
        }
    }
    found
}

fn scan_bundle(path: &Path) -> Vec<PluginDescriptor> {
    // SAFETY: loading an arbitrary third-party dynamic library is inherently
    // unsafe -- see the module doc's Safety section. This is the daemon's
    // one deliberate crossing of that line, confined to this function.
    let entry = match unsafe { clack_host::entry::PluginEntry::load(path) } {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("clap_host: cannot load {}: {e}", path.display());
            return Vec::new();
        }
    };
    let Some(factory) = entry.get_plugin_factory() else {
        return Vec::new();
    };
    factory
        .plugin_descriptors()
        .filter_map(|d| {
            let id = d.id()?.to_str().ok()?.to_string();
            let name = d.name()?.to_str().ok()?.to_string();
            Some(PluginDescriptor { id, name, path: path.to_path_buf() })
        })
        .collect()
}

// ── the host implementation CLAP's spec requires ───────────────────────
//
// CLAP splits a host into three thread-scoped handlers (main-thread,
// audio-thread, and a thread-safe "shared" handler bridging them). heihachi
// needs none of their callback surface beyond what params requires to read
// back a plugin's declared parameter count, so these are the minimum
// non-unit implementations that satisfy `HostHandlers`.

struct Shared {
    params: std::sync::OnceLock<Option<PluginParams>>,
}

impl<'a> SharedHandler<'a> for Shared {
    fn initializing(&self, instance: InitializingPluginHandle<'a>) {
        let _ = self.params.set(instance.get_extension());
    }
    fn request_restart(&self) {}
    fn request_process(&self) {}
    fn request_callback(&self) {}
}

impl HostParamsImplShared for Shared {
    fn request_flush(&self) {}
}

struct MainThread;

impl<'a> MainThreadHandler<'a> for MainThread {}

impl HostParamsImplMainThread for MainThread {
    fn rescan(&mut self, _flags: clack_extensions::params::ParamRescanFlags) {}
    fn clear(&mut self, _param_id: clack_host::utils::ClapId, _flags: clack_extensions::params::ParamClearFlags) {}
}

struct RenderHost;

impl HostHandlers for RenderHost {
    type Shared<'a> = Shared;
    type MainThread<'a> = MainThread;
    type AudioProcessor<'a> = ();

    fn declare_extensions(builder: &mut HostExtensions<Self>, _shared: &Self::Shared<'_>) {
        builder.register::<HostParams>();
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ClapHostError {
    #[error("plugin entry could not be loaded: {0}")]
    Entry(#[from] clack_host::entry::PluginEntryError),
    #[error("plugin instantiation failed: {0}")]
    Instance(#[from] PluginInstanceError),
    #[error("plugin failed to start processing: {0}")]
    StartProcessing(String),
    #[error("plugin has no id \"{0}\" in bundle {1}")]
    NotFound(String, PathBuf),
}

/// A loaded, activated plugin, ready to render blocks.
///
/// Owns everything `clack-host` needs kept alive for the lifetime of a
/// render: the entry (the loaded dynamic library), the instance, and its
/// stopped/started audio processor state, which this type manages
/// internally so callers never see clack's activate/start/stop/deactivate
/// state machine directly.
pub struct HostedPlugin {
    instance: PluginInstance<RenderHost>,
    pending_params: Vec<ParamValueEvent>,
    activated: AtomicBool,
}

impl HostedPlugin {
    /// Load and instantiate (but not yet activate) the plugin named by
    /// `descriptor.id` inside its bundle.
    pub fn load(descriptor: &PluginDescriptor) -> Result<Self, ClapHostError> {
        // SAFETY: see `scan_bundle` -- loading a third-party dynamic library
        // is unsafe by nature, not made safe by having scanned it already.
        let entry = unsafe { clack_host::entry::PluginEntry::load(&descriptor.path) }?;
        let host_info = HostInfo::new("heihachi", "heihachi", "https://github.com", "0.1.0")
            .map_err(|_| ClapHostError::NotFound(descriptor.id.clone(), descriptor.path.clone()))?;

        let plugin_id = std::ffi::CString::new(descriptor.id.clone())
            .map_err(|_| ClapHostError::NotFound(descriptor.id.clone(), descriptor.path.clone()))?;

        let instance = PluginInstance::<RenderHost>::new(
            |_| Shared { params: std::sync::OnceLock::new() },
            |_shared| MainThread,
            &entry,
            &plugin_id,
            &host_info,
        )?;

        Ok(Self { instance, pending_params: Vec::new(), activated: AtomicBool::new(false) })
    }

    /// Stage a parameter value to be sent with the next [`render`](Self::render)
    /// call. CLAP has no synchronous "set parameter" call; a value change is
    /// only ever communicated as an event delivered during `process()`.
    pub fn set_param(&mut self, param_id: u32, value: f64) {
        self.pending_params.push(ParamValueEvent::new(
            0,
            clack_host::utils::ClapId::new(param_id),
            Pckn::match_all(),
            value,
            clack_host::utils::Cookie::empty(),
        ));
    }

    /// Render `frames` samples of silence-in, capturing whatever the plugin
    /// produces -- an instrument's own generator output, or an effect's
    /// response to silence (usually a tail, or nothing).
    ///
    /// Blocks are capped at 4096 frames per `process()` call, matching the
    /// `max_frames_count` activation bound below; larger requests are
    /// rendered over multiple calls, carrying pending parameter events on
    /// the first block only.
    pub fn render(&mut self, frames: usize, sample_rate: u32) -> Result<Vec<f32>, ClapHostError> {
        const BLOCK: usize = 4096;
        const CHANNELS: usize = 2;

        let config = PluginAudioConfiguration {
            sample_rate: sample_rate as f64,
            min_frames_count: 1,
            max_frames_count: BLOCK as u32,
        };
        let stopped = self.instance.activate(|_, _| (), config)?;
        self.activated.store(true, Ordering::Release);
        let mut started = stopped
            .start_processing()
            .map_err(|e| ClapHostError::StartProcessing(e.to_string()))?;

        let mut output = Vec::with_capacity(frames * CHANNELS);
        let mut remaining = frames;
        let mut first_block = true;

        let mut input_ports = AudioPorts::with_capacity(CHANNELS, 1);
        let mut output_ports = AudioPorts::with_capacity(CHANNELS, 1);

        while remaining > 0 {
            let this_block = remaining.min(BLOCK);
            let mut input_silence = vec![[0.0f32; BLOCK]; CHANNELS];
            let mut output_block = vec![[0.0f32; BLOCK]; CHANNELS];

            let mut input_event_buf: Vec<ParamValueEvent> =
                if first_block { std::mem::take(&mut self.pending_params) } else { Vec::new() };
            first_block = false;
            let input_events = InputEvents::from_buffer(&input_event_buf);
            let mut output_event_buf = EventBuffer::new();
            let mut output_events = OutputEvents::from_buffer(&mut output_event_buf);

            let input_audio = input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_silence.iter_mut().map(|b| InputChannel::constant(&mut b[..this_block])),
                ),
            }]);
            let mut output_audio = output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(
                    output_block.iter_mut().map(|b| &mut b[..this_block]),
                ),
            }]);

            let status = started.process(
                &input_audio,
                &mut output_audio,
                &input_events,
                &mut output_events,
                None,
                None,
            )?;

            for frame in 0..this_block {
                for ch in 0..CHANNELS {
                    output.push(output_block[ch][frame]);
                }
            }

            if status == ProcessStatus::Sleep {
                // The plugin has nothing more to say for the rest of this
                // request; the remaining frames stay silent rather than
                // spending CPU re-driving a plugin that has already slept.
                let silent_remaining = (remaining - this_block) * CHANNELS;
                output.extend(std::iter::repeat(0.0f32).take(silent_remaining));
                remaining = 0;
            } else {
                remaining -= this_block;
            }
            input_event_buf.clear();
        }

        let stopped = started.stop_processing();
        self.instance.deactivate(stopped);
        self.activated.store(false, Ordering::Release);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scanning_a_nonexistent_directory_yields_nothing_not_a_panic() {
        let dirs = vec![PathBuf::from("Z:/this/does/not/exist/at/all")];
        assert!(scan(&dirs).is_empty());
    }

    #[test]
    fn scanning_an_empty_directory_yields_nothing() {
        let dir = std::env::temp_dir().join(format!(
            "heihachi-clap-scan-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(scan(&[dir.clone()]).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
