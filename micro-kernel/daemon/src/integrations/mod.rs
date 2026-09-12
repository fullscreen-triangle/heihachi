//! Integrations with the tools already on the machine.
//!
//! Each is read-only and degrades to a note rather than an error: a render
//! that cannot be decoded, a project that cannot be parsed and a model that
//! cannot be reached are all facts about the run, carried as values.
//!
//! `fl_control` is the one exception to "read-only": it dispatches a narrow
//! set of transport/mixer/pattern commands to a connected FL MIDI Controller
//! Script. It keeps the same degrade-to-a-value discipline -- a dispatch
//! with nothing connected yields an anomaly, never a hang or a panic.

pub mod analysis;
pub mod clap_host;
pub mod fl;
pub mod fl_control;
pub mod ollama;
