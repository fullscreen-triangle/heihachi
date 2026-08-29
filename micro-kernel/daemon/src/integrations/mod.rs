//! Integrations with the tools already on the machine.
//!
//! Each is read-only and degrades to a note rather than an error: a render
//! that cannot be decoded, a project that cannot be parsed and a model that
//! cannot be reached are all facts about the run, carried as values.

pub mod analysis;
pub mod fl;
pub mod ollama;
