//! The heihachi micro-kernel: a runtime graph, two languages, and the
//! integrations that feed them.
//!
//! The kernel is `graph`; `lang` holds the two front ends; `integrations`
//! reads the tools already on the machine; `studio` turns a render into a
//! node; `server` exposes all of it to a paired interface on loopback.

pub mod compose;
pub mod graph;
pub mod integrations;
pub mod lang;
pub mod server;
pub mod studio;
