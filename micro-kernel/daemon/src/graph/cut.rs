//! Contact graphs, maximum flow, and the accountability query.
//!
//! "Why did this happen?" is answered here as a minimum cut over the
//! provenance graph: the set of emissions jointly necessary for an outcome.
//! This is a post-hoc query and belongs in a report, never in a render loop.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

pub const MEDIUM: &str = "@medium";

/// A finite weighted graph with a distinguished medium vertex.
#[derive(Debug, Clone)]
pub struct ContactGraph {
    declared_floor: f64,
    weights: BTreeMap<(String, String), f64>,
    items: BTreeSet<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("declared floor must be strictly positive, got {0}")]
    NonPositiveFloor(f64),
    #[error("weight {weight} is below the declared floor {floor}")]
    BelowFloor { weight: f64, floor: f64 },
}

fn key(u: &str, v: &str) -> (String, String) {
    if u <= v {
        (u.to_string(), v.to_string())
    } else {
        (v.to_string(), u.to_string())
    }
}

impl ContactGraph {
    pub fn new(declared_floor: f64) -> Result<Self, GraphError> {
        if !(declared_floor > 0.0) {
            return Err(GraphError::NonPositiveFloor(declared_floor));
        }
        Ok(Self {
            declared_floor,
            weights: BTreeMap::new(),
            items: BTreeSet::new(),
        })
    }

    pub fn link(&mut self, u: &str, v: &str, weight: f64) -> Result<(), GraphError> {
        if weight < self.declared_floor {
            return Err(GraphError::BelowFloor { weight, floor: self.declared_floor });
        }
        self.weights.insert(key(u, v), weight);
        for x in [u, v] {
            if x != MEDIUM {
                self.items.insert(x.to_string());
            }
        }
        Ok(())
    }

    /// Join an item to the medium, so that every item has a finite
    /// separation cost and every cut is non-empty.
    pub fn attach(&mut self, v: &str, weight: f64) -> Result<(), GraphError> {
        self.link(v, MEDIUM, weight)
    }

    /// The realised floor: the smallest weight actually present, or the
    /// declared floor when no contact has been committed yet.
    pub fn floor(&self) -> f64 {
        match self.weights.values().copied().reduce(f64::min) {
            Some(smallest) => smallest,
            None => self.declared_floor,
        }
    }

    pub fn omega(&self) -> f64 {
        self.weights.values().sum()
    }

    pub fn vertices(&self) -> Vec<String> {
        let mut set = BTreeSet::new();
        for (u, v) in self.weights.keys() {
            set.insert(u.clone());
            set.insert(v.clone());
        }
        set.into_iter().collect()
    }

    fn capacities(&self) -> BTreeMap<String, BTreeMap<String, f64>> {
        let mut cap: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        for x in self.vertices() {
            cap.entry(x).or_default();
        }
        for ((u, v), w) in &self.weights {
            *cap.entry(u.clone()).or_default().entry(v.clone()).or_insert(0.0) += w;
            *cap.entry(v.clone()).or_default().entry(u.clone()).or_insert(0.0) += w;
        }
        cap
    }

    /// Maximum flow by Edmonds-Karp, with the source side of a minimum cut.
    ///
    /// Each undirected edge is modelled as two opposed arcs of equal
    /// capacity, which is the standard reduction.
    pub fn max_flow(&self, source: &str, sink: &str) -> (f64, BTreeSet<String>) {
        if source == sink {
            return (0.0, BTreeSet::from([source.to_string()]));
        }
        let mut cap = self.capacities();
        if !cap.contains_key(source) || !cap.contains_key(sink) {
            return (0.0, BTreeSet::from([source.to_string()]));
        }

        let mut flow = 0.0;
        loop {
            // breadth-first search for an augmenting path
            let mut parent: BTreeMap<String, String> = BTreeMap::new();
            parent.insert(source.to_string(), source.to_string());
            let mut queue = VecDeque::from([source.to_string()]);
            while let Some(u) = queue.pop_front() {
                if parent.contains_key(sink) {
                    break;
                }
                let neighbours: Vec<(String, f64)> = cap
                    .get(&u)
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
                    .unwrap_or_default();
                for (v, c) in neighbours {
                    if c > 1e-12 && !parent.contains_key(&v) {
                        parent.insert(v.clone(), u.clone());
                        queue.push_back(v);
                    }
                }
            }
            if !parent.contains_key(sink) {
                break;
            }

            // bottleneck along the path
            let mut bottleneck = f64::INFINITY;
            let mut v = sink.to_string();
            while v != source {
                let u = parent[&v].clone();
                bottleneck = bottleneck.min(cap[&u][&v]);
                v = u;
            }
            // augment
            let mut v = sink.to_string();
            while v != source {
                let u = parent[&v].clone();
                *cap.get_mut(&u).unwrap().get_mut(&v).unwrap() -= bottleneck;
                *cap.entry(v.clone()).or_default().entry(u.clone()).or_insert(0.0) +=
                    bottleneck;
                v = u;
            }
            flow += bottleneck;
        }

        let mut reachable = BTreeSet::from([source.to_string()]);
        let mut stack = vec![source.to_string()];
        while let Some(u) = stack.pop() {
            let neighbours: Vec<(String, f64)> = cap
                .get(&u)
                .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
                .unwrap_or_default();
            for (v, c) in neighbours {
                if c > 1e-12 && !reachable.contains(&v) {
                    reachable.insert(v.clone());
                    stack.push(v);
                }
            }
        }
        (flow, reachable)
    }

    /// Minimum weight of a cut separating two items.
    pub fn separation(&self, u: &str, v: &str) -> f64 {
        self.max_flow(u, v).0
    }

    /// Separation cost of an item against the medium.
    pub fn strength(&self, v: &str) -> f64 {
        self.separation(v, MEDIUM)
    }

    /// Normalised separation, in (0, 1].
    pub fn alignment(&self, x: &str, target: &str) -> f64 {
        let omega = self.omega();
        if omega <= 0.0 {
            return 1.0;
        }
        if x == target {
            return self.floor() / omega;
        }
        self.separation(x, target) / omega
    }

    /// The emissions jointly necessary for an outcome: a minimum cut
    /// separating it from the exogenous inputs.
    pub fn accountable_set(&self, source: &str, outcome: &str) -> Accountability {
        let (weight, near) = self.max_flow(source, outcome);
        let mut cut = Vec::new();
        for ((u, v), w) in &self.weights {
            if near.contains(u) != near.contains(v) {
                cut.push(CutEdge { from: u.clone(), to: v.clone(), weight: *w });
            }
        }
        cut.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
        Accountability { weight, edges: cut }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutEdge {
    pub from: String,
    pub to: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Accountability {
    /// Total weight of the cut: the cost of severing the outcome.
    pub weight: f64,
    /// The jointly necessary contacts, heaviest first.
    pub edges: Vec<CutEdge>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimum over every bipartition. Exponential; small graphs only.
    fn brute_force(g: &ContactGraph, source: &str, sink: &str) -> f64 {
        let free: Vec<String> = g
            .vertices()
            .into_iter()
            .filter(|x| x != source && x != sink)
            .collect();
        let mut best = f64::INFINITY;
        for mask in 0u32..(1 << free.len()) {
            let mut side = BTreeSet::from([source.to_string()]);
            for (i, x) in free.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    side.insert(x.clone());
                }
            }
            let mut total = 0.0;
            for ((u, v), w) in &g.weights {
                if side.contains(u) != side.contains(v) {
                    total += w;
                }
            }
            best = best.min(total);
        }
        best
    }

    #[test]
    fn max_flow_agrees_with_brute_force() {
        // a deterministic spread of shapes rather than a random sweep, so
        // the test is reproducible without carrying a seed
        let shapes: &[&[(&str, &str, f64)]] = &[
            &[("a", "b", 0.3), ("b", "c", 0.5)],
            &[("a", "b", 0.9), ("a", "c", 0.2), ("b", "c", 0.4)],
            &[("a", "b", 0.1), ("b", "c", 0.1), ("c", "d", 0.1), ("a", "d", 0.7)],
            &[("a", "b", 1.0), ("b", "c", 1.0), ("c", "d", 1.0), ("d", "e", 1.0)],
        ];
        for shape in shapes {
            let mut g = ContactGraph::new(0.02).unwrap();
            for (u, v, w) in *shape {
                g.link(u, v, *w).unwrap();
            }
            for v in g.items.clone() {
                g.attach(&v, 0.25).unwrap();
            }
            for v in g.items.clone() {
                let mf = g.separation(&v, MEDIUM);
                let bf = brute_force(&g, &v, MEDIUM);
                assert!((mf - bf).abs() < 1e-9, "{v}: flow {mf} vs brute {bf}");
            }
        }
    }

    #[test]
    fn weight_below_declared_floor_is_refused() {
        let mut g = ContactGraph::new(0.05).unwrap();
        assert!(g.link("a", "b", 0.01).is_err());
        assert!(g.link("a", "b", 0.05).is_ok());
    }

    #[test]
    fn accountable_set_is_non_empty_for_a_connected_outcome() {
        let mut g = ContactGraph::new(0.02).unwrap();
        g.link("source", "mid", 0.6).unwrap();
        g.link("mid", "outcome", 0.4).unwrap();
        g.attach("mid", 0.9).unwrap();
        let acc = g.accountable_set("source", "outcome");
        assert!(acc.weight > 0.0);
        assert!(!acc.edges.is_empty());
    }
}
