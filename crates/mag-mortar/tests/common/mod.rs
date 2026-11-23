//! Pools and jobs written by hand, for the loop's tests.

#![allow(dead_code)]

use mag_by::{
    Backend, Blob, Chain, Cost, DeviceId, Dtype, Edge, Graph, Input, Job, Mark, Member, Need,
    Offer, Part, Pin, Step, Tolerance, Unit,
};

pub fn member(name: &str, remote: bool) -> Member {
    Member {
        name: name.into(),
        remote,
        give_back_seconds: 5.0,
    }
}

pub fn device(member: &str, index: u32) -> DeviceId {
    DeviceId {
        member: member.into(),
        index,
    }
}

pub fn unit(name: &str) -> Unit {
    Unit {
        name: name.into(),
        version: 1,
    }
}

/// A device offering `memory_gib`, f32 and bf16, and `costs` as (unit, seconds per step).
pub fn offer(d: DeviceId, backend: Backend, memory_gib: u64, costs: &[(&str, f64)]) -> Offer {
    Offer {
        device: d,
        backend,
        memory_bytes: memory_gib << 30,
        tolerances: vec![Tolerance::F32, Tolerance::Bf16],
        dtypes: vec![Dtype::F32, Dtype::Bf16, Dtype::I32],
        costs: costs
            .iter()
            .map(|&(u, s)| Cost {
                unit: unit(u),
                seconds_per_step: s,
                band: 0.05,
                estimate: false,
            })
            .collect(),
    }
}

pub fn part(name: &str, unit_name: &str, mark: Mark, memory_gib: u64) -> Part {
    Part {
        name: name.into(),
        unit: unit(unit_name),
        mark,
        step: Step::Item,
        memory_bytes: memory_gib << 30,
        weights: vec![],
        tolerance: Tolerance::Bf16,
        pin: None,
        note: None,
    }
}

pub fn edge(from: &str, to: Option<&str>) -> Edge {
    Edge {
        from: from.into(),
        to: to.map(Into::into),
        dtype: Dtype::Bf16,
        bytes: 1 << 20,
        release: false,
        note: None,
    }
}

pub fn job(
    name: &str,
    priority: u32,
    inputs: Vec<Input>,
    parts: Vec<Part>,
    edges: Vec<Edge>,
) -> Graph {
    Graph {
        job: Job {
            name: name.into(),
            priority,
            need: Need::default(),
            note: None,
        },
        inputs,
        parts,
        chains: Vec::<Chain>::new(),
        edges,
        note: None,
    }
}

pub fn input(name: &str, local_only: bool) -> Input {
    Input {
        name: name.into(),
        bytes: 1 << 20,
        local_only,
    }
}

pub fn local_weights() -> Blob {
    Blob {
        hash: "trained-on-local-data".into(),
        bytes: 1 << 20,
        local_only: true,
    }
}

pub fn pinned(mut p: Part, pin: Pin) -> Part {
    p.pin = Some(pin);
    p
}
