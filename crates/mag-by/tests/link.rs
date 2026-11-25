//! The words of a link between two members, written as JSON and read back.

use mag_by::{
    Backend, CONTROL, Control, DeviceId, Dtype, Graph, Member, Offer, Open, Refusal, Tensor,
    Tolerance, Trust, WIRE_VERSION,
};
use serde_json::{Value, json};

fn round_trip<T>(value: &T, expected: Value)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    assert_eq!(serde_json::to_value(value).unwrap(), expected);
    assert_eq!(&serde_json::from_value::<T>(expected).unwrap(), value);
}

#[test]
fn the_wire_has_its_own_version() {
    assert_eq!(WIRE_VERSION, 1);
}

#[test]
fn an_opening_names_its_wire_version_and_its_job() {
    round_trip(
        &Open {
            wire: WIRE_VERSION,
            job: "label".into(),
        },
        json!({ "wire": 1, "job": "label" }),
    );
    for bad in [
        json!({ "job": "label" }),
        json!({ "wire": 1 }),
        json!({ "wire": 1, "job": "label", "member": "laptop" }),
    ] {
        assert!(serde_json::from_value::<Open>(bad).is_err());
    }
}

#[test]
fn a_tensor_names_its_edge_step_dtype_shape_and_bytes() {
    let into_part = Tensor {
        from: "b0".into(),
        to: Some("b1".into()),
        step: 7,
        dtype: Dtype::Bf16,
        shape: vec![2, 3],
        bytes: 12,
    };
    round_trip(
        &into_part,
        json!({ "from": "b0", "to": "b1", "step": 7, "dtype": "bf16", "shape": [2, 3],
                "bytes": 12 }),
    );
    // A result of the job has no `to`.
    let result = Tensor {
        to: None,
        ..into_part
    };
    round_trip(
        &result,
        json!({ "from": "b0", "step": 7, "dtype": "bf16", "shape": [2, 3], "bytes": 12 }),
    );
    for bad in [
        json!({ "from": "b0", "to": null, "step": 7, "dtype": "bf16", "shape": [], "bytes": 0 }),
        json!({ "from": "b0", "step": 7, "dtype": "bf16", "shape": [] }),
        json!({ "from": "b0", "step": -1, "dtype": "bf16", "shape": [], "bytes": 0 }),
        json!({ "from": "b0", "step": 0, "dtype": "bf16", "shape": [], "bytes": 0, "data": [] }),
    ] {
        assert!(serde_json::from_value::<Tensor>(bad).is_err());
    }
}

#[test]
fn the_control_channel_is_for_no_job_a_name_no_job_may_have() {
    assert_eq!(CONTROL, "");
    round_trip(
        &Open {
            wire: WIRE_VERSION,
            job: CONTROL.into(),
        },
        json!({ "wire": 1, "job": "" }),
    );
    let mut graph = Graph::from_json(include_str!("../graphs/example.json")).unwrap();
    assert_eq!(graph.check(), Ok(()));
    graph.job.name = CONTROL.into();
    assert_eq!(graph.check(), Err(Refusal::EmptyName { what: "job" }));
}

#[test]
fn a_member_joins_with_itself_and_its_offers_and_leaves() {
    let join = Control::Join {
        member: Member {
            name: "laptop".into(),
            trust: Trust::Site,
            give_back_seconds: 2.0,
        },
        offers: vec![Offer {
            device: DeviceId {
                member: "laptop".into(),
                index: 0,
            },
            backend: Backend::Flex,
            memory_bytes: 1 << 30,
            tolerances: vec![Tolerance::Exact],
            dtypes: vec![Dtype::F32],
            costs: vec![],
        }],
    };
    round_trip(
        &join,
        json!({ "join": {
            "member": { "name": "laptop", "trust": "site", "give_back_seconds": 2.0 },
            "offers": [{
                "device": { "member": "laptop", "index": 0 },
                "backend": "flex",
                "memory_bytes": 1073741824u64,
                "tolerances": ["exact"],
                "dtypes": ["f32"],
                "costs": []
            }]
        } }),
    );
    round_trip(&Control::Leave, json!("leave"));
    for bad in [
        json!({ "join": { "offers": [] } }),
        json!({ "join": {
            "member": { "name": "laptop", "trust": "site", "give_back_seconds": 2.0 },
            "offers": [], "address": "192.0.2.1:7340"
        } }),
        json!("join"),
        json!({ "leave": {} }),
        json!("stay"),
    ] {
        assert!(
            serde_json::from_value::<Control>(bad.clone()).is_err(),
            "{bad}"
        );
    }
}
