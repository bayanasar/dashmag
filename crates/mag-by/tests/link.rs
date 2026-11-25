//! The words of a link between two members, written as JSON and read back.

use mag_by::{Dtype, Open, Tensor, WIRE_VERSION};
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
