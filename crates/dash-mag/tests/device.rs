//! Choosing the device by its index: before the first tensor, the device in use again, and what is
//! refused. Its own test binary, so that no other test has made a tensor first.

use dash_mag::{DType, Tensor, use_device};

#[test]
fn the_device_is_chosen_before_the_first_tensor() {
    // On Flex and MLX there is one device; on CUDA and ROCm a second card may not be there, and
    // choosing it would fail at the first tensor, so this chooses device 0 there.
    if cfg!(any(feature = "cuda", feature = "rocm")) {
        use_device(0).unwrap();
    } else {
        let why = use_device(1).expect_err("one device").0;
        assert!(why.starts_with("use_device: "), "{why}");
        assert!(why.contains("has one device, 0; asked for 1"), "{why}");
        use_device(0).unwrap();
    }
    let t = Tensor::ones(&[2], DType::F32).unwrap();
    assert_eq!(t.to_f32().unwrap(), [1.0, 1.0]);
    // The device in use, again: nothing. Another: refused.
    use_device(0).unwrap();
    let why = use_device(1).expect_err("in use").0;
    assert!(why.starts_with("use_device: "), "{why}");
}
