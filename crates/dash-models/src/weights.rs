//! Weights in and out of a module.
//!
//! On disk, weights are safetensors in PyTorch's convention: the module's parameter paths
//! with enum variants dropped (torchvision's names), linear weights as `[out, in]`, batch
//! norm as `weight`/`bias`. Such a file is also a PyTorch `state_dict` (less the
//! `num_batches_tracked` counters), so every tool reads it, and a Burn upgrade that stops
//! reading Burn's own records cannot strand it.
//!
//! Loading is strict: every parameter must be filled and every tensor in the file must
//! land somewhere. A partial load runs and answers, with whatever the initialiser left in
//! the gaps — the failure that looks like a bad model rather than a bad file.

use burn::store::{
    ApplyResult, BurnToPyTorchAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};

/// PyTorch counts batch-norm steps in a tensor; Burn keeps no such state.
fn ignorable(path: &str) -> bool {
    path.ends_with("num_batches_tracked")
}

fn check(result: ApplyResult) -> Result<(), String> {
    if !result.errors.is_empty() {
        return Err(format!("weights did not load: {:?}", result.errors));
    }
    if !result.missing.is_empty() {
        let missing: Vec<_> = result.missing.iter().map(|(path, _)| path).collect();
        return Err(format!(
            "parameters with no tensor in the file: {missing:?}"
        ));
    }
    let unused: Vec<_> = result.unused.iter().filter(|p| !ignorable(p)).collect();
    if !unused.is_empty() {
        return Err(format!("tensors in the file with no parameter: {unused:?}"));
    }
    Ok(())
}

/// The module's weights as safetensors bytes, in PyTorch's convention.
pub fn to_safetensors<M: ModuleSnapshot>(model: &M) -> Result<Vec<u8>, String> {
    let mut store = SafetensorsStore::from_bytes(None)
        .skip_enum_variants(true)
        .with_to_adapter(BurnToPyTorchAdapter);
    model.save_into(&mut store).map_err(|e| e.to_string())?;
    store.get_bytes().map_err(|e| e.to_string())
}

/// Fill `model` from safetensors bytes written by [`to_safetensors`] (or by PyTorch).
pub fn load_safetensors<M: ModuleSnapshot>(model: &mut M, bytes: Vec<u8>) -> Result<(), String> {
    let mut store = SafetensorsStore::from_bytes(Some(bytes))
        .skip_enum_variants(true)
        .with_from_adapter(PyTorchToBurnAdapter);
    check(model.load_from(&mut store).map_err(|e| e.to_string())?)
}

/// Fill `model` from a PyTorch checkpoint (`.pth`), read in Rust. `remap` renames tensors
/// first, as `(regex, replacement)` pairs, for checkpoints whose names differ from the
/// module's (torchvision's `classifier.0` and `classifier.3`, for one).
#[cfg(feature = "pytorch")]
pub fn load_pytorch<M: ModuleSnapshot>(
    model: &mut M,
    path: impl Into<std::path::PathBuf>,
    remap: &[(&str, &str)],
) -> Result<(), String> {
    // Indices stay as written: renumbering gaps would move torchvision's `classifier.3`.
    let mut store = burn::store::PytorchStore::from_file(path).map_indices_contiguous(false);
    for (from, to) in remap {
        store = store.with_key_remapping(*from, *to);
    }
    check(model.load_from(&mut store).map_err(|e| e.to_string())?)
}
