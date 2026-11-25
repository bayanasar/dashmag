//! A training loop for multi-head classifiers on Burn modules, written to the recipe a PyTorch run
//! used, and the Burn-tensor modules it still needs.
//!
//! The modules training uses are `dash-mag`'s, each written once and held there to values recorded
//! from PyTorch: the multi-head loss (`dash_mag::loss`), AdamW and clipping (`dash_mag::optim`),
//! `OneCycleLR` (`dash_mag::schedule`), macro-F1 (`dash_mag::metrics`) and the choice of a
//! calibration shift (`dash_mag::calibrate`). What is host arithmetic, with no tensor, this crate
//! takes from there. The loop here trains a Burn `Module`, which `dash-mag`'s tensors do not reach,
//! so until it moves onto `dash-mag` it keeps Burn-tensor versions of two of them, each reproduced
//! as PyTorch computes it and checked against PyTorch's values:
//! - [`loss`]: cross-entropy per head, class-weighted, masked targets dropped, reduced as
//!   PyTorch's weighted mean; the heads with a target in the batch averaged.
//! - [`optim`]: AdamW as `torch.optim.AdamW` computes it, with beta1 set per step; with
//!   the schedules attached, held to PyTorch over a whole schedule.
//! - [`schedule`]: `OneCycleLR` as a Burn scheduler, over `dash_mag::schedule::OneCycle`.
//!
//! The rest is the loop: [`train`], with [`train::predict`]; [`data`] and [`cache`], which read
//! and write the decoded frames it trains on; [`calibrate::calibrate`], which calibrates a trained
//! run on a cache; and the `dash-train` command line. A training loop is the application's, not
//! the library's: these parts are moving to the application that uses them.
//!
//! Not reproduced, and stated:
//! - Random streams. Shuffles, crops, flips, dropout masks and the heads' initial weights
//!   come from this crate's generators and Burn's, not PyTorch's, so a seed here is not
//!   the same run as that seed there. Comparisons are between sets of seeds.
//! - The crop's resize. torchvision resizes a `uint8` crop with its own antialiased
//!   bilinear filter; here it is `dash-image`'s, which is Pillow's.
//! - Batch norm's running variance. Burn keeps the biased batch variance, PyTorch the
//!   unbiased one; they differ by a factor n / (n - 1), n the values per channel in a
//!   batch — at least 6272 for MobileNetV3-Small at batch 128 and 224 x 224, so 1.6e-4
//!   at most.

pub mod cache;
pub mod calibrate;
pub mod data;
pub mod loss;
pub mod optim;
pub mod rng;
pub mod schedule;
pub mod train;
