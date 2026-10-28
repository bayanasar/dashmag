//! The modules training uses, on any Burn device, written to the recipe a PyTorch run used.
//!
//! Each part of the recipe is reproduced as PyTorch computes it and checked against values
//! recorded from PyTorch, so that a model trained with these modules differs from one trained
//! by PyTorch by the noise of training and not by a different recipe:
//! - [`loss`]: cross-entropy per head, class-weighted, masked targets dropped, reduced as
//!   PyTorch's weighted mean; the heads with a target in the batch averaged.
//! - [`optim`]: AdamW as `torch.optim.AdamW` computes it, with beta1 set per step; with
//!   the schedules attached, held to PyTorch over a whole schedule.
//! - [`schedule`]: `OneCycleLR` — learning rate per parameter group, and beta1.
//! - [`clip`]: gradients clipped by their global norm, as `torch.nn.utils.clip_grad_norm_`
//!   clips them for PPO and GRPO: one norm over every gradient a module's parameters have,
//!   accumulated in f32 and returned as a tensor, with no read back to the host.
//! - [`metrics`]: macro-F1 per head on validation, rounded as the recipe rounds it before
//!   it picks the best epoch.
//! - [`calibrate`]: one class's logit shifted by the value best on validation, folded
//!   into its head's bias.
//!
//! The crate still also holds a training loop for multi-head classifiers built on these
//! modules: [`train`], with [`train::predict`]; [`data`] and [`cache`], which read and write
//! the decoded frames it trains on; [`calibrate::calibrate`], which calibrates a trained run
//! on a cache; and the `dash-train` command line. A training loop is the application's, not
//! the library's: these parts are moving to the application that uses them.
//!
//! Not reproduced, and stated:
//! - Random streams. Shuffles, crops, flips, dropout masks and the heads' initial weights
//!   come from this crate's generators and Burn's, not PyTorch's, so a seed here is not
//!   the same run as that seed there. Comparisons are between sets of seeds.
//! - The crop's resize. torchvision resizes a `uint8` crop with its own antialiased
//!   bilinear filter; here it is `dash-image`'s, which is Pillow's.
//! - A bf16 gradient's norm. torch rounds each gradient's norm, the total and the clipping
//!   coefficient to bf16; [`clip`] keeps them in f32 and rounds only the coefficient, for the
//!   multiplication.
//! - Batch norm's running variance. Burn keeps the biased batch variance, PyTorch the
//!   unbiased one; they differ by a factor n / (n - 1), n the values per channel in a
//!   batch — at least 6272 for MobileNetV3-Small at batch 128 and 224 x 224, so 1.6e-4
//!   at most.

pub mod cache;
pub mod calibrate;
pub mod clip;
pub mod data;
pub mod loss;
pub mod metrics;
pub mod optim;
pub mod rng;
pub mod schedule;
pub mod train;
