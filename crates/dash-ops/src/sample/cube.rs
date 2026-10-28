//! The kernel: one launch per call, a cube of up to 1024 units per row, which finds the cuts with
//! no sort of the row and no host round trip. Every pass below reads the row once, in vectors
//! along it, four vectors in flight per unit; each unit takes the vectors `unit, unit + units,
//! ...`, so it meets its tokens in increasing id.
//!
//! 1. **Statistics** (always): each unit's largest unmasked logit (the lowest id on ties), its
//!    lowest unmasked id, and the online sum `sum exp(z - m)` (for top-p alone, and for any cut
//!    at a temperature other than 1, also `sum exp((z - m) tau)`), rescaled once a vector where the
//!    vector raises the unit's maximum; the cube combines them into `m`, the argmax, `lse` and
//!    `W`, the tempered sum over `U` that `tempered_logprob` takes where a cut leaves `P` short of
//!    `U`. Greedy decoding ends here.
//! 2. **Candidates** (top-k or top-p): a lower bound on the cut, every token above which is
//!    gathered into shared memory (an atomic append per plane, then a sort, so that the order does
//!    not depend on the units' timing). For top-k, the `k`-th largest of the units' maxima: they
//!    are `k` distinct tokens, so the `k`-th largest logit is no smaller. For top-p alone, the
//!    tempered logit `ln((1 - p) W / |U|)` less a margin of 0.01: the tokens below it weigh less
//!    than `(1 - p) W` together, so the cut lies above it. The same pass sums the gathered tokens'
//!    weights in eight half-octave bands of `1 - s`; when they overflow the shared memory, the
//!    pass after gathers only the bands down to the one where the weight from the top reaches
//!    `p W`, or, when those too overflow (by less than eight times), first sums that band's weights
//!    in eight equal sub-bands and then gathers down to the sub-band where the weight reaches it.
//!    When the candidates fit ([`capacity`], 8192 on CUDA), the cube sorts them (bitonic,
//!    descending by logit, ascending by id), reads the `k`-th, scans the sorted weights for top-p
//!    and draws among the kept tokens there.
//! 3. **Refinement** (when they still do not fit: a flat distribution under top-p, a `k` above the
//!    units): a radix selection over the logits' order-preserving bits, three bits a pass, the
//!    eight sub-bands' counts (and weights) summed per unit and across the cube in a fixed order,
//!    until one key is left or the band holds no more than the capacity, which is then gathered and
//!    finished as in 2; first for the `k`-th logit, then, over what top-k keeps, for top-p's.
//! 4. **Gumbel pass** (the temperature alone, or after 3): every kept token's score, its words
//!    drawn four tokens to a Philox call; every unit starts from the argmax's score (the argmax is
//!    always kept), and a group of four whose best tempered logit is more than 23 below the score
//!    to beat (23 being above the largest `g`) draws no words. It also sums the kept weights: with
//!    nothing cut, over `U`, the one sum both tempered log-probabilities take.
//!
//! Every sum the cuts compare is formed in an order fixed by the device's plane width and the cube
//! size, and the shared-memory appends are sorted before use, so that a device gives the same bits
//! on every run. Each pass costs its unit's arithmetic more than its reads (on an RTX 5070 Ti a
//! pass takes as long over f32 logits as over bf16), and a row has one multiprocessor: a batch of
//! eight rows leaves most of a large GPU idle.
//!
//! The kernel asks of the device cubes of 256 to 1024 units of at most 32 planes, plane operations
//! (one partial result per plane in shared memory), 12 bytes of shared memory per candidate slot
//! beside 2 KiB of partial results, and stores of the logits' and the tokens' dtypes: CUDA has
//! them all; wgpu on lavapipe or OpenGL (no plane operations) and CubeCL's CPU runtime (as many
//! units per cube as cores) run the reference.

use super::{Cuts, SampleBackend, SampleOptions, probe};
use crate::cube::Requirements;
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::ops::IntTensorOps;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{DType, IntDType, Shape, TensorData, TensorMetadata};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::tensor_vector_size_parallel;
use std::sync::atomic::Ordering;

/// Partial results per reduction: one per plane, at most 32 planes to a cube.
const SLOTS: usize = 32;
/// The most units per cube, and the fewest.
const MAX_UNITS: u32 = 1024;
const MIN_UNITS: u32 = 256;
/// The largest candidate capacity: 96 KiB of shared memory.
const MAX_CAPACITY: usize = 8192;
/// Vectors each unit holds in flight in a pass over the row.
const UNROLL: usize = 4;
/// The half-octave bands of `1 - s` the first top-p gather sums its tokens' weights in; the last
/// takes everything from `s = -11` down.
const BANDS: usize = 8;
const _: () = assert!(
    BANDS == 8,
    "band_of and the kernel's last band are written for 8 bands"
);
/// An id no token has: the vocabulary is below 2^32 - 1.
const NONE: u32 = u32::MAX;

/// The order-preserving key of a logit: a larger logit has a larger key, and `-0` the key of `+0`.
#[cube]
fn key_of(x: f32) -> u32 {
    let raw = u32::reinterpret(x);
    let bits = select(raw == 0x8000_0000u32, 0u32, raw);
    select(bits >= 0x8000_0000u32, !bits, bits | 0x8000_0000u32)
}

/// The logit of a key.
#[cube]
fn value_of(key: u32) -> f32 {
    f32::reinterpret(select(key >= 0x8000_0000u32, key ^ 0x8000_0000u32, !key))
}

/// The band of a tempered logit `s <= 0`: `x = 1 - s` in half octaves, `[1, 1.5)` the first,
/// `[1.5, 2)` the second, `[2, 3)` the third and so on, from `x = 12` on the last. Non-decreasing
/// as `s` falls.
#[cube]
fn band_of(s: f32) -> u32 {
    min((u32::reinterpret(1.0f32 - s) >> 22u32) - 254u32, 7u32)
}

/// The mask bits of the vector whose first token is `first_id` (bit `e` for its element `e`): one
/// word, since a vector of `n` tokens starts at a multiple of `n` and `n` divides 32. 0 without a
/// mask.
#[cube]
fn mask_word(bits: &Tensor<u32>, first_id: u32, #[comptime] has_mask: bool) -> u32 {
    let mut word = 0u32;
    if has_mask {
        word = bits[(first_id >> 5u32) as usize] >> (first_id & 31u32);
    }
    word
}

/// Philox4x32-10 (the module `philox` holds the host's, with the known answers): the four words of
/// `counter` under `key`, the key bumped after every round.
#[cube]
fn philox(
    c0_in: u32,
    c1_in: u32,
    c2_in: u32,
    c3_in: u32,
    k0_in: u32,
    k1_in: u32,
) -> (u32, u32, u32, u32) {
    let mut c0 = c0_in;
    let mut c1 = c1_in;
    let mut c2 = c2_in;
    let mut c3 = c3_in;
    let mut k0 = k0_in;
    let mut k1 = k1_in;
    #[unroll]
    for _ in 0..10u32 {
        let hi0 = u32::mul_hi(0xD251_1F53u32, c0);
        let lo0 = 0xD251_1F53u32 * c0;
        let hi1 = u32::mul_hi(0xCD9E_8D57u32, c2);
        let lo1 = 0xCD9E_8D57u32 * c2;
        let n0 = hi1 ^ c1 ^ k0;
        let n2 = hi0 ^ c3 ^ k1;
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
        k0 += 0x9E37_79B9u32;
        k1 += 0xBB67_AE85u32;
    }
    (c0, c1, c2, c3)
}

/// The Gumbel variate of a word, as `dash_contract::sample` defines it, `E` in the module
/// documentation's two forms.
#[cube]
fn gumbel(x: u32) -> f32 {
    let upper = x >= 0x8000_0000u32;
    let y = select(upper, !x, x);
    // 2^-33, from its bits: exact whatever the compiler prints.
    let a = f32::cast_from((y << 1u32) | 1u32) * f32::reinterpret(0x2F00_0000u32);
    let e_lower = -(a.ln());
    // -ln_1p(-a) from the module documentation's two forms, not CubeCL's log1p. Each product
    // that meets a sum is exact, so a fused multiply-add changes no bit.
    let e_small = a * ((6.0f32 - a) / (6.0f32 - 4.0f32 * a));
    let w = 1.0f32 - a;
    let e_large = -(w.ln()) * (a / (1.0f32 - w));
    // 2^-6, from its bits.
    let e_upper = select(a <= f32::reinterpret(0x3C80_0000u32), e_small, e_large);
    -(select(upper, e_upper, e_lower).ln())
}

/// [`gumbel`] alone, a unit per word: what [`gumbel_variates`] runs.
#[cube(launch_unchecked)]
fn gumbel_variates_kernel(words: &[u32], out: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < words.len() {
        out[i] = gumbel(words[i]);
    }
}

/// The variates [`gumbel`] gives for `words` on the device `client` serves, from the function run
/// alone: a unit per word, which every CubeCL device runs, whether or not it takes the sampling
/// kernel. For the tests.
pub(crate) fn gumbel_variates(client: &Client, words: &[u32]) -> Vec<f32> {
    let n = words.len();
    if n == 0 {
        return Vec::new();
    }
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let input = client.create_from_slice(&bytes);
    let output = client.empty(4 * n);
    // SAFETY: unit `i < n` reads word `i` and writes variate `i`, of buffers of `n` each.
    unsafe {
        gumbel_variates_kernel::launch_unchecked(
            client,
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input, n),
            BufferArg::from_raw_parts(output.clone(), n),
        );
    }
    let bytes = client.read_one(output).expect("the variates read back");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// The generator's key and counter halves.
#[derive(CubeType, Clone, Copy)]
struct Draw {
    key_lo: u32,
    key_hi: u32,
    offset_lo: u32,
    offset_hi: u32,
}

/// The Gumbel variate of token `i` of row `row`.
#[cube]
fn token_gumbel(i: u32, row: u32, draw: &Draw) -> f32 {
    let (w0, w1, w2, w3) = philox(
        i >> 2u32,
        row,
        draw.offset_lo,
        draw.offset_hi,
        draw.key_lo,
        draw.key_hi,
    );
    let k = i & 3u32;
    gumbel(select(
        k == 0u32,
        w0,
        select(k == 1u32, w1, select(k == 2u32, w2, w3)),
    ))
}

/// The cube's shared memory: partial results, broadcast slots, the candidates and their weights,
/// and the append counter.
#[derive(CubeType)]
struct Smem {
    red_f: Shared<[f32]>,
    red_u: Shared<[u32]>,
    out_f: Shared<[f32]>,
    out_u: Shared<[u32]>,
    keys: Shared<[u32]>,
    ids: Shared<[u32]>,
    wts: Shared<[f32]>,
    count: Shared<[Atomic<u32>]>,
}

#[cube]
impl Smem {
    fn new(#[comptime] capacity: usize) -> Smem {
        Smem {
            red_f: Shared::<[f32]>::new_slice(comptime!(8 * SLOTS)),
            red_u: Shared::<[u32]>::new_slice(comptime!(8 * SLOTS)),
            out_f: Shared::<[f32]>::new_slice(8usize),
            out_u: Shared::<[u32]>::new_slice(16usize),
            keys: Shared::<[u32]>::new_slice(capacity),
            ids: Shared::<[u32]>::new_slice(capacity),
            wts: Shared::<[f32]>::new_slice(capacity),
            count: Shared::<[Atomic<u32>]>::new_slice(1usize),
        }
    }
}

/// The sum over the cube, the same in every unit.
#[cube]
fn sum_f32(x: f32, smem: &mut Smem, #[comptime] slot: usize) -> f32 {
    let plane = (UNIT_POS_X / PLANE_DIM) as usize;
    let p = plane_sum(x);
    if UNIT_POS_PLANE == 0u32 {
        smem.red_f[slot * SLOTS + plane] = p;
    }
    sync_cube();
    if UNIT_POS_X == 0u32 {
        let planes = (CUBE_DIM_X / PLANE_DIM) as usize;
        let mut total = 0.0f32;
        for k in 0..planes {
            total += smem.red_f[slot * SLOTS + k];
        }
        smem.out_f[slot] = total;
    }
    workgroup_uniform_load(&smem.out_f[slot])
}

/// The sum over the cube of a count, the same in every unit.
#[cube]
fn sum_u32(x: u32, smem: &mut Smem, #[comptime] slot: usize) -> u32 {
    let plane = (UNIT_POS_X / PLANE_DIM) as usize;
    let p = plane_sum(x);
    if UNIT_POS_PLANE == 0u32 {
        smem.red_u[slot * SLOTS + plane] = p;
    }
    sync_cube();
    if UNIT_POS_X == 0u32 {
        let planes = (CUBE_DIM_X / PLANE_DIM) as usize;
        let mut total = 0u32;
        for k in 0..planes {
            total += smem.red_u[slot * SLOTS + k];
        }
        smem.out_u[slot] = total;
    }
    workgroup_uniform_load(&smem.out_u[slot])
}

/// The minimum over the cube, the same in every unit.
#[cube]
fn min_u32(x: u32, smem: &mut Smem, #[comptime] slot: usize) -> u32 {
    let plane = (UNIT_POS_X / PLANE_DIM) as usize;
    let p = plane_min(x);
    if UNIT_POS_PLANE == 0u32 {
        smem.red_u[slot * SLOTS + plane] = p;
    }
    sync_cube();
    if UNIT_POS_X == 0u32 {
        let planes = (CUBE_DIM_X / PLANE_DIM) as usize;
        let mut low = 0xFFFF_FFFFu32;
        for k in 0..planes {
            low = min(low, smem.red_u[slot * SLOTS + k]);
        }
        smem.out_u[slot] = low;
    }
    workgroup_uniform_load(&smem.out_u[slot])
}

/// The largest `x` over the cube and the lowest `i` among the units holding it, the same in every
/// unit.
#[cube]
fn argmax(x: f32, i: u32, smem: &mut Smem, #[comptime] slot: usize) -> (f32, u32) {
    let plane = (UNIT_POS_X / PLANE_DIM) as usize;
    let top = plane_max(x);
    let at = plane_min(select(x == top, i, NONE));
    if UNIT_POS_PLANE == 0u32 {
        smem.red_f[slot * SLOTS + plane] = top;
        smem.red_u[slot * SLOTS + plane] = at;
    }
    sync_cube();
    if UNIT_POS_X == 0u32 {
        let planes = (CUBE_DIM_X / PLANE_DIM) as usize;
        let mut best = smem.red_f[slot * SLOTS];
        let mut best_at = smem.red_u[slot * SLOTS];
        for k in 1..planes {
            let value = smem.red_f[slot * SLOTS + k];
            let index = smem.red_u[slot * SLOTS + k];
            if value > best || (value == best && index < best_at) {
                best = value;
                best_at = index;
            }
        }
        smem.out_f[slot] = best;
        smem.out_u[slot] = best_at;
    }
    let best = workgroup_uniform_load(&smem.out_f[slot]);
    (best, workgroup_uniform_load(&smem.out_u[slot]))
}

/// Sorts `keys[0..n)` and `ids[0..n)` (`n` a power of two) by key, descending, and by id,
/// ascending, among equal keys: a bitonic network.
#[cube]
fn bitonic(smem: &mut Smem, n: u32) {
    let mut size = 2u32;
    while size <= n {
        let mut stride = size / 2u32;
        while stride > 0u32 {
            sync_cube();
            let mut t = UNIT_POS_X;
            while t < n / 2u32 {
                let lo = 2u32 * t - (t & (stride - 1u32));
                let hi = lo + stride;
                let (ka, kb) = (smem.keys[lo as usize], smem.keys[hi as usize]);
                let (ia, ib) = (smem.ids[lo as usize], smem.ids[hi as usize]);
                let a_first = ka > kb || (ka == kb && ia < ib);
                // The block holding `lo` ends in descending order where its `size` bit is clear.
                let descending = (lo & size) == 0u32;
                if a_first != descending {
                    smem.keys[lo as usize] = kb;
                    smem.keys[hi as usize] = ka;
                    smem.ids[lo as usize] = ib;
                    smem.ids[hi as usize] = ia;
                }
                t += CUBE_DIM_X;
            }
            stride /= 2u32;
        }
        size *= 2u32;
    }
    sync_cube();
}

/// Pads `keys`/`ids` from `n` to the next power of two with entries that sort last and sorts them.
#[cube]
fn sort_candidates(smem: &mut Smem, n: u32) {
    let mut padded = 1u32;
    while padded < n {
        padded *= 2u32;
    }
    let mut j = UNIT_POS_X + n;
    while j < padded {
        smem.keys[j as usize] = 0u32;
        smem.ids[j as usize] = NONE;
        j += CUBE_DIM_X;
    }
    bitonic(smem, padded);
}

/// `wts[0..n)` replaced by its inclusive prefix sums, in a fixed order: each unit sums a run of
/// `ceil(n / units)` entries in turn, and the runs' totals are summed across units in order.
#[cube]
fn scan(smem: &mut Smem, n: u32, #[comptime] slot: usize) {
    sync_cube();
    let per = n.div_ceil(CUBE_DIM_X);
    let start = min(UNIT_POS_X * per, n);
    let end = min(start + per, n);
    let mut run = 0.0f32;
    for j in start..end {
        run += smem.wts[j as usize];
        smem.wts[j as usize] = run;
    }
    let before_in_plane = plane_exclusive_sum(run);
    let plane_total = plane_sum(run);
    let plane = UNIT_POS_X / PLANE_DIM;
    if UNIT_POS_PLANE == 0u32 {
        smem.red_f[slot * SLOTS + plane as usize] = plane_total;
    }
    sync_cube();
    let mut before = 0.0f32;
    for k in 0..plane {
        before += smem.red_f[slot * SLOTS + k as usize];
    }
    let offset = before + before_in_plane;
    for j in start..end {
        smem.wts[j as usize] += offset;
    }
    sync_cube();
}

/// The row's place and what every pass over it needs.
#[derive(CubeType, Clone, Copy)]
struct Row {
    /// Its first vector.
    base: usize,
    /// Its vectors.
    vecs: usize,
    /// Its index.
    index: u32,
    /// Its largest unmasked logit.
    m: f32,
    /// The inverse temperature.
    tau: f32,
}

/// What a gather takes: unmasked tokens with `key >= floor`, `key & mask == lo` and
/// `(z - m) tau >= s_floor`; capped once, a band at most `band_cap`; capped twice, a band below
/// `band_cap` or that band's sub-band at most `sub_cap`.
#[derive(CubeType, Clone, Copy)]
struct Take {
    floor: u32,
    mask: u32,
    lo: u32,
    s_floor: f32,
    band_cap: u32,
    sub_cap: u32,
    /// Band `band_cap`'s sub-bands: eight equal steps of `x = 1 - s` from `x_lo`, `x_scale` of
    /// them to a unit of `x`.
    x_lo: f32,
    x_scale: f32,
}

/// The sub-band, 0 to 7, of a tempered logit `s` in band `take.band_cap`.
#[cube]
fn sub_of(s: f32, take: &Take) -> u32 {
    min(
        u32::cast_from(max((1.0f32 - s - take.x_lo) * take.x_scale, 0.0f32)),
        7u32,
    )
}

/// Whether a gather takes element `e` of a vector; `capped` as [`Take`] says.
#[cube]
fn takes(
    z: f32,
    e: u32,
    word: u32,
    valid: bool,
    row: &Row,
    take: &Take,
    #[comptime] capped: u32,
) -> bool {
    let key = key_of(z);
    let s = (z - row.m) * row.tau;
    let mut ok = valid
        && ((word >> e) & 1u32) == 0u32
        && key >= take.floor
        && (key & take.mask) == take.lo
        && s >= take.s_floor;
    if comptime!(capped == 1) {
        ok = ok && band_of(s) <= take.band_cap;
    }
    if comptime!(capped == 2) {
        let band = band_of(s);
        ok = ok
            && (band < take.band_cap || (band == take.band_cap && sub_of(s, take) <= take.sub_cap));
    }
    ok
}

/// Appends to `keys`/`ids` every token `take` takes, as far as the capacity goes, and returns how
/// many there are (beyond the capacity too). With `bands` 1, also sums their counts and weights by
/// band into `band_n` and `band_w`; with 2, those of band `take.band_cap` by sub-band. Every unit
/// runs the same steps, so that the plane operations (an exclusive sum and one atomic add per
/// plane and step) see whole planes.
#[cube]
#[allow(clippy::too_many_arguments)]
fn gather<E: Float, N: Size>(
    logits: &Tensor<Vector<E, N>>,
    bits: &Tensor<u32>,
    row: &Row,
    take: &Take,
    smem: &mut Smem,
    band_n: &mut Array<u32>,
    band_w: &mut Array<f32>,
    #[comptime] has_mask: bool,
    #[comptime] capped: u32,
    #[comptime] bands: u32,
    #[comptime] capacity: usize,
) -> u32 {
    let n = logits.vector_size();
    let units = CUBE_DIM_X as usize;
    if UNIT_POS_X == 0u32 {
        smem.count[0].store(0u32);
    }
    sync_cube();
    let steps = row.vecs.div_ceil(units * UNROLL);
    for step in 0..steps {
        let q0 = step * units * UNROLL + UNIT_POS_X as usize;
        let mut vectors = Array::<Vector<E, N>>::new(UNROLL);
        #[unroll]
        for u in 0..UNROLL {
            let q = q0 + u * units;
            vectors[u] = logits[row.base + select(q < row.vecs, q, 0usize)];
        }
        // Which of the step's tokens it takes, bit `u n + e`.
        let mut flags = 0u32;
        let mut mine = 0u32;
        #[unroll]
        for u in 0..UNROLL {
            let q = q0 + u * units;
            let word = mask_word(bits, (q * n) as u32, has_mask);
            #[unroll]
            for e in 0..n {
                let z = f32::cast_from(vectors[u].extract(e));
                if takes(z, e as u32, word, q < row.vecs, row, take, capped) {
                    flags |= comptime!(1u32 << (u * n + e) as u32);
                    mine += 1u32;
                    if comptime!(bands > 0) {
                        let s = (z - row.m) * row.tau;
                        let mut b = band_of(s);
                        if comptime!(bands == 2) {
                            // Only band `band_cap`, by sub-band; the others fall outside 0..8.
                            b = select(b == take.band_cap, sub_of(s, take), 8u32);
                        }
                        let w = s.exp();
                        #[unroll]
                        for j in 0..BANDS {
                            if b == j as u32 {
                                band_n[j] += 1u32;
                                band_w[j] += w;
                            }
                        }
                    }
                }
            }
        }
        let before = plane_exclusive_sum(mine);
        let total = plane_sum(mine);
        let mut first = 0u32;
        if UNIT_POS_PLANE == 0u32 && total > 0u32 {
            first = smem.count[0].fetch_add(total);
        }
        let mut slot = plane_broadcast(first, 0u32) + before;
        if flags != 0u32 {
            #[unroll]
            for u in 0..UNROLL {
                let q = q0 + u * units;
                #[unroll]
                for e in 0..n {
                    if (flags & comptime!(1u32 << (u * n + e) as u32)) != 0u32 {
                        if (slot as usize) < capacity {
                            smem.keys[slot as usize] =
                                key_of(f32::cast_from(vectors[u].extract(e)));
                            smem.ids[slot as usize] = (q * n + e) as u32;
                        }
                        slot += 1u32;
                    }
                }
            }
        }
    }
    workgroup_uniform_load_atomic(&smem.count[0])
}

/// The tempered weights of `keys[0..n)` into `wts`.
#[cube]
fn weigh(smem: &mut Smem, row: &Row, n: u32) {
    let mut j = UNIT_POS_X;
    while j < n {
        smem.wts[j as usize] = ((value_of(smem.keys[j as usize]) - row.m) * row.tau).exp();
        j += CUBE_DIM_X;
    }
    sync_cube();
}

/// How many of the sorted `keys[0..n)` are at least `key`.
#[cube]
fn count_at_least(smem: &mut Smem, n: u32, key: u32, #[comptime] slot: usize) -> u32 {
    let mut mine = 0u32;
    let mut j = UNIT_POS_X;
    while j < n {
        if smem.keys[j as usize] >= key {
            mine += 1u32;
        }
        j += CUBE_DIM_X;
    }
    sum_u32(mine, smem, slot)
}

/// The first `j < n` with `offset + wts[j] >= target`, `n - 1` if none reaches it (rounding).
#[cube]
fn first_reaching(
    smem: &mut Smem,
    n: u32,
    offset: f32,
    target: f32,
    #[comptime] slot: usize,
) -> u32 {
    let mut mine = 0xFFFF_FFFFu32;
    let mut j = UNIT_POS_X;
    while j < n {
        if offset + smem.wts[j as usize] >= target {
            mine = min(mine, j);
        }
        j += CUBE_DIM_X;
    }
    let first = min_u32(mine, smem, slot);
    select(first == NONE, n - 1u32, first)
}

/// The Gumbel-max draw among the sorted candidates `[0, n)`: the token, the same in every unit.
#[cube]
fn draw_candidates(smem: &mut Smem, row: &Row, draw: &Draw, n: u32, neg_inf: f32) -> u32 {
    let mut best = neg_inf;
    let mut best_at = 0xFFFF_FFFFu32;
    let mut j = UNIT_POS_X;
    while j < n {
        let i = smem.ids[j as usize];
        let s = (value_of(smem.keys[j as usize]) - row.m) * row.tau;
        let score = s + token_gumbel(i, row.index, draw);
        if score > best || (score == best && i < best_at) {
            best = score;
            best_at = i;
        }
        j += CUBE_DIM_X;
    }
    let (_, token) = argmax(best, best_at, smem, 4usize);
    token
}

/// From per-unit counts and weights by band (or sub-band), the cube's: the first band, from the
/// top, where `above_w` and the weight of the bands so far reach `target` (the last if none does,
/// by rounding); `above_n` and the count of the bands so far; and the weight above that band. The
/// count of the band itself is left in `out_u[15]`. The same in every unit.
#[cube]
fn band_cut(
    band_n: &Array<u32>,
    band_w: &Array<f32>,
    smem: &mut Smem,
    target: f32,
    above_w: f32,
    above_n: u32,
) -> (u32, u32, f32) {
    // Shared memory past the slots the reductions use: the candidates' arrays, free until the
    // next gather fills them.
    let plane = (UNIT_POS_X / PLANE_DIM) as usize;
    #[unroll]
    for b in 0..BANDS {
        let c = plane_sum(band_n[b]);
        let w = plane_sum(band_w[b]);
        if UNIT_POS_PLANE == 0u32 {
            smem.keys[b * SLOTS + plane] = c;
            smem.wts[b * SLOTS + plane] = w;
        }
    }
    sync_cube();
    if UNIT_POS_X == 0u32 {
        let planes = (CUBE_DIM_X / PLANE_DIM) as usize;
        // The last band unless the weight reaches the target sooner.
        let mut cut = 7u32;
        let mut count = above_n;
        let mut weight = above_w;
        let mut before = above_w;
        let mut last = 0u32;
        let mut open = true;
        for b in 0..BANDS {
            if open {
                let mut c = 0u32;
                let mut w = 0.0f32;
                for k in 0..planes {
                    c += smem.keys[b * SLOTS + k];
                    w += smem.wts[b * SLOTS + k];
                }
                before = weight;
                count += c;
                weight += w;
                last = c;
                if weight >= target {
                    cut = b as u32;
                    open = false;
                }
            }
        }
        smem.out_u[13] = cut;
        smem.out_u[14] = count;
        smem.out_u[15] = last;
        smem.out_f[5] = before;
    }
    let cut = workgroup_uniform_load(&smem.out_u[13]);
    let count = workgroup_uniform_load(&smem.out_u[14]);
    (cut, count, workgroup_uniform_load(&smem.out_f[5]))
}

/// The per-unit statistics of step 1.
#[derive(CubeType)]
struct Stats {
    /// The largest unmasked logit, and the lowest id holding it.
    mx: f32,
    ix: u32,
    /// The lowest unmasked id.
    first: u32,
    /// `sum exp(z - mx)` and `sum exp((z - mx) tau)`.
    s1: f32,
    sw: f32,
}

/// Step 1 over one vector.
#[cube]
#[allow(clippy::too_many_arguments)]
fn stats_vector<E: Float, N: Size>(
    vector: Vector<E, N>,
    first_id: u32,
    word: u32,
    st: &mut Stats,
    tau: f32,
    neg_inf: f32,
    #[comptime] has_mask: bool,
    #[comptime] with_w: bool,
) {
    let n = vector.vector_size();
    let mut top = neg_inf;
    let mut top_at = 0xFFFF_FFFFu32;
    #[unroll]
    for e in 0..n {
        let z = f32::cast_from(vector.extract(e));
        let unmasked = ((word >> e as u32) & 1u32) == 0u32;
        if unmasked {
            if has_mask {
                st.first = min(st.first, first_id + e as u32);
            }
            if z > top {
                top = z;
                top_at = first_id + e as u32;
            }
        }
    }
    if top > st.mx {
        st.s1 *= (st.mx - top).exp();
        if with_w {
            st.sw *= ((st.mx - top) * tau).exp();
        }
        st.mx = top;
        st.ix = top_at;
    }
    #[unroll]
    for e in 0..n {
        let z = f32::cast_from(vector.extract(e));
        let unmasked = ((word >> e as u32) & 1u32) == 0u32;
        if unmasked && z > neg_inf {
            st.s1 += (z - st.mx).exp();
            if with_w {
                st.sw += ((z - st.mx) * tau).exp();
            }
        }
    }
}

/// Step 3's counts (and weights) of one vector's tokens in the band `key & mask == lo`, by their
/// next digit (`key >> shift & digit`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn radix_vector<E: Float, N: Size>(
    vector: Vector<E, N>,
    word: u32,
    row: &Row,
    floor: u32,
    mask: u32,
    lo: u32,
    shift: u32,
    digit: u32,
    counts: &mut Array<u32>,
    weights: &mut Array<f32>,
    #[comptime] mass: bool,
) {
    let n = vector.vector_size();
    #[unroll]
    for e in 0..n {
        let z = f32::cast_from(vector.extract(e));
        let key = key_of(z);
        if ((word >> e as u32) & 1u32) == 0u32 && key >= floor && (key & mask) == lo {
            let sub = (key >> shift) & digit;
            let mut w = 0.0f32;
            if mass {
                w = ((z - row.m) * row.tau).exp();
            }
            // Registers, not an indexed array: one add per sub-band, each predicated.
            #[unroll]
            for j in 0..8u32 {
                if sub == j {
                    counts[j as usize] += 1u32;
                    if mass {
                        weights[j as usize] += w;
                    }
                }
            }
        }
    }
}

/// The radix selection of step 3 over the unmasked tokens with `key >= floor`: in count mode the
/// key of the `target_count`-th largest; in mass mode (`mass`) the largest key whose tokens and
/// those above it weigh at least `fraction` of the total (`total`, or the tokens' own when it is
/// negative).
#[cube]
#[allow(clippy::too_many_arguments)]
fn radix<E: Float, N: Size>(
    logits: &Tensor<Vector<E, N>>,
    bits: &Tensor<u32>,
    row: &Row,
    floor: u32,
    target_count: u32,
    fraction: f32,
    total: f32,
    smem: &mut Smem,
    neg_inf: f32,
    #[comptime] mass: bool,
    #[comptime] has_mask: bool,
    #[comptime] capacity: usize,
) -> u32 {
    let n = logits.vector_size();
    let units = CUBE_DIM_X as usize;
    let plane = (UNIT_POS_X / PLANE_DIM) as usize;
    let mut lo = 0u32;
    let mut mask = 0u32;
    let mut done = 0u32;
    let mut above_n = 0u32;
    let mut above_w = 0.0f32;
    let mut target_w = total * fraction;
    let mut first_pass = true;
    let mut result = 0u32;
    let mut found = false;
    let mut unused_n = Array::<u32>::new(1usize);
    let mut unused_w = Array::<f32>::new(1usize);
    while !found {
        let width = min(3u32, 32u32 - done);
        let shift = 32u32 - done - width;
        let digit = (1u32 << width) - 1u32;
        let mut counts = Array::<u32>::new(8usize);
        let mut weights = Array::<f32>::new(8usize);
        #[unroll]
        for j in 0..8usize {
            counts[j] = 0u32;
            weights[j] = 0.0f32;
        }
        let mut q = UNIT_POS_X as usize;
        while q + (UNROLL - 1) * units < row.vecs {
            let mut vectors = Array::<Vector<E, N>>::new(UNROLL);
            #[unroll]
            for u in 0..UNROLL {
                vectors[u] = logits[row.base + q + u * units];
            }
            #[unroll]
            for u in 0..UNROLL {
                let word = mask_word(bits, ((q + u * units) * n) as u32, has_mask);
                radix_vector(
                    vectors[u],
                    word,
                    row,
                    floor,
                    mask,
                    lo,
                    shift,
                    digit,
                    &mut counts,
                    &mut weights,
                    mass,
                );
            }
            q += UNROLL * units;
        }
        while q < row.vecs {
            let word = mask_word(bits, (q * n) as u32, has_mask);
            radix_vector(
                logits[row.base + q],
                word,
                row,
                floor,
                mask,
                lo,
                shift,
                digit,
                &mut counts,
                &mut weights,
                mass,
            );
            q += units;
        }
        #[unroll]
        for j in 0..8usize {
            let c = plane_sum(counts[j]);
            let mut w = 0.0f32;
            if mass {
                w = plane_sum(weights[j]);
            }
            if UNIT_POS_PLANE == 0u32 {
                smem.red_u[j * SLOTS + plane] = c;
                smem.red_f[j * SLOTS + plane] = w;
            }
        }
        sync_cube();
        // One unit decides; the others take its decision through shared memory.
        if UNIT_POS_X == 0u32 {
            let planes = (CUBE_DIM_X / PLANE_DIM) as usize;
            let mut c_sub = Array::<u32>::new(8usize);
            let mut w_sub = Array::<f32>::new(8usize);
            let mut w_all = 0.0f32;
            #[unroll]
            for j in 0..8usize {
                let mut c = 0u32;
                let mut w = 0.0f32;
                for k in 0..planes {
                    c += smem.red_u[j * SLOTS + k];
                    w += smem.red_f[j * SLOTS + k];
                }
                c_sub[j] = c;
                w_sub[j] = w;
                w_all += w;
            }
            if mass && first_pass && total < 0.0f32 {
                target_w = w_all * fraction;
            }
            // The sub-band, from the top, where the count or the weight reaches the target.
            let mut chosen = 8u32;
            let mut acc_n = above_n;
            let mut acc_w = above_w;
            let mut lowest = 8u32;
            let mut lowest_n = above_n;
            let mut lowest_w = above_w;
            #[unroll]
            for jj in 0..8u32 {
                let j = 7u32 - jj;
                let c = c_sub[j as usize];
                let w = w_sub[j as usize];
                if chosen == 8u32 && c > 0u32 {
                    let reach = select(mass, acc_w + w >= target_w, acc_n + c >= target_count);
                    if reach {
                        chosen = j;
                    } else {
                        lowest = j;
                        lowest_n = acc_n;
                        lowest_w = acc_w;
                        acc_n += c;
                        acc_w += w;
                    }
                }
            }
            if chosen == 8u32 {
                // Only rounding leaves the weight short of the target: take the lowest sub-band.
                chosen = lowest;
                acc_n = lowest_n;
                acc_w = lowest_w;
            }
            smem.out_u[8] = lo | (chosen << shift);
            smem.out_u[9] = mask | (digit << shift);
            smem.out_u[10] = done + width;
            smem.out_u[11] = acc_n;
            smem.out_u[12] = c_sub[chosen as usize];
            smem.out_f[6] = acc_w;
            smem.out_f[7] = target_w;
        }
        lo = workgroup_uniform_load(&smem.out_u[8]);
        mask = workgroup_uniform_load(&smem.out_u[9]);
        done = workgroup_uniform_load(&smem.out_u[10]);
        above_n = workgroup_uniform_load(&smem.out_u[11]);
        let band = workgroup_uniform_load(&smem.out_u[12]);
        above_w = workgroup_uniform_load(&smem.out_f[6]);
        target_w = workgroup_uniform_load(&smem.out_f[7]);
        first_pass = false;
        if done == 32u32 {
            result = lo;
            found = true;
        } else if (band as usize) <= capacity {
            // The band fits: gather it and finish in shared memory.
            let take = Take {
                floor,
                mask,
                lo,
                s_floor: neg_inf,
                band_cap: 0u32,
                sub_cap: 0u32,
                x_lo: 0.0f32,
                x_scale: 0.0f32,
            };
            let gathered = gather(
                logits,
                bits,
                row,
                &take,
                smem,
                &mut unused_n,
                &mut unused_w,
                has_mask,
                0u32,
                0u32,
                capacity,
            );
            sort_candidates(smem, gathered);
            if mass {
                weigh(smem, row, gathered);
                scan(smem, gathered, 5usize);
                let at = first_reaching(smem, gathered, above_w, target_w, 6usize);
                result = workgroup_uniform_load(&smem.keys[at as usize]);
            } else {
                result =
                    workgroup_uniform_load(&smem.keys[(target_count - above_n - 1u32) as usize]);
            }
            found = true;
        }
    }
    result
}

/// The best score of step 4 so far, its token, and the kept weights' sum.
#[derive(CubeType)]
struct Best {
    score: f32,
    at: u32,
    total: f32,
}

/// Step 4 over one vector's tokens with `key >= floor`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn gumbel_vector<E: Float, N: Size>(
    vector: Vector<E, N>,
    first_id: u32,
    word: u32,
    row: &Row,
    draw: &Draw,
    floor: u32,
    best: &mut Best,
) {
    let n = vector.vector_size();
    if comptime!(n.is_multiple_of(4)) {
        #[unroll]
        for group in 0..comptime!(n / 4) {
            let i0 = first_id + (4 * group) as u32;
            let mut s = Array::<f32>::new(4usize);
            let mut kept = Array::<u32>::new(4usize);
            let mut top = best.score - 24.0f32;
            #[unroll]
            for t in 0..4usize {
                let e = 4 * group + t;
                let z = f32::cast_from(vector.extract(e));
                let keep = ((word >> e as u32) & 1u32) == 0u32 && key_of(z) >= floor;
                let st = (z - row.m) * row.tau;
                s[t] = st;
                kept[t] = select(keep, 1u32, 0u32);
                if keep {
                    best.total += st.exp();
                    top = max(top, st);
                }
            }
            if top + 23.0f32 >= best.score {
                let (w0, w1, w2, w3) = philox(
                    i0 >> 2u32,
                    row.index,
                    draw.offset_lo,
                    draw.offset_hi,
                    draw.key_lo,
                    draw.key_hi,
                );
                #[unroll]
                for t in 0..4u32 {
                    let word = select(
                        t == 0u32,
                        w0,
                        select(t == 1u32, w1, select(t == 2u32, w2, w3)),
                    );
                    if kept[t as usize] == 1u32 {
                        let score = s[t as usize] + gumbel(word);
                        let i = i0 + t;
                        if score > best.score || (score == best.score && i < best.at) {
                            best.score = score;
                            best.at = i;
                        }
                    }
                }
            }
        }
    } else {
        #[unroll]
        for e in 0..n {
            let i = first_id + e as u32;
            let z = f32::cast_from(vector.extract(e));
            if ((word >> e as u32) & 1u32) == 0u32 && key_of(z) >= floor {
                let st = (z - row.m) * row.tau;
                best.total += st.exp();
                if st + 23.0f32 >= best.score {
                    let score = st + token_gumbel(i, row.index, draw);
                    if score > best.score || (score == best.score && i < best.at) {
                        best.score = score;
                        best.at = i;
                    }
                }
            }
        }
    }
}

/// Step 4 over the unmasked tokens with `key >= floor`: `(token, sum of their weights)`, the same
/// in every unit. Every unit starts from the score of the argmax `top` (kept whatever the cuts, its
/// tempered logit 0), so that a group of tokens that cannot beat it skips its words.
#[cube]
#[allow(clippy::too_many_arguments)]
fn gumbel_pass<E: Float, N: Size>(
    logits: &Tensor<Vector<E, N>>,
    bits: &Tensor<u32>,
    row: &Row,
    draw: &Draw,
    floor: u32,
    top: u32,
    smem: &mut Smem,
    #[comptime] has_mask: bool,
) -> (u32, f32) {
    let n = logits.vector_size();
    let units = CUBE_DIM_X as usize;
    let mut best = Best {
        score: token_gumbel(top, row.index, draw),
        at: top,
        total: 0.0f32,
    };
    let mut q = UNIT_POS_X as usize;
    while q + (UNROLL - 1) * units < row.vecs {
        let mut vectors = Array::<Vector<E, N>>::new(UNROLL);
        #[unroll]
        for u in 0..UNROLL {
            vectors[u] = logits[row.base + q + u * units];
        }
        #[unroll]
        for u in 0..UNROLL {
            let first_id = ((q + u * units) * n) as u32;
            let word = mask_word(bits, first_id, has_mask);
            gumbel_vector(vectors[u], first_id, word, row, draw, floor, &mut best);
        }
        q += UNROLL * units;
    }
    while q < row.vecs {
        let first_id = (q * n) as u32;
        let word = mask_word(bits, first_id, has_mask);
        gumbel_vector(
            logits[row.base + q],
            first_id,
            word,
            row,
            draw,
            floor,
            &mut best,
        );
        q += units;
    }
    let total = sum_f32(best.total, smem, 3usize);
    let (_, token) = argmax(best.score, best.at, smem, 4usize);
    (token, total)
}

/// One cube per row of `logits` `[B, V / N]` (vectors of `N` along the contiguous rows): the
/// token, its log-probability under the model, under the model at the temperature and under the
/// distribution drawn from. `with_w` asks step 1 for the tempered sum over `U`.
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::collapsible_if,
    reason = "conditions known when the kernel compiles stay apart from those known when it runs"
)]
fn sample_kernel<E: Float, I: Int, N: Size>(
    logits: &Tensor<Vector<E, N>>,
    bits: &Tensor<u32>,
    tokens: &mut Tensor<I>,
    logprob: &mut Tensor<f32>,
    tempered: &mut Tensor<f32>,
    sampling: &mut Tensor<f32>,
    key_lo: u32,
    key_hi: u32,
    offset_lo: u32,
    offset_hi: u32,
    unmasked: u32,
    top_k: u32,
    top_p: f32,
    tau: f32,
    neg_inf: f32,
    nan: f32,
    #[comptime] greedy: bool,
    #[comptime] cut_k: bool,
    #[comptime] cut_p: bool,
    #[comptime] has_mask: bool,
    #[comptime] with_w: bool,
    #[comptime] capacity: usize,
    #[define(E, I)] _dtypes: [ElemType; 2],
) {
    let n = logits.vector_size();
    let units = CUBE_DIM_X as usize;
    let unit = UNIT_POS_X;
    let index = CUBE_POS_X;
    let vecs = logits.shape(1) / n;
    let base = index as usize * (logits.stride(0) / n);
    let mut smem = Smem::new(capacity);
    let draw = Draw {
        key_lo,
        key_hi,
        offset_lo,
        offset_hi,
    };

    // 1. Statistics.
    let mut st = Stats {
        mx: neg_inf,
        ix: 0xFFFF_FFFFu32,
        first: 0xFFFF_FFFFu32,
        s1: 0.0f32,
        sw: 0.0f32,
    };
    let mut q = unit as usize;
    while q + (UNROLL - 1) * units < vecs {
        let mut vectors = Array::<Vector<E, N>>::new(UNROLL);
        #[unroll]
        for u in 0..UNROLL {
            vectors[u] = logits[base + q + u * units];
        }
        #[unroll]
        for u in 0..UNROLL {
            let first_id = ((q + u * units) * n) as u32;
            let word = mask_word(bits, first_id, has_mask);
            stats_vector(
                vectors[u], first_id, word, &mut st, tau, neg_inf, has_mask, with_w,
            );
        }
        q += UNROLL * units;
    }
    while q < vecs {
        let first_id = (q * n) as u32;
        let word = mask_word(bits, first_id, has_mask);
        stats_vector(
            logits[base + q],
            first_id,
            word,
            &mut st,
            tau,
            neg_inf,
            has_mask,
            with_w,
        );
        q += units;
    }
    let (m, arg) = argmax(st.mx, st.ix, &mut smem, 0usize);
    let mut lowest = 0u32;
    if has_mask {
        lowest = min_u32(st.first, &mut smem, 1usize);
    }
    let s1 = sum_f32(st.s1 * (st.mx - m).exp(), &mut smem, 2usize);
    let lse = m + s1.ln();
    let degenerate = m == neg_inf;
    let row = Row {
        base,
        vecs,
        index,
        m,
        tau,
    };

    let mut token = arg;
    let mut kept_total = 1.0f32;
    let mut w_total = 0.0f32;
    if degenerate {
        token = lowest;
    } else if !greedy {
        if with_w {
            w_total = sum_f32(st.sw * ((st.mx - m) * tau).exp(), &mut smem, 3usize);
        }
        if cut_k || cut_p {
            // 2. Candidates.
            let mut take = Take {
                floor: 0u32,
                mask: 0u32,
                lo: 0u32,
                s_floor: neg_inf,
                band_cap: 0u32,
                sub_cap: 0u32,
                x_lo: 0.0f32,
                x_scale: 0.0f32,
            };
            let mut band_n = Array::<u32>::new(BANDS);
            let mut band_w = Array::<f32>::new(BANDS);
            #[unroll]
            for b in 0..BANDS {
                band_n[b] = 0u32;
                band_w[b] = 0.0f32;
            }
            if cut_k {
                if top_k <= CUBE_DIM_X {
                    // A unit without an unmasked token offers the lowest key.
                    let mine = select(
                        unit_has_token(&st, unit, vecs, has_mask),
                        key_of(st.mx),
                        0u32,
                    );
                    smem.keys[unit as usize] = mine;
                    smem.ids[unit as usize] = unit;
                    bitonic(&mut smem, CUBE_DIM_X);
                    take.floor = workgroup_uniform_load(&smem.keys[(top_k - 1u32) as usize]);
                }
            } else {
                take.s_floor =
                    ((1.0f32 - top_p) * w_total / f32::cast_from(unmasked)).ln() - 0.01f32;
            }
            let mut found = gather(
                logits,
                bits,
                &row,
                &take,
                &mut smem,
                &mut band_n,
                &mut band_w,
                has_mask,
                0u32,
                comptime!(if cut_k { 0u32 } else { 1u32 }),
                capacity,
            );
            // Nested, not one condition: the outer one is known when the kernel compiles, the
            // inner one only when it runs.
            if !cut_k {
                if (found as usize) > capacity {
                    // The bands down to the one where the weight from the top reaches p W.
                    let target = w_total * top_p;
                    let (cap, narrowed, above_w) =
                        band_cut(&band_n, &band_w, &mut smem, target, 0.0f32, 0u32);
                    take.band_cap = cap;
                    if (narrowed as usize) <= capacity {
                        found = gather(
                            logits,
                            bits,
                            &row,
                            &take,
                            &mut smem,
                            &mut band_n,
                            &mut band_w,
                            has_mask,
                            1u32,
                            0u32,
                            capacity,
                        );
                    } else if (narrowed as usize) <= 8 * capacity {
                        // That band in eight sub-bands, the bands above it whole: a band of more
                        // than eight capacities' tokens would not fit by a sub-band either, on a
                        // spread of logits like the one that filled it.
                        let above_n = narrowed - workgroup_uniform_load(&smem.out_u[15]);
                        take.x_lo = f32::reinterpret((254u32 + cap) << 22u32);
                        let mut x_hi = f32::reinterpret((255u32 + cap) << 22u32);
                        if cap == 7u32 {
                            x_hi = 1.0f32 - take.s_floor;
                        }
                        take.x_scale = 8.0f32 / (x_hi - take.x_lo);
                        #[unroll]
                        for b in 0..BANDS {
                            band_n[b] = 0u32;
                            band_w[b] = 0.0f32;
                        }
                        let _ = gather(
                            logits,
                            bits,
                            &row,
                            &take,
                            &mut smem,
                            &mut band_n,
                            &mut band_w,
                            has_mask,
                            1u32,
                            2u32,
                            capacity,
                        );
                        let (sub, narrowed, _) =
                            band_cut(&band_n, &band_w, &mut smem, target, above_w, above_n);
                        take.sub_cap = sub;
                        if (narrowed as usize) <= capacity {
                            found = gather(
                                logits,
                                bits,
                                &row,
                                &take,
                                &mut smem,
                                &mut band_n,
                                &mut band_w,
                                has_mask,
                                2u32,
                                0u32,
                                capacity,
                            );
                        }
                    }
                }
            }
            if (found as usize) <= capacity {
                sort_candidates(&mut smem, found);
                let mut kept = found;
                if cut_k {
                    let kth = workgroup_uniform_load(&smem.keys[(top_k - 1u32) as usize]);
                    kept = count_at_least(&mut smem, found, kth, 5usize);
                }
                weigh(&mut smem, &row, kept);
                if cut_p {
                    scan(&mut smem, kept, 5usize);
                    let mut total = w_total;
                    if cut_k {
                        total = workgroup_uniform_load(&smem.wts[(kept - 1u32) as usize]);
                    }
                    let at = first_reaching(&mut smem, kept, 0.0f32, total * top_p, 6usize);
                    let cut = workgroup_uniform_load(&smem.keys[at as usize]);
                    kept = count_at_least(&mut smem, kept, cut, 7usize);
                    kept_total = workgroup_uniform_load(&smem.wts[(kept - 1u32) as usize]);
                } else {
                    let mut mine = 0.0f32;
                    let mut j = unit;
                    while j < kept {
                        mine += smem.wts[j as usize];
                        j += CUBE_DIM_X;
                    }
                    kept_total = sum_f32(mine, &mut smem, 3usize);
                }
                token = draw_candidates(&mut smem, &row, &draw, kept, neg_inf);
            } else {
                // 3. Refinement, then 4.
                let mut cut = 0u32;
                if cut_k {
                    cut = radix(
                        logits, bits, &row, take.floor, top_k, 0.0f32, 0.0f32, &mut smem, neg_inf,
                        false, has_mask, capacity,
                    );
                }
                if cut_p {
                    let mut total = w_total;
                    if cut_k {
                        total = -1.0f32;
                    }
                    cut = radix(
                        logits, bits, &row, cut, 0u32, top_p, total, &mut smem, neg_inf, true,
                        has_mask, capacity,
                    );
                }
                let (t, total) =
                    gumbel_pass(logits, bits, &row, &draw, cut, arg, &mut smem, has_mask);
                token = t;
                kept_total = total;
            }
        } else {
            // 4. The temperature alone.
            let (t, total) = gumbel_pass(logits, bits, &row, &draw, 0u32, arg, &mut smem, has_mask);
            token = t;
            kept_total = total;
        }
    }

    if unit == 0u32 {
        let z_t = f32::cast_from(
            logits[base + (token as usize) / n].extract_dynamic((token as usize) % n),
        );
        tokens[index as usize] = I::cast_from(token);
        if degenerate {
            logprob[index as usize] = nan;
            tempered[index as usize] = nan;
            sampling[index as usize] = nan;
        } else {
            let model = z_t - lse;
            logprob[index as usize] = model;
            if greedy {
                tempered[index as usize] = 0.0f32;
                sampling[index as usize] = 0.0f32;
            } else {
                // Over U: `logprob` at tau = 1; else step 1's tempered sum where a cut leaves P
                // short of U, and with nothing cut step 4's, which P = U makes the same.
                let mut over_u = model;
                if tau != 1.0f32 {
                    let mut total = kept_total;
                    if cut_k || cut_p {
                        total = w_total;
                    }
                    over_u = (z_t - m) * tau - total.ln();
                }
                tempered[index as usize] = over_u;
                if cut_k || cut_p {
                    sampling[index as usize] = (z_t - m) * tau - kept_total.ln();
                } else {
                    sampling[index as usize] = over_u;
                }
            }
        }
    }
}

/// Whether the unit met an unmasked token in step 1: its lowest unmasked id is set (with a mask),
/// or it read a vector (without one, every token is unmasked).
#[cube]
fn unit_has_token(st: &Stats, unit: u32, vecs: usize, #[comptime] has_mask: bool) -> bool {
    let mut has = (unit as usize) < vecs;
    if has_mask {
        has = st.first != NONE;
    }
    has
}

/// The units per cube the kernel takes on the device `client` serves: as many as it allows, up to
/// 1024 and 32 planes, a power of two.
fn units(client: &Client) -> u32 {
    let hardware = &client.properties().hardware;
    let most = hardware
        .max_units_per_cube
        .min(hardware.max_cube_dim.0)
        .min(MAX_UNITS)
        .min(SLOTS as u32 * hardware.plane_size_min.max(1));
    if most == 0 { 0 } else { 1 << most.ilog2() }
}

/// The bytes of shared memory the kernel holds with `capacity` candidate slots.
fn shared_bytes(capacity: usize) -> usize {
    4 * (16 * SLOTS + 8 + 16 + 1 + 3 * capacity)
}

/// The candidate slots of the kernel's fast path on the device `client` serves: the largest power
/// of two up to 8192 that the shared memory holds, at least the units per cube (and the bands'
/// partial results); 0 if none.
pub(crate) fn capacity(client: &Client) -> usize {
    let units = units(client) as usize;
    let shared = client.properties().hardware.max_shared_memory_size;
    let mut capacity = MAX_CAPACITY;
    while capacity >= units.max(BANDS * SLOTS) {
        if shared_bytes(capacity) <= shared {
            return capacity;
        }
        capacity /= 2;
    }
    0
}

/// What the kernel asks for `b` rows, read in vectors of `vector` elements.
fn requirements(client: &Client, b: usize, vector: usize) -> Option<Requirements> {
    let units = units(client);
    let capacity = capacity(client);
    (units >= MIN_UNITS && capacity > 0).then(|| Requirements {
        shared_bytes: shared_bytes(capacity),
        vector_size: vector,
        min_plane_width: Some(units / SLOTS as u32),
        ..Requirements::new((units, 1), (b, 1), 6)
    })
}

/// Whether the kernel takes logits `[B, V]` in `dtype` with tokens in `int` on the device `client`
/// serves: it indexes them with 32-bit offsets.
pub(crate) fn kernel_takes(client: &Client, dtype: DType, int: DType, [b, v]: [usize; 2]) -> bool {
    v > 0
        && b.checked_mul(v).is_some_and(|len| len < NONE as usize)
        && matches!(dtype, DType::F32 | DType::BF16 | DType::F16)
        && requirements(client, b, 1)
            .is_some_and(|r| r.met_on(client, &[dtype, int, DType::U32, DType::F32]))
}

impl SampleBackend for CubeBackend {
    fn sample(
        logits: FloatTensor<Self>,
        options: SampleOptions,
        out_dtype: IntDType,
    ) -> (
        IntTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let shape = logits.shape();
        assert_eq!(
            shape.num_dims(),
            2,
            "sample: logits must be [B, V], got {shape:?}"
        );
        let [b, v] = shape.dims::<2>();
        let client = logits.client.clone();
        if !kernel_takes(&client, logits.dtype, out_dtype.into(), [b, v]) {
            return super::reference::sample::<Self>(logits, &options, out_dtype);
        }
        let device = logits.device.clone();
        let empty = |shape: Shape, dtype: DType| {
            empty_device_contiguous_dtype(client.clone(), device.clone(), shape, dtype)
        };
        let tokens = empty(Shape::new([b]), out_dtype.into());
        let logprob = empty(Shape::new([b]), DType::F32);
        let tempered = empty(Shape::new([b]), DType::F32);
        let sampling = empty(Shape::new([b]), DType::F32);
        if b == 0 {
            return (tokens, logprob, tempered, sampling);
        }
        let logits = into_contiguous(logits);
        let dtype = logits.dtype;
        let vector = tensor_vector_size_parallel(
            client.io_optimized_vector_sizes(dtype.size()),
            logits.meta.shape(),
            logits.meta.strides(),
            1,
        );
        let requirements = requirements(&client, b, vector).expect("the kernel takes it");
        if !requirements.met_on(&client, &[dtype, out_dtype.into(), DType::U32, DType::F32]) {
            return super::reference::sample::<Self>(logits, &options, out_dtype);
        }
        let units = units(&client);
        let capacity = capacity(&client);
        let has_mask = !options.masked.is_empty();
        let mut words = vec![0u32; if has_mask { v.div_ceil(32) } else { 1 }];
        for &id in &options.masked {
            words[id as usize / 32] |= 1 << (id % 32);
        }
        let bits =
            CubeBackend::int_from_data(TensorData::new(words.clone(), [words.len()]), &device);
        let unmasked = v - options.masked.len();
        let greedy = options.is_greedy();
        let cut_k = !greedy && options.top_k_cut(unmasked).is_some();
        let cut_p = !greedy && options.top_p_cut().is_some();
        let tau = options.inverse_temperature();
        // The tempered sum over U in step 1: top-p alone compares it, and where a cut leaves P
        // short of U `tempered_logprob` takes it, but at tau = 1, where it is `logprob`.
        let with_w = (cut_p && !cut_k) || ((cut_k || cut_p) && tau != 1.0);
        let (key, offset) = super::philox::key_and_offset(options.seed, options.offset);
        probe::KERNEL_LAUNCHES.fetch_add(1, Ordering::SeqCst);
        // SAFETY: each cube reads its row's vectors below `V / N` (the logits are contiguous and
        // `N` divides `V`), mask words below `ceil(V / 32)` (only with a mask), candidate slots
        // below the capacity, and writes its row's four outputs; the shapes come from the tensors.
        unsafe {
            sample_kernel::launch_unchecked(
                &client,
                CubeCount::Static(b as u32, 1, 1),
                CubeDim::new_1d(units),
                vector,
                logits.into_tensor_arg(),
                bits.into_tensor_arg(),
                tokens.clone().into_tensor_arg(),
                logprob.clone().into_tensor_arg(),
                tempered.clone().into_tensor_arg(),
                sampling.clone().into_tensor_arg(),
                key[0],
                key[1],
                offset[0],
                offset[1],
                unmasked as u32,
                options.top_k.min(u32::MAX as usize) as u32,
                options.top_p,
                tau,
                f32::NEG_INFINITY,
                f32::NAN,
                greedy,
                cut_k,
                cut_p,
                has_mask,
                with_w,
                capacity,
                [
                    dtype_to_storage_type(dtype),
                    dtype_to_storage_type(out_dtype.into()),
                ],
            );
        }
        (tokens, logprob, tempered, sampling)
    }
}
