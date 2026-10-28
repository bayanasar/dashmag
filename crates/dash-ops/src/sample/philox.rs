//! Philox4x32-10 on the host, as the reference draws its words; the kernel computes the same
//! rounds on the device. Salmon, Moraes, Dror and Shaw, "Parallel random numbers: as easy as 1, 2,
//! 3" (SC 2011), and their Random123 library, whose known-answer vectors the tests below check.

/// The round multipliers.
pub(crate) const M0: u32 = 0xD251_1F53;
pub(crate) const M1: u32 = 0xCD9E_8D57;
/// The Weyl sequence that bumps the key between rounds.
pub(crate) const W0: u32 = 0x9E37_79B9;
pub(crate) const W1: u32 = 0xBB67_AE85;

/// Four words for `counter` under `key`: ten rounds, the key bumped by `(W0, W1)` before each but
/// the first.
pub(crate) fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = counter;
    let mut k = key;
    for round in 0..10 {
        if round > 0 {
            k = [k[0].wrapping_add(W0), k[1].wrapping_add(W1)];
        }
        let p0 = u64::from(M0) * u64::from(c[0]);
        let p1 = u64::from(M1) * u64::from(c[2]);
        c = [
            (p1 >> 32) as u32 ^ c[1] ^ k[0],
            p1 as u32,
            (p0 >> 32) as u32 ^ c[3] ^ k[1],
            p0 as u32,
        ];
    }
    c
}

/// The key and the upper half of the counter for `seed` and `offset`, as the module documentation
/// gives them: `(seed lo, seed hi)` and `(offset lo, offset hi)`.
pub(crate) fn key_and_offset(seed: u64, offset: u64) -> ([u32; 2], [u32; 2]) {
    (
        [seed as u32, (seed >> 32) as u32],
        [offset as u32, (offset >> 32) as u32],
    )
}

/// The value the Gumbel variate is computed from for the word `x`, signed by its branch: `a` for
/// `x < 2^31` (where `E = -ln a`), `-a` otherwise (where `E = -ln_1p(-a)`), with
/// `a = (2y + 1) / 2^33` rounded to f32 and `y` the word or its complement (the module
/// documentation).
pub(crate) fn gumbel_input(x: u32) -> f32 {
    let upper = x >> 31 == 1;
    let y = if upper { !x } else { x };
    // (2y + 1) < 2^32 rounds to f32 to nearest, ties to even, as the device's conversion does;
    // the scaling by 2^-33 is exact.
    let a = ((y << 1) | 1) as f32 * f32::from_bits(0x2F00_0000);
    if upper { -a } else { a }
}

/// The Gumbel inputs of row `row` of a `[B, V]` sample: one per token, `[V]`.
pub(crate) fn row_inputs(seed: u64, offset: u64, row: u32, v: usize) -> Vec<f32> {
    let (key, [lo, hi]) = key_and_offset(seed, offset);
    let mut out = Vec::with_capacity(v);
    for group in 0..v.div_ceil(4) {
        let words = philox4x32_10([group as u32, row, lo, hi], key);
        for word in words.iter().take((v - 4 * group).min(4)) {
            out.push(gumbel_input(*word));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random123's known-answer vectors for Philox4x32-10 (`kat_vectors`).
    #[test]
    fn philox_gives_random123s_known_answers() {
        let cases: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
            (
                [0; 4],
                [0; 2],
                [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
            ),
            (
                [u32::MAX; 4],
                [u32::MAX; 2],
                [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
            ),
            (
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0],
                [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
            ),
        ];
        for (counter, key, want) in cases {
            assert_eq!(philox4x32_10(counter, key), want, "{counter:x?} {key:x?}");
        }
    }

    /// The scale is 2^-33, and the inputs stay within (0, 1/2] in magnitude at both ends.
    #[test]
    fn gumbel_inputs_stay_inside_the_open_interval() {
        assert_eq!(f32::from_bits(0x2F00_0000), 2f32.powi(-33));
        for x in [0, 1, (1 << 31) - 1, 1 << 31, u32::MAX - 1, u32::MAX] {
            let a = gumbel_input(x).abs();
            assert!(a > 0.0 && a <= 0.5, "{x:#x}: {a}");
        }
        assert_eq!(gumbel_input(0), 2f32.powi(-33));
        assert_eq!(gumbel_input(u32::MAX), -(2f32.powi(-33)));
    }
}
