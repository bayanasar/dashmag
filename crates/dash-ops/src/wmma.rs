//! AMD RDNA4's 16 x 16 x 16 matrix products from bf16 or f16 into f32, as the kernels written for
//! them find them (`attention/wmma.rs`, `quant/wmma.rs`): CubeCL offers the product on RDNA3 as
//! well, in another register layout, so a kernel takes a device only where the layout read from
//! it, through `position_of_nth`, is RDNA4's: lane `l` holds, of a 16 x 16 A (`m` x `k`), row
//! `l % 16` and the eight columns `8 (l / 16)..8 (l / 16) + 8`; of a B (`k` x `n`), column `l % 16`
//! and the eight rows `8 (l / 16)..`; of the f32 accumulator (`m` x `n`), column `l % 16` and the
//! eight rows `8 (l / 16)..`.

use burn::backend::DType;
use burn::backend::cubecl::{dtype_to_elem_type, dtype_to_storage_type};
use cubecl::client::Client;
use cubecl::cmma::{MatrixIdent, MmaDefinition};
use cubecl::ir::features::MmaConfig;
use cubecl::ir::{DeviceProperties, ElemType, FloatKind};
use cubecl::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

/// The plane width RDNA4's layout is given for.
const PLANE: u32 = 32;

/// Whether the device `client` serves has a 16 x 16 x 16 product from `dtype` into f32 in RDNA4's
/// register layout.
pub(crate) fn rdna4_products(client: &Client, dtype: DType) -> bool {
    matches!(dtype, DType::BF16 | DType::F16)
        && mma_16x16x16(client.properties(), dtype)
        && wmma_layout(client, dtype)
}

/// Whether the device has a 16 x 16 x 16 MMA from `dtype` into f32.
fn mma_16x16x16(properties: &DeviceProperties, dtype: DType) -> bool {
    let elem = dtype_to_elem_type(dtype);
    properties.features.matmul.mma.contains(&MmaConfig {
        a_type: elem,
        b_type: elem,
        cd_type: ElemType::Float(FloatKind::F32),
        m: 16,
        n: 16,
        k: 16,
    })
}

/// The row and column of element `e` of lane `lane` in RDNA4's layout, of A (`ident` 0), B (1)
/// and the accumulator (2).
fn wmma_expected(ident: usize, lane: u32, e: u32) -> (u32, u32) {
    let (index, half) = (lane % 16, 8 * (lane / 16) + e);
    match ident {
        0 => (index, half),
        _ => (half, index),
    }
}

/// Whether the 16 x 16 x 16 MMA's register layout, as `position_of_nth` gives it on the device
/// `client` serves, is RDNA4's, which the kernels on it are written for (RDNA3's differs).
/// Read from each device, by [`layout_kernel`], the first time a kernel asks in each dtype, and
/// kept for that device and dtype: a process with an RDNA4 card and an RDNA3 one reads both.
fn wmma_layout(client: &Client, dtype: DType) -> bool {
    // The device and its runtime, as CubeCL names its service, and the dtype.
    static LAYOUTS: Mutex<Option<HashMap<(String, DType), bool>>> = Mutex::new(None);
    let key = (format!("{:?}", client.service_id()), dtype);
    if let Some(&known) = LAYOUTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .get(&key)
    {
        return known;
    }
    let matches = read_wmma_layout(client, dtype);
    LAYOUTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(key, matches);
    matches
}

/// [`wmma_layout`]'s reading: [`layout_kernel`] on the device, its positions against
/// [`wmma_expected`].
fn read_wmma_layout(client: &Client, dtype: DType) -> bool {
    let n = 32 * 3 * 8 * 2;
    let pos = client.empty(4 * n);
    // SAFETY: one plane of 32 units; lane `l` writes elements `48 l..48 l + 48` of `n`.
    unsafe {
        layout_kernel::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(PLANE),
            BufferArg::from_raw_parts(pos.clone(), n),
            dtype_to_storage_type(dtype),
        );
    }
    let Ok(bytes) = client.read_one(pos) else {
        return false;
    };
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    (0..32u32).all(|lane| {
        (0..3).all(|ident| {
            (0..8u32).all(|e| {
                let at = (((lane as usize * 3 + ident) * 8) + e as usize) * 2;
                (words[at], words[at + 1]) == wmma_expected(ident, lane, e)
            })
        })
    })
}

/// Lane `l` of one plane writes, for A, B and the accumulator of the 16 x 16 x 16 MMA in that
/// order, the row and column of each of its eight elements to `pos[((3 l + ident) 8 + e) 2..]`
/// (rows and columns as `position_of_nth` gives them): what the kernels written for RDNA4's
/// layout are checked against.
#[cube(launch_unchecked)]
fn layout_kernel<E: Float>(pos: &mut [u32], #[define(E)] _dtype: ElemType) {
    let def = MmaDefinition::<E, E, f32>::new(16usize, 16usize, 16usize);
    let lane = UNIT_POS_PLANE;
    #[unroll]
    for n in 0..8u32 {
        let (row, col) = def.position_of_nth(lane, n, MatrixIdent::A);
        let at = ((lane * 3) * 8 + n) as usize * 2;
        pos[at] = row;
        pos[at + 1] = col;
        let (row, col) = def.position_of_nth(lane, n, MatrixIdent::B);
        let at = ((lane * 3 + 1) * 8 + n) as usize * 2;
        pos[at] = row;
        pos[at + 1] = col;
        let (row, col) = def.position_of_nth(lane, n, MatrixIdent::Accumulator);
        let at = ((lane * 3 + 2) * 8 + n) as usize * 2;
        pos[at] = row;
        pos[at + 1] = col;
    }
}
