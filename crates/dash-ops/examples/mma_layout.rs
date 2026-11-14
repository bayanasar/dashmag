//! The register layout of the GPU's manual 16 x 16 x 16 MMA from bf16 into f32, as CubeCL's
//! `position_of_nth` gives it, and one product checked against the host: which row and column of
//! A (16 x 16, `m` x `k`), B (`k` x `n`) and the accumulator (`m` x `n`) each lane's elements
//! hold. A kernel that keeps a product's accumulator in registers as the next product's operand
//! depends on these.
//!
//! `cargo run --release --example mma_layout --features rocm` (or `cuda`, which has no 16 x 16 x
//! 16 manual MMA and says so).

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
fn main() {
    eprintln!("mma_layout needs --features cuda or rocm");
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn main() {
    probe::run();
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
mod probe {
    use burn::backend::DType;
    use burn::backend::cubecl::dtype_to_storage_type;
    use burn::tensor::Device;
    use cubecl::cmma::{MatrixIdent, MmaDefinition};
    use cubecl::ir::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::prelude::*;

    /// Elements per lane of every fragment this probe reads: 16 x 16 over 32 lanes.
    const ELEMS: usize = 8;

    /// Lane `l` writes, for each of A, B and the accumulator in that order, the row and column
    /// of each of its elements to `pos[((l * 3 + ident) * 8 + e) * 2..][..2]`, and its elements
    /// of `a b` (A and B read from `a` and `b`, row-major 16 x 16) to `d[l * 8 + e]`.
    #[cube(launch_unchecked)]
    fn layout_kernel<E: Float>(
        a: &[E],
        b: &[E],
        pos: &mut [u32],
        d: &mut [f32],
        #[define(E)] _dtype: ElemType,
    ) {
        let def = MmaDefinition::<E, E, f32>::new(16usize, 16usize, 16usize);
        let lane = UNIT_POS_PLANE;

        let size_a = def.vector_size(MatrixIdent::A);
        let size!(NA) = size_a;
        let elems_a = def.elems_per_lane(MatrixIdent::A);
        let count_a = comptime!(elems_a / size_a);
        let mut reg_a = Array::<Vector<E, NA>>::new(count_a);
        #[unroll]
        for i in 0..count_a {
            let mut reg = Vector::empty();
            #[unroll]
            for e in 0..size_a {
                let n = (i * size_a + e) as u32;
                let (row, col) = def.position_of_nth(lane, n, MatrixIdent::A);
                reg.insert(e, a[(row * 16 + col) as usize]);
                let at = ((lane * 3) * 8 + n) as usize * 2;
                pos[at] = row;
                pos[at + 1] = col;
            }
            reg_a[i] = reg;
        }

        let size_b = def.vector_size(MatrixIdent::B);
        let size!(NB) = size_b;
        let elems_b = def.elems_per_lane(MatrixIdent::B);
        let count_b = comptime!(elems_b / size_b);
        let mut reg_b = Array::<Vector<E, NB>>::new(count_b);
        #[unroll]
        for i in 0..count_b {
            let mut reg = Vector::empty();
            #[unroll]
            for e in 0..size_b {
                let n = (i * size_b + e) as u32;
                let (row, col) = def.position_of_nth(lane, n, MatrixIdent::B);
                reg.insert(e, b[(row * 16 + col) as usize]);
                let at = ((lane * 3 + 1) * 8 + n) as usize * 2;
                pos[at] = row;
                pos[at + 1] = col;
            }
            reg_b[i] = reg;
        }

        let size_c = def.vector_size(MatrixIdent::Accumulator);
        let size!(NC) = size_c;
        let elems_c = def.elems_per_lane(MatrixIdent::Accumulator);
        let count_c = comptime!(elems_c / size_c);
        let mut reg_c = Array::<Vector<f32, NC>>::new(count_c);
        #[unroll]
        for i in 0..count_c {
            reg_c[i] = Vector::new(0.0f32);
        }
        let out = def.execute(&reg_a, &reg_b, &reg_c);
        #[unroll]
        for i in 0..count_c {
            #[unroll]
            for e in 0..size_c {
                let n = (i * size_c + e) as u32;
                let (row, col) = def.position_of_nth(lane, n, MatrixIdent::Accumulator);
                let at = ((lane * 3 + 2) * 8 + n) as usize * 2;
                pos[at] = row;
                pos[at + 1] = col;
                d[(lane * 8 + n) as usize] = out[i].extract(e);
            }
        }
    }

    /// The device's CubeCL client.
    fn client(device: &Device) -> cubecl::client::Client {
        fn of(device: &burn::backend::DispatchDevice) -> cubecl::client::Client {
            match device {
                burn::backend::DispatchDevice::Cube(device) => device.client(),
                burn::backend::DispatchDevice::Autodiff(device) => of(device),
                #[allow(unreachable_patterns)]
                _ => panic!("a CubeCL device"),
            }
        }
        of(device.as_dispatch())
    }

    fn gpu() -> Device {
        #[cfg(feature = "cuda")]
        return Device::cuda(0);
        #[cfg(not(feature = "cuda"))]
        Device::rocm(0)
    }

    fn a_value(m: usize, k: usize) -> f32 {
        ((m * 3 + k * 5) % 7) as f32 - 3.0
    }

    fn b_value(k: usize, n: usize) -> f32 {
        ((k * 2 + n) % 5) as f32 - 2.0
    }

    pub fn run() {
        let device = gpu();
        let client = client(&device);
        let bf16 = ElemType::Float(FloatKind::BF16);
        let config = MmaConfig {
            a_type: bf16,
            b_type: bf16,
            cd_type: ElemType::Float(FloatKind::F32),
            m: 16,
            n: 16,
            k: 16,
        };
        if !client.properties().features.matmul.mma.contains(&config) {
            println!("no manual 16 x 16 x 16 MMA from bf16 into f32 on this device");
            return;
        }
        let half = |f: fn(usize, usize) -> f32| -> Vec<u8> {
            (0..256)
                .flat_map(|i| {
                    burn::backend::bf16::from_f32(f(i / 16, i % 16))
                        .to_bits()
                        .to_le_bytes()
                })
                .collect()
        };
        let a = client.create_from_slice(&half(a_value));
        let b = client.create_from_slice(&half(b_value));
        let pos = client.empty(32 * 3 * ELEMS * 2 * 4);
        let d = client.empty(32 * ELEMS * 4);
        // SAFETY: one plane of 32; lane `l` writes `pos` and `d` at its own offsets, within the
        // buffers' 1536 and 256 elements, and reads `a` and `b` at rows and columns below 16.
        unsafe {
            layout_kernel::launch_unchecked(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(a, 256),
                BufferArg::from_raw_parts(b, 256),
                BufferArg::from_raw_parts(pos.clone(), 32 * 3 * ELEMS * 2),
                BufferArg::from_raw_parts(d.clone(), 32 * ELEMS),
                dtype_to_storage_type(DType::BF16),
            );
        }
        let words = |bytes: Vec<u8>| -> Vec<u32> {
            bytes
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect()
        };
        let pos = words(client.read_one(pos).expect("positions").to_vec());
        let d: Vec<f32> = words(client.read_one(d).expect("products").to_vec())
            .into_iter()
            .map(f32::from_bits)
            .collect();

        for (ident, name) in ["A (m, k)", "B (k, n)", "accumulator (m, n)"]
            .iter()
            .enumerate()
        {
            println!("{name}: lane: (row, col) of elements 0..8");
            for lane in 0..32 {
                let cells: Vec<String> = (0..ELEMS)
                    .map(|e| {
                        let at = ((lane * 3 + ident) * ELEMS + e) * 2;
                        format!("({},{})", pos[at], pos[at + 1])
                    })
                    .collect();
                println!("  {lane:2}: {}", cells.join(" "));
            }
        }

        // Each lane's products against the host's, at the accumulator positions it reported.
        let mut wrong = 0;
        let mut covered = vec![0u32; 256];
        for lane in 0..32 {
            for e in 0..ELEMS {
                let at = ((lane * 3 + 2) * ELEMS + e) * 2;
                let (m, n) = (pos[at] as usize, pos[at + 1] as usize);
                covered[m * 16 + n] += 1;
                let expected: f32 = (0..16).map(|k| a_value(m, k) * b_value(k, n)).sum();
                if d[lane * ELEMS + e] != expected {
                    wrong += 1;
                }
            }
        }
        let once = covered.iter().all(|&c| c == 1);
        println!("product: {wrong} of 256 wrong; each accumulator element held once: {once}");
    }
}
