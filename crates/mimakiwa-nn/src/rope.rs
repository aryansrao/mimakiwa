use burn::tensor::{backend::Backend, Tensor, TensorData};

/// Apply RoPE to query and key tensors (positions starting at 0).
/// q, k: [batch, n_heads, seq_len, head_dim]
pub fn apply_rope<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    rope_base: f32,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    apply_rope_offset(q, k, rope_base, 0, device)
}

/// Apply RoPE with an absolute position offset (for KV-cached decode).
/// Tokens are at positions offset..offset+seq_len in the full sequence.
pub fn apply_rope_offset<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    rope_base: f32,
    offset: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let q_rope = rope_rotate(q, rope_base, offset, device);
    let k_rope = rope_rotate(k, rope_base, offset, device);
    (q_rope, k_rope)
}

fn rope_rotate<B: Backend>(
    x: Tensor<B, 4>,
    rope_base: f32,
    offset: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [b, h, s, hd] = x.dims();
    let half = hd / 2;

    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0_f32 / rope_base.powf(2.0 * i as f32 / hd as f32))
        .collect();

    let n = b * h * s * half;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for _bi in 0..b {
        for _hi in 0..h {
            for si in 0..s {
                for fi in 0..half {
                    let angle = (si + offset) as f32 * freqs[fi];
                    cos_data.push(angle.cos());
                    sin_data.push(angle.sin());
                }
            }
        }
    }

    let cos_t = Tensor::<B, 4>::from_data(TensorData::new(cos_data, [b, h, s, half]), device);
    let sin_t = Tensor::<B, 4>::from_data(TensorData::new(sin_data, [b, h, s, half]), device);

    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
    let x2 = x.clone().slice([0..b, 0..h, 0..s, half..hd]);

    let out1 = x1.clone() * cos_t.clone() - x2.clone() * sin_t.clone();
    let out2 = x1 * sin_t + x2 * cos_t;

    Tensor::cat(vec![out1, out2], 3)
}
