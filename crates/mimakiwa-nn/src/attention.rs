use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor, TensorData, Bool, activation},
};
use crate::rope::{apply_rope, apply_rope_offset};

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_base: f32,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        embed_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        rope_base: f32,
        device: &B::Device,
    ) -> Self {
        assert_eq!(n_heads % n_kv_heads, 0);
        let head_dim = embed_dim / n_heads;
        Self {
            q_proj: LinearConfig::new(embed_dim, n_heads * head_dim).with_bias(false).init(device),
            k_proj: LinearConfig::new(embed_dim, n_kv_heads * head_dim).with_bias(false).init(device),
            v_proj: LinearConfig::new(embed_dim, n_kv_heads * head_dim).with_bias(false).init(device),
            o_proj: LinearConfig::new(n_heads * head_dim, embed_dim).with_bias(false).init(device),
            n_heads,
            n_kv_heads,
            head_dim,
            rope_base,
        }
    }

    /// Training forward — full sequence attention with causal mask.
    /// x: [batch, seq_len, embed_dim] → [batch, seq_len, embed_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _d] = x.dims();
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let hd = self.head_dim;
        let groups = nh / nkv;
        let scale = (hd as f32).sqrt().recip();
        let device = x.device();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q.reshape([b, s, nh, hd]).swap_dims(1, 2);
        let k = k.reshape([b, s, nkv, hd]).swap_dims(1, 2);
        let v = v.reshape([b, s, nkv, hd]).swap_dims(1, 2);

        let (q, k) = apply_rope(q, k, self.rope_base, &device);

        let k = if groups > 1 { k.repeat_dim(1, groups) } else { k };
        let v = if groups > 1 { v.repeat_dim(1, groups) } else { v };

        let scores = q.matmul(k.swap_dims(2, 3)) * scale;
        let causal_mask = build_causal_mask::<B>(b, nh, s, &device);
        let scores = scores.mask_fill(causal_mask, f32::NEG_INFINITY);

        let probs = activation::softmax(scores, 3);
        let out = probs.matmul(v);
        let out = out.swap_dims(1, 2).reshape([b, s, nh * hd]);
        self.o_proj.forward(out)
    }

    /// Inference forward with KV-cache support.
    ///
    /// Pass `past_kv = None` for prefill (full prompt). Pass `past_kv = Some(...)`
    /// for decode (single token per step). Returns (output, full_k, full_v) where
    /// full_k/v include the past cache plus the new entries.
    pub fn forward_kv(
        &self,
        x: Tensor<B, 3>,
        past_kv: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
        let past_len = past_kv.as_ref().map(|(k, _)| k.dims()[2]).unwrap_or(0);
        let [b, s, _d] = x.dims();
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let hd = self.head_dim;
        let groups = nh / nkv;
        let scale = (hd as f32).sqrt().recip();
        let device = x.device();

        let q = self.q_proj.forward(x.clone()).reshape([b, s, nh, hd]).swap_dims(1, 2);
        let k_new = self.k_proj.forward(x.clone()).reshape([b, s, nkv, hd]).swap_dims(1, 2);
        let v_new = self.v_proj.forward(x).reshape([b, s, nkv, hd]).swap_dims(1, 2);

        // RoPE with absolute position offset so decode positions are correct
        let (q, k_new) = apply_rope_offset(q, k_new, self.rope_base, past_len, &device);

        // Concatenate with cached KV
        let (k_all, v_all) = match past_kv {
            Some((pk, pv)) => (
                Tensor::cat(vec![pk, k_new], 2),
                Tensor::cat(vec![pv, v_new], 2),
            ),
            None => (k_new, v_new),
        };

        let k_exp = if groups > 1 { k_all.clone().repeat_dim(1, groups) } else { k_all.clone() };
        let v_exp = if groups > 1 { v_all.clone().repeat_dim(1, groups) } else { v_all.clone() };

        // [b, nh, s, total_kv_len]
        let scores = q.matmul(k_exp.swap_dims(2, 3)) * scale;

        // Causal mask for prefill only; decode (s=1) always attends to all past
        let scores = if s > 1 {
            let mask = build_causal_mask::<B>(b, nh, s, &device);
            scores.mask_fill(mask, f32::NEG_INFINITY)
        } else {
            scores
        };

        let probs = activation::softmax(scores, 3);
        let out = probs.matmul(v_exp).swap_dims(1, 2).reshape([b, s, nh * hd]);
        (self.o_proj.forward(out), k_all, v_all)
    }
}

fn build_causal_mask<B: Backend>(
    batch: usize,
    heads: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 4, Bool> {
    let n = batch * heads * seq_len * seq_len;
    let data: Vec<bool> = (0..n)
        .map(|idx| {
            let s2 = seq_len * seq_len;
            let pos = idx % s2;
            let row = pos / seq_len;
            let col = pos % seq_len;
            col > row
        })
        .collect();
    Tensor::<B, 4, Bool>::from_data(
        TensorData::new(data, [batch, heads, seq_len, seq_len]),
        device,
    )
}
