use burn::{
    module::{Module, AutodiffModule},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor, Int, TensorData},
    record::CompactRecorder,
    backend::{Autodiff, Wgpu},
};
use mimakiwa_nn::{RMSNorm, Attention, SwiGLUMLP};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::path::Path;

// ── Backend types ─────────────────────────────────────────────────────────────

pub type InferBackend = Wgpu;
pub type TrainBackend = Autodiff<InferBackend>;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MimakiwaConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub ffn_dim: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
}

impl MimakiwaConfig {
    /// ~15M params — trains in ~20 min on M1 release mode
    pub fn compact(vocab_size: usize) -> Self {
        let d = 256;
        Self {
            vocab_size,
            embed_dim: d,
            n_layers: 6,
            n_heads: 8,
            n_kv_heads: 8,
            ffn_dim: d * 5 / 2,
            max_seq_len: 512,
            rope_base: 10000.0,
        }
    }

    /// ~85M params — trains in ~60 min on M1 release mode
    pub fn small(vocab_size: usize) -> Self {
        let d = 512;
        Self {
            vocab_size,
            embed_dim: d,
            n_layers: 8,
            n_heads: 8,
            n_kv_heads: 8,
            ffn_dim: d * 11 / 4,
            max_seq_len: 1024,
            rope_base: 10000.0,
        }
    }

    /// ~350M params — trains in ~4 hrs on M1 release mode
    pub fn medium(vocab_size: usize) -> Self {
        let d = 768;
        Self {
            vocab_size,
            embed_dim: d,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 12,
            ffn_dim: d * 8 / 3,
            max_seq_len: 2048,
            rope_base: 10000.0,
        }
    }

    pub fn param_count(&self) -> usize {
        let d = self.embed_dim;
        let v = self.vocab_size;
        let hd = d / self.n_heads;
        let nkv = self.n_kv_heads;
        let ffn = self.ffn_dim;
        let embed = v * d;
        let attn = d * d + d * (nkv * hd) * 2 + d * d;
        let mlp = d * ffn * 3;
        let norms = d * 2;
        let layer = attn + mlp + norms;
        embed + self.n_layers * layer + d + d * v
    }
}

// ── Transformer block ─────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    norm1: RMSNorm<B>,
    attn: Attention<B>,
    norm2: RMSNorm<B>,
    mlp: SwiGLUMLP<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(cfg: &MimakiwaConfig, device: &B::Device) -> Self {
        Self {
            norm1: RMSNorm::new(cfg.embed_dim, device),
            attn: Attention::new(cfg.embed_dim, cfg.n_heads, cfg.n_kv_heads, cfg.rope_base, device),
            norm2: RMSNorm::new(cfg.embed_dim, device),
            mlp: SwiGLUMLP::new(cfg.embed_dim, cfg.ffn_dim, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = x.clone() + self.attn.forward(self.norm1.forward(x));
        h.clone() + self.mlp.forward(self.norm2.forward(h))
    }

    /// KV-cached forward for inference. Returns (output, new_k, new_v).
    pub fn forward_kv(
        &self,
        x: Tensor<B, 3>,
        past_kv: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
        let normed = self.norm1.forward(x.clone());
        let (attn_out, k, v) = self.attn.forward_kv(normed, past_kv);
        let h = x + attn_out;
        let out = h.clone() + self.mlp.forward(self.norm2.forward(h));
        (out, k, v)
    }
}

// ── Inner model (Burn Module) ─────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct MimakiwaModelInner<B: Backend> {
    embed: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    norm: RMSNorm<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> MimakiwaModelInner<B> {
    pub fn new(cfg: &MimakiwaConfig, device: &B::Device) -> Self {
        Self {
            embed: EmbeddingConfig::new(cfg.vocab_size, cfg.embed_dim).init(device),
            layers: (0..cfg.n_layers)
                .map(|_| TransformerBlock::new(cfg, device))
                .collect(),
            norm: RMSNorm::new(cfg.embed_dim, device),
            lm_head: LinearConfig::new(cfg.embed_dim, cfg.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }

    /// Training forward — full sequence, no KV cache.
    /// ids: [batch, seq_len] → logits: [batch, seq_len, vocab_size]
    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embed.forward(ids);
        for layer in &self.layers {
            x = layer.forward(x);
        }
        self.lm_head.forward(self.norm.forward(x))
    }

    /// KV-cached forward for inference.
    /// ids: [1, s] — s=full_prompt for prefill, s=1 for decode.
    /// Returns (logits [1, s, vocab], updated_kv_per_layer).
    pub fn forward_kv(
        &self,
        ids: Tensor<B, 2, Int>,
        past_kvs: Option<Vec<(Tensor<B, 4>, Tensor<B, 4>)>>,
    ) -> (Tensor<B, 3>, Vec<(Tensor<B, 4>, Tensor<B, 4>)>) {
        let mut x = self.embed.forward(ids);
        let n = self.layers.len();
        let mut new_kvs = Vec::with_capacity(n);
        for (i, layer) in self.layers.iter().enumerate() {
            let past = past_kvs.as_ref().map(|v| v[i].clone());
            let (out, k, v) = layer.forward_kv(x, past);
            x = out;
            new_kvs.push((k, v));
        }
        let logits = self.lm_head.forward(self.norm.forward(x));
        (logits, new_kvs)
    }
}

// ── Generate output ───────────────────────────────────────────────────────────

pub struct GenerateOutput {
    pub tokens: Vec<u32>,
    pub confidence: f32,
}

// ── Public model wrapper ──────────────────────────────────────────────────────

pub struct MimakiwaModel {
    pub cfg: MimakiwaConfig,
    pub inner: Option<MimakiwaModelInner<TrainBackend>>,
    pub device: <TrainBackend as Backend>::Device,
}

impl MimakiwaModel {
    pub fn new(cfg: MimakiwaConfig) -> Self {
        let device = Default::default();
        eprintln!("[model] init {}M params on WGPU/Metal", cfg.param_count() / 1_000_000);
        let inner = MimakiwaModelInner::new(&cfg, &device);
        Self { cfg, inner: Some(inner), device }
    }

    pub fn n_params(&self) -> usize {
        self.cfg.param_count()
    }

    /// Generate tokens from a prompt using KV-cache for fast autoregressive decoding.
    pub fn generate(
        &self,
        ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> GenerateOutput {
        let inner = match self.inner.as_ref() {
            Some(m) => m,
            None => return GenerateOutput { tokens: vec![], confidence: 0.0 },
        };

        let infer = inner.valid();
        let device = &self.device;
        let max_seq = self.cfg.max_seq_len;
        let vocab = self.cfg.vocab_size;
        let nkv = self.cfg.n_kv_heads;
        let head_dim = self.cfg.embed_dim / self.cfg.n_heads;
        let temperature = temperature.max(0.01);
        // Repetition penalty: divide logit by this for already-seen tokens
        let rep_penalty = 1.3f32;

        // Reserve space in the context for the tokens we're about to generate
        let reserved = max_new_tokens.min(max_seq / 2);
        let max_prefill = max_seq.saturating_sub(reserved).max(1);

        let mut context: Vec<u32> = ids.to_vec();
        let mut out_tokens: Vec<u32> = Vec::new();
        let mut top_probs: Vec<f32> = Vec::new();

        // ── Prefill ───────────────────────────────────────────────────────────
        let prefill_len = context.len().min(max_prefill);
        let prefill_input: Vec<i32> = context[context.len() - prefill_len..]
            .iter()
            .map(|&x| x as i32)
            .collect();
        let prefill_t = Tensor::<InferBackend, 2, Int>::from_data(
            TensorData::new(prefill_input, [1, prefill_len]),
            device,
        );
        let (prefill_logits, mut past_kvs) = infer.forward_kv(prefill_t, None);

        // Sample first token from last prefill position
        let last_logit = prefill_logits
            .slice([0..1, (prefill_len - 1)..prefill_len, 0..vocab])
            .reshape([vocab]);

        let probs = logit_to_probs(last_logit, &out_tokens, rep_penalty, temperature);
        if probs.is_empty() {
            return GenerateOutput { tokens: out_tokens, confidence: 0.5 };
        }
        let tok = sample_top_p(&probs, top_p);
        if tok as usize >= vocab {
            return GenerateOutput { tokens: out_tokens, confidence: 0.5 };
        }
        top_probs.push(probs[tok as usize]);
        out_tokens.push(tok);
        context.push(tok);
        if tok == 0 {
            return make_output(out_tokens, top_probs);
        }

        // ── KV-cached decode loop ─────────────────────────────────────────────
        for _ in 1..max_new_tokens.min(512) {
            // Trim oldest KV entry if at the sequence length limit
            let kv_len = past_kvs[0].0.dims()[2];
            if kv_len >= max_seq {
                past_kvs = past_kvs
                    .into_iter()
                    .map(|(k, v)| (
                        k.slice([0..1, 0..nkv, 1..kv_len, 0..head_dim]),
                        v.slice([0..1, 0..nkv, 1..kv_len, 0..head_dim]),
                    ))
                    .collect();
            }

            let last_tok = *context.last().unwrap() as i32;
            let decode_t = Tensor::<InferBackend, 2, Int>::from_data(
                TensorData::new(vec![last_tok], [1, 1]),
                device,
            );
            let (decode_logits, new_kvs) = infer.forward_kv(decode_t, Some(past_kvs));
            past_kvs = new_kvs;

            let logit = decode_logits.reshape([vocab]);
            let probs = logit_to_probs(logit, &out_tokens, rep_penalty, temperature);
            if probs.is_empty() {
                break;
            }
            let tok = sample_top_p(&probs, top_p);
            if tok as usize >= vocab {
                break;
            }
            top_probs.push(probs[tok as usize]);
            out_tokens.push(tok);
            context.push(tok);
            if tok == 0 {
                break;
            }
        }

        make_output(out_tokens, top_probs)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let cfg_path = path.with_extension("cfg.json");
        std::fs::write(&cfg_path, serde_json::to_string(&self.cfg)?)?;
        if let Some(inner) = &self.inner {
            inner
                .clone()
                .save_file(path.to_path_buf(), &CompactRecorder::new())
                .map_err(|e| anyhow::anyhow!("save: {e:?}"))?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let cfg_path = path.with_extension("cfg.json");
        let cfg: MimakiwaConfig =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
        let device: <TrainBackend as Backend>::Device = Default::default();
        let inner = MimakiwaModelInner::<TrainBackend>::new(&cfg, &device)
            .load_file(path.to_path_buf(), &CompactRecorder::new(), &device)
            .map_err(|e| anyhow::anyhow!("load: {e:?}"))?;
        Ok(Self { cfg, inner: Some(inner), device })
    }
}

// ── Inference helpers ─────────────────────────────────────────────────────────

/// Convert a raw logit tensor to a probability distribution.
/// Applies repetition penalty (on recently generated tokens) and temperature,
/// then softmax — all on CPU for simplicity.
fn logit_to_probs(
    logit: Tensor<InferBackend, 1>,
    recent: &[u32],
    rep_penalty: f32,
    temperature: f32,
) -> Vec<f32> {
    let mut logits: Vec<f32> = logit.into_data().to_vec().unwrap_or_default();
    if logits.is_empty() {
        return vec![];
    }
    // Penalise tokens seen in the last 80 generated tokens
    let window = &recent[recent.len().saturating_sub(80)..];
    for &tok in window {
        if let Some(l) = logits.get_mut(tok as usize) {
            *l /= rep_penalty;
        }
    }
    for l in &mut logits {
        *l /= temperature;
    }
    // Numerically stable softmax
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum = exps.iter().sum::<f32>().max(1e-9);
    for e in &mut exps {
        *e /= sum;
    }
    exps
}

fn make_output(tokens: Vec<u32>, top_probs: Vec<f32>) -> GenerateOutput {
    let confidence = if top_probs.is_empty() {
        0.5
    } else {
        (top_probs.iter().sum::<f32>() / top_probs.len() as f32).clamp(0.0, 1.0)
    };
    GenerateOutput { tokens, confidence }
}

fn sample_top_p(probs: &[f32], top_p: f32) -> u32 {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0f32;
    let mut nucleus: Vec<(usize, f32)> = Vec::new();
    for (i, p) in &indexed {
        cumsum += p;
        nucleus.push((*i, *p));
        if cumsum >= top_p {
            break;
        }
    }
    if nucleus.is_empty() {
        return 0;
    }

    let total: f32 = nucleus.iter().map(|(_, p)| p).sum::<f32>().max(1e-9);
    let r: f32 = rand::random::<f32>() * total;
    let mut acc = 0.0f32;
    for (i, p) in &nucleus {
        acc += p;
        if acc >= r {
            return *i as u32;
        }
    }
    nucleus[0].0 as u32
}
