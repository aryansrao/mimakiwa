//! Standalone cloud training binary for Mimakiwa.
//!
//! Backends:
//!   default (no features) — NdArray, pure CPU, works everywhere
//!   --features cuda       — LibTorch CUDA, uses GPU (T4 on Colab)
//!
//! Weights saved with CompactRecorder are backend-portable → loads on Mac WGPU.
//!
//! Usage (all flags optional):
//!   cloud-train [--model compact|small] [--dataset-url URL] [--steps N]
//!               [--batch N] [--seq N] [--vocab N] [--lr F] [--max-mb N] [--out DIR]

use anyhow::Result;
use burn::{
    module::Module as _,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    tensor::{backend::Backend, Int, Tensor, TensorData},
};
use mimakiwa_model::model::{MimakiwaConfig, MimakiwaModelInner};
use mimakiwa_tokenizer::BPETokenizer;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{io::Read, path::PathBuf};

// ── Backend selection ─────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod back {
    use burn::backend::{libtorch::LibTorchDevice, Autodiff, LibTorch};
    pub type CB  = Autodiff<LibTorch>;
    pub type Dev = LibTorchDevice;
    pub fn device() -> Dev { LibTorchDevice::Cuda(0) }
}

#[cfg(not(feature = "cuda"))]
mod back {
    use burn::backend::{Autodiff, NdArray};
    pub type CB  = Autodiff<NdArray>;
    pub type Dev = <CB as burn::tensor::backend::Backend>::Device;
    pub fn device() -> Dev { Default::default() }
}

use back::{CB, Dev};

// ── Dataset ───────────────────────────────────────────────────────────────────

struct Dataset {
    tokens: Vec<u32>,
    seq:    usize,
}

impl Dataset {
    fn new(tokens: Vec<u32>, seq: usize) -> Self { Self { tokens, seq } }
    fn batch(&self, n: usize) -> Vec<(Vec<u32>, Vec<u32>)> {
        let mut rng = rand::thread_rng();
        let max = self.tokens.len().saturating_sub(self.seq + 1);
        (0..n).map(|_| {
            let i = rng.gen_range(0..max);
            (self.tokens[i..i+self.seq].to_vec(), self.tokens[i+1..i+self.seq+1].to_vec())
        }).collect()
    }
}

// ── LR schedule ───────────────────────────────────────────────────────────────

fn lr_at(step: usize, total: usize, warmup: usize, base: f32) -> f32 {
    if step < warmup {
        base * step as f32 / warmup.max(1) as f32
    } else {
        let t = (step - warmup) as f32 / (total - warmup).max(1) as f32;
        base * (1.0 + (std::f32::consts::PI * t).cos()) / 2.0
    }
}

// ── Corpus parsing ────────────────────────────────────────────────────────────

fn wrap_as_qa(text: &str) -> String {
    const PROMPTS: &[&str] = &[
        "Tell me a story.", "Can you share a story?", "Write me a short story.",
        "Give me a story.", "Tell me something.",
    ];
    let mut out = String::with_capacity(text.len() + text.len() / 4);
    let mut pi = 0usize;
    for chunk in text.split("\n\n") {
        let c = chunk.trim();
        if c.len() < 30 { continue; }
        out.push_str("User: "); out.push_str(PROMPTS[pi % PROMPTS.len()]); pi += 1;
        out.push_str("\nAssistant: "); out.push_str(c); out.push_str("\n\n");
    }
    out
}

fn parse_dolly(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let inst = v["instruction"].as_str().unwrap_or("");
            let ctx  = v["context"].as_str().unwrap_or("");
            let resp = v["response"].as_str().unwrap_or("");
            if !resp.is_empty() {
                if ctx.is_empty() {
                    out.push_str(&format!("User: {inst}\nAssistant: {resp}\n\n"));
                } else {
                    out.push_str(&format!("User: {inst}\n{ctx}\nAssistant: {resp}\n\n"));
                }
            }
        }
    }
    out
}

// ── Download ──────────────────────────────────────────────────────────────────

fn download(url: &str, max_mb: usize) -> Result<Vec<u8>> {
    eprintln!("[dl] {url}");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600)).build()?;
    let mut resp = client.get(url).header("User-Agent", "mimakiwa-cloud/0.1").send()?;
    if !resp.status().is_success() { anyhow::bail!("HTTP {}", resp.status()); }
    let max = max_mb * 1_048_576;
    let mut buf = vec![0u8; 65536];
    let mut out = Vec::with_capacity(max.min(4 * 1_048_576));
    let mut got = 0usize;
    loop {
        if got >= max { break; }
        match resp.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let take = (max - got).min(n);
                out.extend_from_slice(&buf[..take]);
                got += take;
                if got % 2_097_152 == 0 { eprintln!("[dl] {} MB…", got / 1_048_576); }
            }
        }
    }
    eprintln!("[dl] {} MB total", out.len() / 1_048_576);
    Ok(out)
}

// ── Config wrapper for JSON serialisation (matches MimakiwaConfig field names) ─

#[derive(Serialize, Deserialize)]
struct CfgJson {
    vocab_size:  usize,
    embed_dim:   usize,
    n_layers:    usize,
    n_heads:     usize,
    n_kv_heads:  usize,
    ffn_dim:     usize,
    max_seq_len: usize,
    rope_base:   f32,
}

impl From<&MimakiwaConfig> for CfgJson {
    fn from(c: &MimakiwaConfig) -> Self {
        Self {
            vocab_size:  c.vocab_size,
            embed_dim:   c.embed_dim,
            n_layers:    c.n_layers,
            n_heads:     c.n_heads,
            n_kv_heads:  c.n_kv_heads,
            ffn_dim:     c.ffn_dim,
            max_seq_len: c.max_seq_len,
            rope_base:   c.rope_base,
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str, default: &str| -> String {
        args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
            .unwrap_or_else(|| default.to_string())
    };

    let model_size    = get("--model",   "compact");  // compact | small
    let url           = get("--dataset-url",
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt");
    let vocab: usize  = get("--vocab",  "8192").parse().unwrap_or(8192);
    let max_mb: usize = get("--max-mb", "20").parse().unwrap_or(20);
    let out_dir       = get("--out",    "./model_out");
    let is_dolly      = url.contains("dolly");

    // Model-size-based defaults (overridable per flag)
    let (def_steps, def_batch, def_seq, def_lr) = match model_size.as_str() {
        "small"  => (20_000usize, 4usize, 512usize, 2e-4f32),
        _        => (20_000usize, 8usize, 256usize, 3e-4f32),
    };
    let steps: usize = get("--steps", &def_steps.to_string()).parse().unwrap_or(def_steps);
    let batch: usize = get("--batch", &def_batch.to_string()).parse().unwrap_or(def_batch);
    let seq:   usize = get("--seq",   &def_seq.to_string()).parse().unwrap_or(def_seq);
    let lr:    f32   = get("--lr",    &def_lr.to_string()).parse().unwrap_or(def_lr);

    let backend_name = if cfg!(feature = "cuda") { "LibTorch/CUDA" } else { "NdArray/CPU" };
    std::fs::create_dir_all(&out_dir)?;
    eprintln!("Mimakiwa cloud trainer | backend={backend_name} | model={model_size} | steps={steps} batch={batch} seq={seq}");

    // 1. Download dataset
    let raw = download(&url, max_mb)?;
    let text = String::from_utf8_lossy(&raw);
    let corpus = if is_dolly { parse_dolly(&text) } else { wrap_as_qa(&text) };
    eprintln!("[corpus] {} chars", corpus.len());

    // 2. BPE tokenizer
    eprintln!("[bpe] training vocab={vocab}…");
    let mut tok = BPETokenizer::new();
    tok.train(&corpus, vocab);
    let tok_path = PathBuf::from(&out_dir).join("tokenizer.json");
    tok.save(&tok_path)?;
    eprintln!("[bpe] vocab={} saved → {}", tok.vocab_size(), tok_path.display());

    // 3. Tokenise corpus
    let tokens = tok.encode_fast(&corpus);
    eprintln!("[data] {} tokens", tokens.len());
    if tokens.len() < seq + 2 { anyhow::bail!("corpus too small for seq={seq}"); }
    let dataset = Dataset::new(tokens, seq);

    // 4. Build model
    let cfg = match model_size.as_str() {
        "small"  => MimakiwaConfig::small(tok.vocab_size()),
        _        => MimakiwaConfig::compact(tok.vocab_size()),
    };
    eprintln!("[model] {} | dim={} layers={} heads={} vocab={}",
        model_size, cfg.embed_dim, cfg.n_layers, cfg.n_heads, cfg.vocab_size);
    let device: Dev = back::device();
    let mut model: Option<MimakiwaModelInner<CB>> = Some(MimakiwaModelInner::new(&cfg, &device));
    let mut optim = AdamWConfig::new().with_weight_decay(0.1).init();
    let warmup = (steps / 20).max(50);

    // 5. Train
    eprintln!("[train] starting…");
    let t0 = std::time::Instant::now();
    for step in 0..steps {
        let cur_lr = lr_at(step, steps, warmup, lr);
        let b_data = dataset.batch(batch);
        let bs = b_data.len();

        let inp_f: Vec<i32> = b_data.iter().flat_map(|(i,_)| i.iter().map(|&x| x as i32)).collect();
        let tgt_f: Vec<i32> = b_data.iter().flat_map(|(_,t)| t.iter().map(|&x| x as i32)).collect();

        let inp_t = Tensor::<CB, 2, Int>::from_data(TensorData::new(inp_f, [bs, seq]), &device);
        let tgt_t = Tensor::<CB, 1, Int>::from_data(TensorData::new(tgt_f, [bs*seq]), &device);

        let inner = model.take().unwrap();
        let logits = inner.forward(inp_t).reshape([bs * seq, cfg.vocab_size]);
        let loss = CrossEntropyLossConfig::new().init(&device).forward(logits, tgt_t);
        let loss_val: f32 = loss.clone().mean().into_scalar();
        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &inner);
        model = Some(optim.step(cur_lr as f64, inner, gp));

        if step % 500 == 0 || step == steps - 1 {
            let el = t0.elapsed().as_secs_f32();
            let tok_s = (step + 1) as f32 * (bs * seq) as f32 / el;
            eprintln!("[train] step {}/{} | loss {:.4} | lr {:.2e} | {:.0} tok/s",
                step+1, steps, loss_val, cur_lr, tok_s);
        }
    }

    // 6. Save
    let model_path = PathBuf::from(&out_dir).join("mimakiwa.bin");
    let cfg_path   = PathBuf::from(&out_dir).join("mimakiwa.cfg.json");

    model.unwrap()
         .save_file(model_path.clone(), &CompactRecorder::new())
         .map_err(|e| anyhow::anyhow!("save model: {e:?}"))?;
    std::fs::write(&cfg_path, serde_json::to_string(&CfgJson::from(&cfg))?)?;

    eprintln!("[done] {}", model_path.display());
    eprintln!("[done] {}", cfg_path.display());
    eprintln!("[done] {}", tok_path.display());
    eprintln!("\nCopy these 3 files to ~/.mimakiwa/ on your Mac, then: cargo run --release");
    Ok(())
}
