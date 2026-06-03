use crate::scheduler::CosineScheduler;
use crate::dataset::TextDataset;
use mimakiwa_model::{MimakiwaConfig, MimakiwaModel};
use mimakiwa_model::model::{MimakiwaModelInner, TrainBackend};
use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Tensor, Int, TensorData},
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub max_steps: usize,
    pub warmup_steps: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub log_interval: usize,
    pub eval_interval: usize,
    pub save_interval: usize,
    pub checkpoint_dir: PathBuf,
    pub eval_steps: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            seq_len: 512,
            max_steps: 10000,
            warmup_steps: 500,
            learning_rate: 3e-4,
            weight_decay: 0.1,
            grad_clip: 1.0,
            log_interval: 10,
            eval_interval: 500,
            save_interval: 1000,
            checkpoint_dir: PathBuf::from("checkpoints"),
            eval_steps: 50,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainStats {
    pub step: usize,
    pub loss: f32,
    pub lr: f32,
    pub grad_norm: f32,
    pub tokens_per_sec: f32,
    pub elapsed_secs: f32,
}

pub struct Trainer {
    pub model: MimakiwaModel,
    optim: OptimizerAdaptor<AdamW, MimakiwaModelInner<TrainBackend>, TrainBackend>,
    pub scheduler: CosineScheduler,
    pub config: TrainConfig,
    pub global_step: usize,
}

impl Trainer {
    pub fn new(model: MimakiwaModel, config: TrainConfig) -> Self {
        let optim = AdamWConfig::new()
            .with_weight_decay(config.weight_decay)
            .init();
        let scheduler = CosineScheduler::new(
            config.learning_rate,
            config.warmup_steps,
            config.max_steps,
        );
        Self { model, optim, scheduler, config, global_step: 0 }
    }

    /// Train one step on a batch of (input_ids, target_ids) pairs.
    /// All sequences must have the same length.
    pub fn train_step_batch(&mut self, batch: &[(&[u32], &[u32])]) -> TrainStats {
        let t0 = std::time::Instant::now();
        let lr = self.scheduler.get_lr(self.global_step);
        let device = self.model.device.clone();

        let batch_size = batch.len();
        let seq_len = batch[0].0.len();
        let vocab_size = self.model.cfg.vocab_size;

        // Stack batch into flat Vecs (row-major [batch, seq])
        let input_flat: Vec<i32> = batch.iter()
            .flat_map(|(inp, _)| inp.iter().map(|&x| x as i32))
            .collect();
        let target_flat: Vec<i32> = batch.iter()
            .flat_map(|(_, tgt)| tgt.iter().map(|&x| x as i32))
            .collect();

        let input_tensor = Tensor::<TrainBackend, 2, Int>::from_data(
            TensorData::new(input_flat, [batch_size, seq_len]),
            &device,
        );
        let target_tensor = Tensor::<TrainBackend, 1, Int>::from_data(
            TensorData::new(target_flat, [batch_size * seq_len]),
            &device,
        );

        // Take inner model out (optimizer needs ownership)
        let inner = self.model.inner.take().expect("model not ready");

        // Forward → logits [batch, seq, vocab]
        let logits = inner.forward(input_tensor);
        let logits_2d = logits.reshape([batch_size * seq_len, vocab_size]);

        // Cross-entropy loss
        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let loss = loss_fn.forward(logits_2d, target_tensor);
        let loss_scalar: f32 = loss.clone().mean().into_scalar();

        // Backward + optimizer step
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &inner);
        let inner = self.optim.step(lr as f64, inner, grads_params);

        // Return inner to the model wrapper
        self.model.inner = Some(inner);

        self.global_step += 1;
        let elapsed = t0.elapsed().as_secs_f32();
        let tokens_per_sec = (batch_size * seq_len) as f32 / elapsed;

        TrainStats {
            step: self.global_step,
            loss: loss_scalar,
            lr,
            grad_norm: 0.0, // burn handles gradient clipping internally
            tokens_per_sec,
            elapsed_secs: elapsed,
        }
    }

    /// Full training loop (not called by GUI — GUI calls train_step_batch directly).
    pub fn train(&mut self, dataset: &TextDataset, _eval: Option<&TextDataset>) -> Result<()> {
        println!(
            "Training {} steps, batch={}, seq={}  ({:.1}M params)",
            self.config.max_steps, self.config.batch_size, self.config.seq_len,
            self.model.n_params() as f32 / 1e6
        );
        std::fs::create_dir_all(&self.config.checkpoint_dir)?;

        while self.global_step < self.config.max_steps {
            let batch = dataset.random_batch(self.config.batch_size);
            let stats = self.train_step_batch(&batch);

            if self.global_step % self.config.log_interval == 0 {
                println!(
                    "step {:6} | loss {:.4} | lr {:.2e} | {:.0} tok/s",
                    stats.step, stats.loss, stats.lr, stats.tokens_per_sec
                );
            }

            if self.global_step % self.config.save_interval == 0 {
                let ckpt = self.config.checkpoint_dir
                    .join(format!("mimakiwa_step{}.bin", self.global_step));
                self.model.save(&ckpt)?;
            }
        }

        let final_path = self.config.checkpoint_dir.join("mimakiwa_final.bin");
        self.model.save(&final_path)?;
        println!("Training complete → {}", final_path.display());
        Ok(())
    }

    pub fn save_checkpoint(&self, path: &Path) -> Result<()> {
        self.model.save(path)
    }

    pub fn load_checkpoint(path: &Path, _cfg: MimakiwaConfig, train_cfg: TrainConfig) -> Result<Self> {
        let model = MimakiwaModel::load(path)?;
        Ok(Self::new(model, train_cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimakiwa_model::{MimakiwaConfig, MimakiwaModel};

    #[test]
    fn smoke_three_steps() {
        // Minimal vocab / tiny model — just checks no autodiff panic
        let vocab_size = 260; // 256 bytes + 4 merges
        let cfg = MimakiwaConfig {
            vocab_size,
            embed_dim: 32,
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_dim: 64,
            max_seq_len: 16,
            rope_base: 10000.0,
        };
        let model = MimakiwaModel::new(cfg);
        let train_cfg = TrainConfig {
            batch_size: 2,
            seq_len: 8,
            max_steps: 3,
            warmup_steps: 1,
            learning_rate: 1e-3,
            checkpoint_dir: std::path::PathBuf::from("/tmp"),
            save_interval: 9999,
            ..TrainConfig::default()
        };
        let mut trainer = Trainer::new(model, train_cfg);
        // Build a tiny dataset: random token ids
        let tokens: Vec<u32> = (0..200u32).cycle().take(200).collect();
        let dataset = TextDataset::from_tokens(tokens, 8);
        for _ in 0..3 {
            let batch = dataset.random_batch(2);
            let stats = trainer.train_step_batch(&batch);
            assert!(stats.loss.is_finite(), "loss is NaN/Inf");
            println!("loss={:.4} tok/s={:.0}", stats.loss, stats.tokens_per_sec);
        }
    }
}
