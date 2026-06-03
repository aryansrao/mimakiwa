// Burn's AdamW handles all optimization. This module provides a compat stub
// so that any external code that imports `mimakiwa_train::optimizer::AdamW`
// still compiles.

/// Compatibility stub — the real optimizer is burn::optim::AdamW used inside Trainer.
pub struct AdamWCompat;

/// No-op gradient clip — burn handles this via the optimizer config.
pub fn clip_grads(_grads: &mut [Vec<f32>], _max_norm: f32) -> f32 { 1.0 }
