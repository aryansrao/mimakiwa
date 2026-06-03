// Cosine learning rate scheduler with linear warmup
// Matches the schedule used in GPT-2, LLaMA training

pub struct CosineScheduler {
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl CosineScheduler {
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            min_lr: base_lr * 0.1,
            warmup_steps,
            total_steps,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    pub fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (step + 1) as f32 / self.warmup_steps as f32
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            // Cosine decay
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps) as f32;
            self.min_lr + 0.5 * (self.base_lr - self.min_lr)
                * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }
}

// WSD (Warmup-Stable-Decay) scheduler from MiniCPM (2024)
// More stable than pure cosine for large models
pub struct WSDScheduler {
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub stable_steps: usize,
    pub decay_steps: usize,
}

impl WSDScheduler {
    pub fn new(base_lr: f32, warmup_steps: usize, stable_steps: usize, decay_steps: usize) -> Self {
        Self {
            base_lr,
            min_lr: base_lr * 0.01,
            warmup_steps,
            stable_steps,
            decay_steps,
        }
    }

    pub fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step + 1) as f32 / self.warmup_steps as f32
        } else if step < self.warmup_steps + self.stable_steps {
            self.base_lr
        } else {
            let decay_step = step - self.warmup_steps - self.stable_steps;
            if decay_step >= self.decay_steps {
                self.min_lr
            } else {
                let t = decay_step as f32 / self.decay_steps as f32;
                self.min_lr + (self.base_lr - self.min_lr) * (1.0 - t)
            }
        }
    }
}
