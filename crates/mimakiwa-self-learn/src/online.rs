use crate::memory::{Experience, ExperienceBuffer};

pub struct SealConfig {
    pub buffer_capacity: usize,
    pub min_reward_to_store: f32,
    pub online_lr: f32, // reserved for future LoRA fine-tuning
}

impl Default for SealConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 1000,
            min_reward_to_store: 0.2,
            online_lr: 1e-5,
        }
    }
}

pub struct SealLearner {
    pub config: SealConfig,
    pub buffer: ExperienceBuffer,
    pub total_updates: usize,
}

pub struct SealResult {
    pub stored: bool,
    pub reward: f32,
    pub updated: bool,
}

impl SealLearner {
    pub fn new(config: SealConfig) -> Self {
        Self {
            buffer: ExperienceBuffer::new(config.buffer_capacity),
            total_updates: 0,
            config,
        }
    }

    /// Record a completed conversation exchange in the experience buffer.
    /// No weight updates (inference-only model); buffer is for future LoRA fine-tuning.
    pub fn record_exchange(&mut self, prompt: &str, response: &str) -> SealResult {
        let response_len = response.split_whitespace().count();
        let length_score = (response_len as f32 / 20.0).min(1.0);
        let overlap: f32 = response.split_whitespace()
            .filter(|w| prompt.contains(*w))
            .count() as f32
            / response_len.max(1) as f32;
        let reward = (length_score * 0.4 + (1.0 - overlap * 0.5) * 0.6).clamp(0.0, 1.0);

        let stored = reward >= self.config.min_reward_to_store;
        if stored {
            self.buffer.push(Experience {
                prompt: prompt.to_string(),
                response: response.to_string(),
                reward,
                timestamp: self.total_updates as u64,
            });
        }
        self.total_updates += 1;
        SealResult { stored, reward, updated: false }
    }
}
