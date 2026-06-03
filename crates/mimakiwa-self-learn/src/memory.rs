use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Experience {
    pub prompt: String,
    pub response: String,
    pub reward: f32,
    pub timestamp: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ExperienceBuffer {
    pub buffer: Vec<Experience>,
    pub capacity: usize,
    pub total_added: u64,
}

impl ExperienceBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { buffer: Vec::with_capacity(capacity), capacity, total_added: 0 }
    }

    pub fn push(&mut self, exp: Experience) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(exp);
        } else {
            let worst = self.buffer.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.reward.partial_cmp(&b.reward).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.buffer[worst] = exp;
        }
        self.total_added += 1;
    }

    pub fn len(&self) -> usize { self.buffer.len() }
    pub fn is_empty(&self) -> bool { self.buffer.is_empty() }

    pub fn mean_reward(&self) -> f32 {
        if self.buffer.is_empty() { return 0.0; }
        self.buffer.iter().map(|e| e.reward).sum::<f32>() / self.buffer.len() as f32
    }
}
