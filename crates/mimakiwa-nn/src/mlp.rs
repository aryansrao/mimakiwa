use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor, activation},
};

/// SwiGLU Feed-Forward Network — burn autodiffs the backward pass.
#[derive(Module, Debug)]
pub struct SwiGLUMLP<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> SwiGLUMLP<B> {
    pub fn new(embed_dim: usize, ffn_dim: usize, device: &B::Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(embed_dim, ffn_dim).with_bias(false).init(device),
            up_proj:   LinearConfig::new(embed_dim, ffn_dim).with_bias(false).init(device),
            down_proj: LinearConfig::new(ffn_dim, embed_dim).with_bias(false).init(device),
        }
    }

    /// x: [batch, seq_len, embed_dim] → [batch, seq_len, embed_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        let up   = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}
