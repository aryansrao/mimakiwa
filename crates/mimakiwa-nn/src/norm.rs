use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor, TensorData},
};

/// RMS Layer Norm — burn autodiffs the backward pass.
#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    scale: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        let scale = Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![1.0f32; dim], [dim]),
            device,
        ));
        Self { scale, eps: 1e-5 }
    }

    /// x: [batch, seq_len, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_b, _s, d] = x.dims();
        let mean_sq = x.clone().powf_scalar(2.0).mean_dim(2); // [b, s, 1]
        let rms = (mean_sq + self.eps as f32).sqrt();           // [b, s, 1]
        let normed = x / rms;
        let scale = self.scale.val().reshape([1, 1, d]);
        normed * scale
    }
}
