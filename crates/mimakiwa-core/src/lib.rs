// Thin shim — burn replaces all hand-rolled tensor math.
// Other crates should import from burn directly; this crate exists
// only so workspace references compile cleanly.
pub use burn::tensor::{backend::Backend, Shape, Tensor, Int, Float};
pub use burn::backend::{Autodiff, Wgpu};
pub use burn::backend::wgpu::WgpuDevice;
