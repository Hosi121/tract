use std::time::Instant;
use tract_core::internal::*;
use tract_core::ops::cnn::{Deconv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;

fn data(shape: &[usize], seed: usize) -> TractResult<Tensor> {
    let values: Vec<f32> = (0..shape.iter().product())
        .map(|i| ((i * 17 + seed * 11) % 127) as f32 / 63.0 - 1.0)
        .collect();
    Tensor::from_shape(shape, &values)
}

fn main() -> TractResult<()> {
    println!("layout,ci,co,h,w,kh,kw,stride,dilation,group,sample,us,hash");
    for (ci, co, h, w, kh, kw, stride, dilation, group) in [
        (4, 32, 64, 64, 3, 3, 2, 1, 1),
        (8, 32, 96, 96, 4, 4, 2, 1, 1),
        (4, 32, 128, 128, 5, 3, 2, 1, 1),
        (4, 32, 64, 65, 3, 3, 1, 2, 1),
        (8, 32, 64, 65, 3, 5, 2, 1, 2),
        (8, 32, 128, 128, 2, 2, 2, 1, 1),
        (4, 3, 3, 5, 3, 3, 2, 1, 1),
        (1, 1, 1, 1, 3, 3, 1, 1, 1),
        (1, 1, 16, 16, 3, 3, 1, 1, 1),
        (2, 2, 32, 33, 3, 3, 2, 1, 1),
        (4, 7, 32, 33, 3, 5, 3, 1, 1),
        (4, 8, 32, 33, 3, 3, 1, 1, 1),
        (8, 8, 32, 33, 3, 3, 2, 1, 8),
        (4, 4, 32, 33, 5, 3, 1, 2, 2),
        (4, 4, 3, 8, 3, 3, 2, 1, 1),
        (4, 4, 3, 9, 3, 3, 2, 1, 1),
    ] {
        for format in [DataFormat::NCHW, DataFormat::NHWC] {
            let input_shape = format.from_n_c_hw(1, ci, [h, w])?;
            let input = data(&input_shape.shape, 1)?.into_tvalue();
            let mut model = TypedModel::default();
            let src = model.add_source("input", f32::fact(&input_shape.shape))?;
            let kernel = model.add_const("kernel", data(&[co, ci / group, kh, kw], 2)?)?;
            let bias = model.add_const("bias", data(&[co], 3)?)?;
            let spec = PoolSpec::new(
                format, tvec!(kh, kw), PaddingSpec::Valid,
                Some(tvec!(dilation, dilation)), Some(tvec!(stride, stride)), ci, co,
            );
            let out = model.wire_node("deconv", Deconv::new(spec, KernelFormat::OIHW, tvec!(0,0), group), &[src, kernel, bias])?;
            model.select_output_outlets(&out)?;
            let model = model.into_optimized()?;
            let node = model.nodes().iter().find(|n| n.op.name() == "DeconvSum");
            if let Some(node) = node {
                eprintln!("shape={format:?}/{ci}/{co}/{h}/{w}/{kh}/{kw}/{stride}/{dilation}/{group} op={:?}", node.op);
            } else {
                eprintln!("shape={format:?}/{ci}/{co}/{h}/{w}/{kh}/{kw}/{stride}/{dilation}/{group} other_path");
            }
            let plan = model.into_runnable()?;
            for _ in 0..3 { let _ = plan.run(tvec!(input.clone()))?; }
            let output = plan.run(tvec!(input.clone()))?;
            let tensor = output[0].to_plain_array_view::<f32>()?;
            let mut hash = 1469598103934665603u64;
            for &v in tensor.iter() {
                ensure!(v.is_finite());
                for byte in v.to_bits().to_le_bytes() { hash = (hash ^ byte as u64).wrapping_mul(1099511628211); }
            }
            let repeats = if h * w <= 256 { 512 } else { 8 };
            for sample in 0..9 {
                let start = Instant::now();
                for _ in 0..repeats { std::hint::black_box(plan.run(tvec!(input.clone()))?); }
                let us = start.elapsed().as_secs_f64() * 1e6 / repeats as f64;
                println!("{format:?},{ci},{co},{h},{w},{kh},{kw},{stride},{dilation},{group},{sample},{us:.6},{hash:016x}");
            }
            let after = plan.run(tvec!(input.clone()))?;
            ensure!(after[0].as_bytes() == output[0].as_bytes(), "Output changed after repeated runs");
        }
    }
    Ok(())
}
