use std::time::Instant;
use tract_core::internal::*;
use tract_core::ops::cnn::{Deconv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;

fn data(shape: &[usize], mut seed: u64, scale: f32) -> TractResult<Tensor> {
    let values: Vec<f32> = (0..shape.iter().product())
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * scale
        })
        .collect();
    Tensor::from_shape(shape, &values)
}

fn main() -> TractResult<()> {
    println!("model,batch,sample,us,hash");
    for batch in [1, 4] {
        let input = data(&[batch, 100, 1, 1], 29, 1.0)?.into_tvalue();
        let mut model = TypedModel::default();
        let mut x = model.add_source("input", f32::fact([batch, 100, 1, 1]))?;
        let mut ci = 100;
        for (stage, co) in [512, 256, 128, 64, 3].into_iter().enumerate() {
            let kernel = model.add_const(
                format!("weight{stage}"),
                data(&[co, ci, 4, 4], 73 + stage as u64, 0.04)?,
            )?;
            let bias = model.add_const(format!("bias{stage}"), tensor0(0f32))?;
            let stride = if stage == 0 { 1 } else { 2 };
            let pad = if stage == 0 { 0 } else { 1 };
            let spec = PoolSpec::new(
                DataFormat::NCHW,
                tvec!(4, 4),
                PaddingSpec::Explicit(tvec!(pad, pad), tvec!(pad, pad)),
                Some(tvec!(1, 1)),
                Some(tvec!(stride, stride)),
                ci,
                co,
            );
            x = model.wire_node(
                format!("deconv{stage}"),
                Deconv::new(spec, KernelFormat::OIHW, tvec!(0, 0), 1),
                &[x, kernel, bias],
            )?[0];
            if stage != 4 {
                let scale = model.add_const(
                    format!("bn_scale{stage}"),
                    Tensor::from_shape(&[1, co, 1, 1], &vec![0.98f32; co])?,
                )?;
                let shift = model.add_const(
                    format!("bn_shift{stage}"),
                    data(&[1, co, 1, 1], 113 + stage as u64, 0.01)?,
                )?;
                x = model.wire_node(format!("bn_mul{stage}"), tract_core::ops::math::mul(), &[x, scale])?[0];
                x = model.wire_node(format!("bn_add{stage}"), tract_core::ops::math::add(), &[x, shift])?[0];
                x = model.wire_node(format!("relu{stage}"), tract_core::ops::nn::leaky_relu(0.0), &[x])?[0];
            } else {
                x = model.wire_node("tanh", tract_core::ops::math::tanh(), &[x])?[0];
            }
            ci = co;
        }
        model.select_output_outlets(&[x])?;
        let model = model.into_optimized()?;
        for node in model.nodes().iter().filter(|n| n.op.name() == "DeconvSum") {
            eprintln!("batch={batch} {} {:?}", node.name, node.op);
        }
        let plan = model.into_runnable()?;
        let output = plan.run(tvec!(input.clone()))?;
        ensure!(output[0].shape() == [batch, 3, 64, 64]);
        ensure!(output[0].to_plain_array_view::<f32>()?.iter().all(|x| x.is_finite()));
        let hash = output[0].as_bytes().iter().fold(1469598103934665603u64, |h, b| {
            (h ^ *b as u64).wrapping_mul(1099511628211)
        });
        for _ in 0..3 { std::hint::black_box(plan.run(tvec!(input.clone()))?); }
        for sample in 0..9 {
            let start = Instant::now();
            for _ in 0..4 { std::hint::black_box(plan.run(tvec!(input.clone()))?); }
            let us = start.elapsed().as_secs_f64() * 1e6 / 4.0;
            println!("dcgan64,{batch},{sample},{us:.6},{hash:016x}");
        }
        let after = plan.run(tvec!(input.clone()))?;
        ensure!(after[0].as_bytes() == output[0].as_bytes());
    }
    Ok(())
}
