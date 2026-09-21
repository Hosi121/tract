use tract_core::internal::*;
use tract_core::ops::cnn::{Deconv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::nn::{DataFormat, DataShape};

fn offset(s: &DataShape, n: usize, c: usize, y: usize, x: usize) -> usize {
    n * s.n_stride().copied().unwrap_or(0) + c * s.c_stride()
        + y * s.hw_strides()[0] + x * s.hw_strides()[1]
}

fn round(value: f64, dt: DatumType) -> f64 {
    match dt {
        DatumType::F16 => f16::from_f64(value).to_f64(),
        DatumType::F32 => (value as f32) as f64,
        DatumType::F64 => value,
        _ => unreachable!(),
    }
}

fn main() -> TractResult<()> {
    let mut count = 0;
    for dt in [DatumType::F16, DatumType::F32, DatumType::F64] {
      for sw in [1, 2, 3] {
      for fractional in [false, true] {
        for df in [DataFormat::NCHW, DataFormat::CHW, DataFormat::NHWC, DataFormat::HWC] {
            for mode in 0..5 {
              for adjustment in [0, 1] {
                if adjustment != 0 && (mode == 1 || mode == 2) { continue; }
                for group in [1, 2] {
                    for (h, w) in [(3, 9), (1, 1), (2, 3), (2, 17), (3, 33)] {
                        let n = if df.has_n() { 2 } else { 1 };
                        let (ci, co, kh, kw, sh, dh, dw) = (4, 6, 3, 5, 2, 2, 3);
                        let input = df.from_n_c_hw(n, ci, [h, w])?;
                        let oh_full = (h - 1) * sh + (kh - 1) * dh + 1;
                        let ow_full = (w - 1) * sw + (kw - 1) * dw + 1;
                        let (padding, oh, ow, ph, pw) = match mode {
                            0 => (PaddingSpec::Valid, oh_full + adjustment, ow_full, 0, 0),
                            1 | 2 => {
                                let lower = usize::from(mode == 2);
                                (if mode == 1 { PaddingSpec::SameUpper } else { PaddingSpec::SameLower },
                                 h * sh, w * sw, (oh_full + adjustment - h * sh + lower) / 2,
                                 (ow_full - w * sw + lower) / 2)
                            },
                            3 => (PaddingSpec::Explicit(tvec!(2, 3), tvec!(1, 2)), oh_full + adjustment - 3, ow_full - 5, 2, 3),
                            _ => (PaddingSpec::Explicit(tvec!(4, 7), tvec!(0, 0)), oh_full + adjustment - 4, ow_full - 7, 4, 7),
                        };
                        let output = df.from_n_c_hw(n, co, [oh, ow])?;
                        let mut model = TypedModel::default();
                        let source = model.add_source("input", f32::fact(&input.shape))?;
                        let kernel_values: Vec<f32> = (0..co * (ci / group) * kh * kw).map(|i| (i % 5) as f32 - 2.0).collect();
                        let kernel = model.add_const("kernel", Tensor::from_shape(&[co, ci / group, kh, kw], &kernel_values)?)?;
                        let bias = model.add_const("bias", tensor0(0.0f32))?;
                        let spec = PoolSpec::new(df, tvec!(kh,kw), padding,
                            Some(tvec!(dh,dw)), Some(tvec!(sh,sw)), ci, co);
                        let out = model.wire_node("deconv", Deconv::new(spec, KernelFormat::OIHW, tvec!(adjustment,0), group), &[source,kernel,bias])?;
                        model.select_output_outlets(&out)?;
                        let model = model.into_optimized()?;
                        let node = model.nodes().iter().find(|node| node.op.name() == "DeconvSum").unwrap();
                        let cols: Vec<f64> = (0..n * co * kh * kw * h * w).map(|i| {
                            let value = if fractional { ((i * 7 + i / 11) % 997) as f64 / 499.0 - 1.0 }
                                else { ((i * 7 + i / 11) % 5) as f64 - 2.0 };
                            round(value, dt)
                        }).collect();
                        let fact = model.outlet_fact(node.inputs[0])?;
                        let gemm = Tensor::from_shape(fact.shape.as_concrete().unwrap(), &cols)?.cast_to_dt(dt)?.into_owned();
                        let mut expected = vec![0f64; n * co * oh * ow];
                        for b in 0..n { for c in 0..co { for y in 0..oh { for x in 0..ow {
                            expected[offset(&output,b,c,y,x)] = (c % 3) as f64 - 1.0;
                        }}}}
                        let bias = Tensor::from_shape(&output.shape, &expected)?.cast_to_dt(dt)?.into_owned();
                        for b in 0..n { for c in 0..co { for ky in 0..kh { for kx in 0..kw {
                            for y in 0..h { for x in 0..w {
                                let oy = (y * sh + ky * dh) as isize - ph as isize;
                                let ox = (x * sw + kx * dw) as isize - pw as isize;
                                if oy < 0 || ox < 0 || oy >= oh as isize || ox >= ow as isize { continue; }
                                let i = (((b * co + c) * kh + ky) * kw + kx) * h * w + y * w + x;
                                let index = offset(&output,b,c,oy as usize,ox as usize);
                                expected[index] = round(expected[index] + cols[i], dt);
                            }}
                        }}}}
                        let actual = node.op.eval(&EvalContext::out_of_plan(), tvec!(gemm.into_tvalue(), bias.into_tvalue()))?;
                        let actual = actual[0].clone().into_tensor().cast_to_dt(DatumType::F64)?.into_owned();
                        ensure!(actual.to_plain_array_view::<f64>()?.iter().zip(&expected).all(|(a,b)| a.to_bits() == b.to_bits()), "Mismatch: {dt:?} {df:?} mode={mode} adjustment={adjustment} sw={sw} group={group} h={h} w={w} fractional={fractional}");
                        count += 1;
                    }
                }
            }
        }
              }
      }
      }
    }
    println!("Verified {count} cases for F16, F32, and F64");
    Ok(())
}
