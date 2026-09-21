# NCHW DeconvSum measurements

This change reads and adds complete input rows for each kernel tap. It reduces memory jumps in the generic CPU path for 2D NCHW and CHW transposed convolution.

The added path requires contiguous buffers and input width of at least 16. It keeps the addition order for each output and adds no heap buffer or unsafe code. The existing 2x2, stride-2 path is unchanged. NHWC and HWC use the existing path.

## Results

Intel Core Ultra 7 255H, WSL2 x86-64, Rust 1.97.1, release build, one thread pinned to CPU 0. The measured base is `98df95c8f31dd623257e39cb4ce11b12419c27aa`. The PR uses a later main commit, `92a30df`; the CPU crates used here have no changes between these two commits.

| Workload | Base (ms) | Change (ms) | Time reduction |
|---|---:|---:|---:|
| DCGAN generator, batch 1 | 3.304 | 2.549 | 22.9% |
| DCGAN generator, batch 4 | 14.591 | 11.752 | 19.5% |
| ConvTranspose, CI=4, CO=32, 64x64, K=3x3, stride=2 | 1.438 | 1.076 | 25.2% |
| ConvTranspose, CI=4, CO=32, 64x65, K=3x3, stride=1, dilation=2 | 0.983 | 0.664 | 32.5% |
| ConvTranspose, CI=8, CO=32, 96x96, K=4x4, stride=2 | 19.321 | 18.070 | 6.5% |
| ConvTranspose, CI=4, CO=32, 128x128, K=5x3, stride=2 | 34.140 | 31.779 | 6.9% |

The generator has five ConvTranspose layers, channel counts 512, 256, 128, 64, and 3, a channel affine operation at each intermediate stage, ReLU, and final Tanh. The input is `[N,100,1,1]`; the output is `[N,3,64,64]`. Its shape follows the [PyTorch DCGAN example](https://github.com/pytorch/examples/blob/acc295dc7b90714f1bf47f06004fc19a7fe235c4/dcgan/main.py). We use fixed synthetic weights and inputs. This is a model architecture test, not a trained checkpoint or an image quality test. The added path runs in the last two stages.

The model test uses six pairs of independent processes. Each process takes nine samples with four runs per sample. Process order alternates between base/change and change/base. The paired time reductions are 21.7-23.2% for batch 1 and 17.4-21.3% for batch 4. Model setup and optimization are outside the timed section. `plan.run`, its buffer allocations, all operators, and output release are inside it. Output hashes match between builds. Each process also checks the complete output bytes after its timed runs.

The operator test uses four process pairs and nine samples per process. Most samples contain eight runs; small cases use 512. The test includes GEMM and bias, not only DeconvSum. Raw results, shapes, process samples, output hashes, and optimized operator records are in [models.json](models.json) and [audit_unrolled.json](audit_unrolled.json).

Small fallback cases with input width 8 and 9 measured 1.4% and 1.0% slower (0.034 and 0.027 microseconds). Unchanged NHWC and depthwise paths also vary; one depthwise control measured 11.3% slower. Group-2 results vary between processes. These small or unrelated changes are not evidence of a general gain. ARM and GPU performance is not measured.

## Correctness and checks

- All 99 selected deconvolution tests pass with `PROPTEST_CASES=512`, including raw, decluttered, and optimized models. See [the test log](audit_final_tests.log).
- The [independent reference test](src/bin/audit_edges.rs) passes 5,760 cases: F16/F32/F64, NCHW/CHW/NHWC/HWC, groups 1 and 2, rectangular kernels, dilation, inner strides 1/2/3, and several padding modes. F16 and F32 reference sums round after each addition. Outputs match bit for bit. Nonzero adjustments are tested with Valid and Explicit padding; SAME with nonzero adjustment is outside this extra test.
- The committed suite cases include widths 15, 16, and 17 to test both sides of the width guard.
- `cargo fmt --all --check` and `git diff --check` pass. `cargo clippy --all-targets`, as used by the upstream CI, also completes. It reports existing warnings in linalg, CLI, and transformers; none are in changed code. The full `cargo clippy --workspace` stops at an Apple framework dependency on Linux. See [the Linux lint log](publish_linux_clippy.log) and [the workspace log](publish_workspace_clippy.log).

## Reproduce

This branch contains the probes and results only. Clone tract into the `tract` directory, then use the same lock file for both builds. The patch is the exact two-file code and test change.

```bash
git clone https://github.com/sonos/tract.git tract
git -C tract checkout 98df95c8f31dd623257e39cb4ce11b12419c27aa
CARGO_INCREMENTAL=0 cargo build --release --locked --bin model_bench --bin audit_bench -j6
cp target/release/model_bench model_base
cp target/release/audit_bench operator_base
git -C tract apply ../candidate.patch
CARGO_INCREMENTAL=0 cargo build --release --locked --bin model_bench --bin audit_bench --bin audit_edges -j6
cp target/release/model_bench model_candidate
cp target/release/audit_bench operator_candidate
target/release/audit_edges
python3 compare.py . --kind tract --pairs 6 --threads 1 --base-name model_base --candidate-name model_candidate --output replay_models.json
python3 compare.py . --kind tract --pairs 4 --threads 1 --base-name operator_base --candidate-name operator_candidate --output replay_operators.json
PROPTEST_CASES=512 CARGO_INCREMENTAL=0 cargo test --manifest-path tract/Cargo.toml --release -p test-unit-core deconv -- --test-threads=1
```

Do not compile or run other CPU work during the timed comparison. The source files, lock file, patch, and raw results have SHA-256 hashes in `publication.json`.
