# Vitis Libraries Operator Substitution Opportunities

This note reviews the custom HLS implementation that ships with the project and
highlights which parts of the `ecg_cnn_inference` kernel could be replaced by
pre-built primitives from the [Vitis Libraries](https://github.com/Xilinx/Vitis_Libraries).
The goal is to help you decide where reusing library operators can shorten the
implementation cycle or improve resource/performance characteristics on Pynq-Z2.

## 1. Current Kernel Building Blocks

The exported inference core fuses several stages together (`FPGA/hls_cnn/cnn_inference.cpp`):

1. `conv_bn_relu_pool` — performs 2D convolution, applies batch normalisation and
   ReLU, then performs 2×2 max pooling for three successive feature extractor
   stages.
2. `flatten` — converts the final activation map into a 1D vector.
3. `dense_layer` — implements the two fully connected layers (with an optional
   ReLU on the hidden layer).
4. `softmax_layer` — generates normalised probabilities from the logits.

Each block is currently hand-written with loop pipelining and BRAM buffering
pragma to control resource usage.【F:FPGA/hls_cnn/cnn_inference.cpp†L1-L202】【F:FPGA/hls_cnn/cnn_inference.cpp†L202-L242】

## 2. Candidate Replacements in Vitis Libraries

| Kernel Stage | Relevant Vitis Library | Notes |
|--------------|------------------------|-------|
| Convolution | `xf::cv::filter2D` / `xf::cv::Conv2D` (Vision L1) | Supports multi-channel 2D convolution on streaming images. Would require separating the fused BN/ReLU/Pool pipeline and reshaping tensors into the expected video stream (AXI4-Stream) format. |
| Batch Normalisation + ReLU | `xf::blas::relu` (BLAS L1) | ReLU is available as a vector primitive; batch norm would need either a simple custom kernel or embedding the scale/offset multiply-add inside a GEMM pre-processing step. |
| Max Pooling | `xf::cv::MaxPool` (Vision L1) | Operates on tiled images with configurable pooling window/stride. Needs explicit hand-off between convolution output and the pooling primitive. |
| Dense Layers | `xf::blas::gemv` / `xf::blas::gemm` (BLAS L1) | Fully connected layers map naturally to matrix–vector multiply. GEMM kernels already support fixed-point data types and block-level tiling pragmas to balance DSP usage. |
| Softmax | `xf::blas::softmax` (BLAS L1 Transformer primitives) | Provides a numerically stable exponential/normalisation pipeline; expect to convert the logits vector to the library’s stream interface. |

## 3. Integration Considerations

1. **Interface alignment**: Many Vision L1 operators consume and emit AXI4-Stream
   video frames. The current design uses on-chip arrays. Introducing library
   operators therefore means instantiating stream adapters (e.g. `hls::stream<ap_axiu<...>>`)
   and potentially rewriting the top-level dataflow to pass streams between
   primitives.
2. **Quantisation parameters**: Library operators must be configured with the
   same fixed-point width/fract bits (`ap_fixed<CNN_HLS_TOTAL_BITS, CNN_HLS_INTEGER_BITS>`)
   that are emitted by the Python export. Ensure template parameters match to
   avoid overflow.
3. **Fusion trade-offs**: The existing kernel fuses convolution, BN, ReLU, and
   pooling to reduce memory traffic. Breaking the fusion to use library blocks
   may increase buffering requirements. Evaluate whether the resource/performance
   trade-off is acceptable before committing.
4. **Verification**: Library blocks are verified, but substituting them will
   change the scheduling of the dataflow graph. Re-run C-simulation, co-sim, and
   on-board validation with the provided Python regression traces to ensure
   numerical parity.

## 4. When to Prefer the Custom Kernel

- If you require tightly fused layers to minimise BRAM usage or latency, the
  bespoke implementation may remain preferable.
- The current kernel already integrates batch norm and pooling without extra
  passes over memory; splitting them can increase latency unless carefully
  scheduled.
- The project exports weight headers tailored to the fused kernel ordering. Any
  library substitution should also update the exporter so tensor layouts match
  the library’s expectations.

## 5. Suggested Migration Path

1. Prototype replacing the dense layers with `xf::blas::gemv` (least intrusive —
   you only need to pack the flattened tensor into a stream and invoke the BLAS
   kernel).
2. If additional acceleration is required, migrate the final convolution block
   to Vision L1 `Conv2D` + `MaxPool`, leaving earlier blocks fused to limit data
   movement.
3. Once streaming interfaces are in place, evaluate the softmax replacement to
   take advantage of BLAS reductions and underflow-resistant exponentiation.

These steps let you incrementally adopt Vitis Libraries where they provide a
clear productivity or maintenance win, while preserving the proven fused kernel
for the remainder of the network until the interface rework is justified.
