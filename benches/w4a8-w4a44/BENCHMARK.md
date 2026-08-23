# NVFP4 Activation Quantization Benchmark

Date: 2026-08-23<br>
GPU: NVIDIA GeForce RTX 5090, 32 GB<br>
Driver: 610.88

## Compared builds

| Configuration | Folder | Build | Activation flag |
|---|---|---:|---|
| Master | `F:\llama-cpp\Release` | `4310aa4f8` (10101) | None |
| W4A8 | `F:\llama-cpp\alpha` | `723dc3efe` (10621) | `--nvfp4-w4a8` |
| W4A44 | `F:\llama-cpp\alpha` | `723dc3efe` (10621) | `--nvfp4-w4a44` |

Master is the unquantized-activation reference path. W4A8 and W4A44 use current binaries built from the `nvfp4-repack-3` branch. The W4A8 TMA load path is built in and does not require a separate flag.

## Models and settings

Each KLD/speed pair uses the same model file.

| Model | GGUF | KLD corpus/base |
|---|---|---|
| Gemma 4 31B | `gemma4-31B-it-Q6_K_RSF-nvfp4-amax.gguf` | `bartowski-calibration.txt` / `gemma4-31B-it-bf16.gguf.kld` |
| Qwen 3.8 27B | `qwen3.8-27B-nvfp4-dynamic.gguf` | `wiki.train.raw` / `qwen3.8-bf16.gguf.kld` |

KLD used 120 chunks, context 2048, batch 2048, ubatch 512, flash attention on, automatic GPU offload, and the specified BF16 reference KLD. Speed used five measured repetitions, FP16 KV cache, flash attention on, ubatch 1024, PP8192, TG256, and full GPU offload. Gemma speed used batch 4096; Qwen speed used batch 8192.

## KLD and PPL

Lower is better for PPL ratio distance from 1.0, KLD, and RMS Δp. Higher is better for same-top-token rate.

### Absolute results

| Model | Configuration | PPL(Q) | PPL(Q)/PPL(base) | Mean KLD | Maximum KLD | p99.9 KLD | p99 KLD | RMS Δp | Same top p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma | Master | 1412.920184 | 1.561780 | 0.814149 | 38.269611 | 22.990194 | 12.653790 | 15.898% | 77.545% |
| Gemma | W4A8 | 1447.902002 | 1.600447 | 0.663381 | 34.832638 | 22.181372 | 11.458063 | 14.363% | 80.052% |
| Gemma | W4A44 | 1450.966315 | 1.603835 | 0.613280 | 36.404282 | 21.776144 | 10.968298 | 13.862% | 80.969% |
| Qwen | Master | 6.646344 | 1.004067 | 0.024517 | 27.594637 | 1.593813 | 0.216252 | 4.230% | 93.950% |
| Qwen | W4A8 | 6.617122 | 0.999652 | 0.015100 | 26.111181 | 0.859331 | 0.123170 | 3.329% | 95.248% |
| Qwen | W4A44 | 6.634260 | 1.002241 | 0.013852 | 21.525572 | 0.830650 | 0.108429 | 3.161% | 95.448% |

### Change relative to master

Negative KLD and RMS changes are improvements. PPL changes are directional and must be interpreted with the ratio to the BF16 base; closeness of that ratio to 1.0 is the relevant agreement measure. Same-top-token changes are relative percentage changes, not percentage-point changes.

| Model | Configuration | PPL Δ | Mean KLD Δ | Maximum KLD Δ | p99.9 KLD Δ | p99 KLD Δ | RMS Δp Δ | Same top p Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma | W4A8 | +2.48% | -18.52% | **-8.98%** | -3.52% | -9.45% | -9.66% | +3.23% |
| Gemma | W4A44 | +2.69% | **-24.67%** | -4.87% | **-5.28%** | **-13.32%** | **-12.81%** | **+4.42%** |
| Qwen | W4A8 | -0.44% | -38.41% | -5.38% | -46.08% | -43.04% | -21.30% | +1.38% |
| Qwen | W4A44 | -0.18% | **-43.50%** | **-21.99%** | **-47.88%** | **-49.86%** | **-25.27%** | **+1.59%** |

### Additional KLD tail

| Model | Configuration | p95 KLD | Δ vs master | p90 KLD | Δ vs master |
|---|---|---:|---:|---:|---:|
| Gemma | Master | 4.636336 | — | 2.006351 | — |
| Gemma | W4A8 | 3.711325 | -19.95% | 1.447602 | -27.85% |
| Gemma | W4A44 | 3.383404 | **-27.02%** | 1.293658 | **-35.52%** |
| Qwen | Master | 0.066713 | — | 0.038863 | — |
| Qwen | W4A8 | 0.038672 | -42.03% | 0.022571 | -41.92% |
| Qwen | W4A44 | 0.033462 | **-49.84%** | 0.019451 | **-49.95%** |

## PP8192 and TG256 speed

Throughput is tokens per second. Values are the mean and standard deviation over five measured repetitions.

### Absolute results

| Model | Configuration | PP8192 | TG256 |
|---|---|---:|---:|
| Gemma | Master | 4579.80 ± 4.54 | 68.65 ± 0.09 |
| Gemma | W4A8 | 4045.79 ± 3.57 | 66.94 ± 0.09 |
| Gemma | W4A44 | 4244.26 ± 5.09 | 65.62 ± 0.07 |
| Qwen | Master | 4893.88 ± 2.76 | 72.67 ± 0.17 |
| Qwen | W4A8 | 4437.64 ± 3.91 | 70.61 ± 0.07 |
| Qwen | W4A44 | 4597.45 ± 3.47 | 69.46 ± 0.08 |

### Change relative to master

| Model | Configuration | PP8192 Δ | TG256 Δ |
|---|---|---:|---:|
| Gemma | W4A8 | -11.66% | **-2.49%** |
| Gemma | W4A44 | **-7.33%** | -4.41% |
| Qwen | W4A8 | -9.32% | **-2.83%** |
| Qwen | W4A44 | **-6.06%** | -4.42% |

## Reproduction commands

Run the master commands from `F:\llama-cpp\Release`. Run W4A8 and W4A44 from `F:\llama-cpp\alpha` and append the corresponding activation flag.

### Gemma KLD

```powershell
.\llama-perplexity.exe -m F:\llama-cpp\models\gemma4-31B-it-Q6_K_RSF-nvfp4-amax.gguf -f ..\models\bartowski-calibration.txt --kl-divergence-base F:\llama-cpp\models\gemma4-31B-it-bf16.gguf.kld -c 2048 -b 2048 -ub 512 -fa on -ngl auto --chunks 120 --kl-divergence
```

### Qwen KLD

```powershell
.\llama-perplexity.exe -m F:\llama-cpp\models\qwen3.8-27B-nvfp4-dynamic.gguf -f ..\models\wiki.train.raw --kl-divergence-base F:\llama-cpp\models\qwen3.8-bf16.gguf.kld -c 2048 -b 2048 -ub 512 -fa on -ngl auto --chunks 120 --kl-divergence
```

### Gemma speed

```powershell
.\llama-bench.exe -m F:\llama-cpp\models\gemma4-31B-it-Q6_K_RSF-nvfp4-amax.gguf -ngl 999 -fa on -ctk f16 -ctv f16 -b 4096 -ub 1024 -p 8192 -n 256 -r 5 -o md
```

### Qwen speed

```powershell
.\llama-bench.exe -m F:\llama-cpp\models\qwen3.8-27B-nvfp4-dynamic.gguf -ngl 999 -fa on -ctk f16 -ctv f16 -b 8192 -ub 1024 -p 8192 -n 256 -r 5 -o md
```

For alpha runs, append exactly one of:

```text
--nvfp4-w4a8
--nvfp4-w4a44
```
