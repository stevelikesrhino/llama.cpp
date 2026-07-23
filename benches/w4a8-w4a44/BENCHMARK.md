# NVFP4 Activation Quantization Benchmark

Date: 2026-07-23  
GPU: NVIDIA GeForce RTX 5090, 32 GB  
Driver: 610.62

## Compared builds

| Configuration | Folder | Build | Activation flag |
|---|---|---:|---|
| Master | `F:\llama-cpp\Release` | `4310aa4f8` (10101) | None |
| W4A44 | `F:\llama-cpp\alpha` | `bb43b5038` (10118) | `--nvfp4-w4a44` |
| W4A8 | `F:\llama-cpp\alpha` | `bb43b5038` (10118) | `--nvfp4-w4a8` |

Master is the unquantized-activation reference path for this comparison. W4A44 and W4A8 use the latest merged alpha binaries.

## Models and settings

Each KLD/speed pair uses the same model file.

| Model | GGUF | KLD corpus/base |
|---|---|---|
| Gemma 4 31B | `gemma4-31B-it-Q6_K_RSF-nvfp4-amax.gguf` | `bartowski-calibration.txt` / `gemma4-31B-it-bf16.gguf.kld` |
| Qwen 3.6 27B | `qwen3.6-27B-nvfp4-amax-mtp-dynamic.gguf` | `wiki.train.raw` / `qwen3.6-bf16.gguf.kld` |

KLD used 120 chunks, context 2048, batch 2048, ubatch 512, flash attention on, and full GPU offload. Speed used five measured repetitions, FP16 KV cache, flash attention on, ubatch 1024, PP8192, and TG256. Gemma speed used batch 4096; Qwen speed used batch 8192.

## KLD and PPL

Lower is better for PPL ratio distance from 1.0, KLD, and RMS Δp. Higher is better for same-top-token rate.

### Absolute results

| Model | Configuration | PPL(Q) | PPL(Q)/PPL(base) | Mean KLD | Maximum KLD | p99.9 KLD | p99 KLD | RMS Δp | Same top p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma | Master | 1412.920184 | 1.561780 | 0.814149 | 38.269611 | 22.990194 | 12.653790 | 15.898% | 77.545% |
| Gemma | W4A44 | 1503.217813 | 1.661591 | 0.614236 | 37.540241 | 21.954350 | 10.881876 | 13.835% | 80.869% |
| Gemma | W4A8 | 1423.859926 | 1.573872 | 0.656019 | 37.708397 | 22.221085 | 11.197418 | 14.337% | 80.155% |
| Qwen | Master | 6.426297 | 1.005461 | 0.061668 | 30.135740 | 12.771930 | 0.607675 | 6.384% | 93.233% |
| Qwen | W4A44 | 6.388694 | 0.999578 | 0.037500 | 30.319977 | 8.705987 | 0.287214 | 4.794% | 95.088% |
| Qwen | W4A8 | 6.391456 | 1.000010 | 0.043650 | 30.070417 | 9.922916 | 0.350049 | 5.205% | 94.647% |

### Change relative to master

Negative KLD and RMS changes are improvements. PPL changes are directional and must be interpreted with the ratio to the BF16 base; closeness of that ratio to 1.0 is the relevant agreement measure. Same-top-token changes are relative percentage changes, not percentage-point changes.

| Model | Configuration | PPL Δ | Mean KLD Δ | Maximum KLD Δ | p99.9 KLD Δ | p99 KLD Δ | RMS Δp Δ | Same top p Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma | W4A44 | +6.39% | **-24.55%** | **-1.91%** | **-4.51%** | **-14.00%** | **-12.98%** | **+4.29%** |
| Gemma | W4A8 | +0.77% | -19.42% | -1.47% | -3.35% | -11.51% | -9.82% | +3.37% |
| Qwen | W4A44 | -0.59% | **-39.19%** | +0.61% | **-31.83%** | **-52.74%** | **-24.91%** | **+1.99%** |
| Qwen | W4A8 | -0.54% | -29.22% | **-0.22%** | -22.31% | -42.40% | -18.47% | +1.52% |

### Additional KLD tail

| Model | Configuration | p95 KLD | Δ vs master | p90 KLD | Δ vs master |
|---|---|---:|---:|---:|---:|
| Gemma | Master | 4.636336 | — | 2.006351 | — |
| Gemma | W4A44 | 3.390528 | **-26.87%** | 1.306113 | **-34.90%** |
| Gemma | W4A8 | 3.687301 | -20.47% | 1.441310 | -28.16% |
| Qwen | Master | 0.106876 | — | 0.053060 | — |
| Qwen | W4A44 | 0.051899 | **-51.44%** | 0.025631 | **-51.69%** |
| Qwen | W4A8 | 0.062307 | -41.70% | 0.030336 | -42.83% |

## PP8192 and TG256 speed

Throughput is tokens per second. Values are the mean and standard deviation over five measured repetitions.

### Absolute results

| Model | Configuration | PP8192 | TG256 |
|---|---|---:|---:|
| Gemma | Master | 4536.67 ± 14.86 | 68.74 ± 0.06 |
| Gemma | W4A44 | 4090.89 ± 13.27 | 65.46 ± 0.08 |
| Gemma | W4A8 | 3598.59 ± 12.69 | 65.74 ± 0.06 |
| Qwen | Master | 4982.34 ± 12.76 | 69.27 ± 0.06 |
| Qwen | W4A44 | 4587.03 ± 13.91 | 65.86 ± 0.07 |
| Qwen | W4A8 | 4129.00 ± 12.66 | 66.39 ± 0.04 |

### Change relative to master

| Model | Configuration | PP8192 Δ | TG256 Δ |
|---|---|---:|---:|
| Gemma | W4A44 | **-9.83%** | -4.77% |
| Gemma | W4A8 | -20.68% | **-4.36%** |
| Qwen | W4A44 | **-7.93%** | -4.92% |
| Qwen | W4A8 | -17.13% | **-4.16%** |

## Reproduction commands

Run the master commands from `F:\llama-cpp\Release`. Run W4A44/W4A8 from `F:\llama-cpp\alpha` and append the corresponding activation flag.

### Gemma KLD

```powershell
.\llama-perplexity.exe -m F:\llama-cpp\models\gemma4-31B-it-Q6_K_RSF-nvfp4-amax.gguf -f ..\models\bartowski-calibration.txt --kl-divergence-base F:\llama-cpp\models\gemma4-31B-it-bf16.gguf.kld -c 2048 -b 2048 -ub 512 -fa on -ngl 99 --chunks 120 --kl-divergence
```

### Qwen KLD

```powershell
.\llama-perplexity.exe -m F:\llama-cpp\models\qwen3.6-27B-nvfp4-amax-mtp-dynamic.gguf -f ..\models\wiki.train.raw --kl-divergence-base F:\llama-cpp\models\qwen3.6-bf16.gguf.kld -c 2048 -b 2048 -ub 512 -fa on -ngl 999 --chunks 120 --kl-divergence
```

### Gemma speed

```powershell
.\llama-bench.exe -m F:\llama-cpp\models\gemma4-31B-it-Q6_K_RSF-nvfp4-amax.gguf -ngl 999 -fa on -ctk f16 -ctv f16 -b 4096 -ub 1024 -p 8192 -n 256 -r 5 -o md
```

### Qwen speed

```powershell
.\llama-bench.exe -m F:\llama-cpp\models\qwen3.6-27B-nvfp4-amax-mtp-dynamic.gguf -ngl 999 -fa on -ctk f16 -ctv f16 -b 8192 -ub 1024 -p 8192 -n 256 -r 5 -o md
```

For alpha runs, append exactly one of:

```text
--nvfp4-w4a44
--nvfp4-w4a8
```
