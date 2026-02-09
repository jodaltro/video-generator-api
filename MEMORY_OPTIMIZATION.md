# Memory Optimization Guide for Wan2.1 Video Generation

## 🚨 TL;DR - Wan2.1 is already lightweight!

The Wan2.1-T2V-1.3B model requires significantly less memory than the previous LTX-2 setup:

- **Previous (LTX-2)**: ~30GB total (19B model + 12B text encoder)
- **Current (Wan2.1)**: ~4GB total (1.3B model with built-in text encoder)

For most users with a GPU of 8GB+ VRAM, the default settings will work without issues.

---

## Memory Usage

### Model Components

| Component | Size |
|-----------|------|
| Wan2.1-T2V-1.3B (FP16) | ~3 GB |
| VAE (FP32) | ~0.5 GB |
| Text Encoder (built-in) | ~0.5 GB |
| **Total** | **~4 GB** |

### By Video Configuration

| Resolution | Frames | Duration | Memory | Status on 8GB |
|------------|--------|----------|--------|----------------|
| 480x320    | 33     | 2.0s     | ~5GB   | ✅ **Safe**    |
| 480x320    | 49     | 3.0s     | ~6GB   | ✅ **Safe**    |
| 640x480    | 33     | 2.0s     | ~7GB   | ⚠️ Tight      |
| 720x480    | 49     | 3.0s     | ~8GB   | ⚠️ Tight      |

## Optimization Options

### 1. **Automatic GPU Cache Clearing** ✅
The API automatically clears GPU cache before each generation, freeing fragmented memory.

- **Enabled by default**
- To disable: Set `CLEAR_CACHE_BEFORE_GENERATION=false` in environment

### 2. **Expandable Memory Segments** ✅
PyTorch memory allocator is configured to use expandable segments, reducing fragmentation.

- **Automatically configured**
- Uses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 3. **CPU Offloading** ✅
For GPUs with very limited memory, enable CPU offloading:

```bash
export ENABLE_CPU_OFFLOAD=true
```

This moves model components to CPU when not in use, significantly reducing VRAM usage at the cost of speed.

## Recommended Settings for Limited GPU Memory

### For 8GB GPU ✅
```json
{
  "prompt": "Your prompt",
  "width": 480,
  "height": 320,
  "num_frames": 33,
  "duration": 2.0,
  "fps": 16
}
```

### For 12GB+ GPU (Higher Quality)
```json
{
  "prompt": "Your prompt",
  "width": 640,
  "height": 480,
  "num_frames": 49,
  "duration": 3.0,
  "fps": 16
}
```

## Troubleshooting

### Still Getting OOM Errors?

1. **Reduce resolution**: Use smaller width/height
2. **Reduce frames**: Generate shorter videos
3. **Enable CPU offload**: `ENABLE_CPU_OFFLOAD=true`
4. **Restart API**: Clear all cached models

## Environment Variables Reference

```bash
# Memory Optimization
ENABLE_CPU_OFFLOAD=false              # Enable CPU offloading (slower but uses less GPU memory)
CLEAR_CACHE_BEFORE_GENERATION=true    # Clear GPU cache before each generation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Set automatically

# Model Settings
MODELS_DIR=/workspace/models           # Directory containing model files
```

## Comparison: LTX-2 vs Wan2.1

| Feature | LTX-2 (Previous) | Wan2.1 (Current) |
|---------|-------------------|------------------|
| Parameters | 19B + 12B text encoder | 1.3B (all-in-one) |
| Model Size | ~30 GB | ~3-4 GB |
| Min VRAM | 24 GB | 8 GB |
| Audio | Yes | No |
| HF Token Required | Yes (gated Gemma) | No (public) |
| Custom Packages | Yes (ltx-pipelines) | No (diffusers) |
