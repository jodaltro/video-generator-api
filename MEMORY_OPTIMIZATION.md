# Memory Optimization Guide for LTX-2 Video Generation

## 🚨 TL;DR - Quick Fix for 24GB GPUs

**If you're getting "CUDA out of memory" errors:**

1. ✅ **Enable Aggressive Offload** (enabled by default): `ENABLE_AGGRESSIVE_OFFLOAD=true`
   - Reduces peak GPU memory from ~23GB to ~12-14GB
   - Generation will be slower due to CPU↔GPU model transfers, but it works reliably on 24GB GPUs
2. ✅ **Use the safe defaults** (already configured in API)
3. ✅ **Test with minimum settings**:
   - Resolution: `320x512`
   - Frames: `33` (~1.3 seconds)
   - With aggressive offload: ~12-14GB GPU usage

4. ✅ With aggressive offload, you can now use `512x768` on 24GB GPUs!

---

## The Problem
LTX-2 is a large model that requires significant GPU memory. The model components (Text Encoder Gemma-3 12B, Transformer 19B, VAE) load into memory and can cause CUDA out of memory errors on GPUs with less than 32GB VRAM.

### Root Cause
When calling the generation endpoint, the DistilledPipeline lazy-loads all model components:
1. **Text Encoder (Gemma-3 12B)**: ~15GB when loading (~23GB total with transformer)
2. **Transformer (LTX-2 19B FP8)**: ~10GB
3. **VAE**: ~2GB
4. **Activation Memory**: Variable based on video resolution/frames

On a 24GB GPU, loading all components simultaneously causes OOM errors.

## Solutions Implemented

### 1. **Aggressive Sequential CPU↔GPU Offloading** ✅ (Recommended)
The API now supports aggressive sequential offloading that moves each model component to GPU only when needed, then offloads it to CPU or deletes it before loading the next component. This dramatically reduces peak GPU memory usage.

- **Enabled by default**: `ENABLE_AGGRESSIVE_OFFLOAD=true`
- **Peak GPU memory**: ~12-14GB (down from ~23GB)
- **Trade-off**: Generation is slower due to CPU↔GPU model transfers
- **How it works**:
  1. Load text encoder to GPU → encode text → delete from GPU
  2. Load video encoder + transformer to GPU → Stage 1 denoising
  3. Offload transformer to CPU → load spatial upsampler → upsample → delete upsampler
  4. Move transformer back to GPU → Stage 2 denoising → delete both
  5. Load video decoder → decode → delete
  6. Load audio decoder + vocoder → decode → delete

To enable/disable:
```bash
export ENABLE_AGGRESSIVE_OFFLOAD=true   # Enable (default)
export ENABLE_AGGRESSIVE_OFFLOAD=false  # Disable for faster generation on 32GB+ GPUs
```

### 2. **Automatic GPU Cache Clearing** ✅
The API now automatically clears GPU cache before each generation, freeing fragmented memory.

- **Enabled by default**
- To disable: Set `CLEAR_CACHE_BEFORE_GENERATION=false` in environment
- **Impact**: Minimal performance cost, significant memory benefit

### 3. **Expandable Memory Segments** ✅
PyTorch memory allocator is now configured to use expandable segments, reducing fragmentation.

- **Automatically configured**
- Uses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **Impact**: Better memory utilization, fewer fragmentation errors

### 4. **CPU Offloading (Legacy)** ⚠️ (Limited Support)
**Note**: The DistilledPipeline used by this API has limited CPU offloading support. Full offloading requires using the complete LTXVideoPipeline.

Current limitation:
- `ENABLE_CPU_OFFLOAD` parameter exists but is not fully effective with DistilledPipeline
- For production use on 24GB GPUs, **reduce video dimensions** instead

To enable experimental offload:
```bash
export ENABLE_CPU_OFFLOAD=true
```

⚠️ **This will NOT solve 24GB memory issues!** Use smaller dimensions instead.

### 5. **Memory Usage Logging** ✅
The API now logs GPU memory usage at key stages:
- Before cache clearing
- After cache clearing
- After model loading
- During generation

## Recommended Settings for Limited GPU Memory

### For 24GB GPU with Aggressive Offload (RECOMMENDED) ✅
```bash
# Aggressive offload is enabled by default!
# This uses ~12-14GB GPU, allowing higher resolutions
{
  "prompt": "Your prompt",
  "width": 512,      # Higher quality possible!
  "height": 768,     # Higher quality possible!
  "num_frames": 33,  # ~1.3 seconds at 25fps
  "duration": 1.3,
  "fps": 25
}
```

**Memory usage**: ~12-14GB peak (with aggressive offload)

### For 24GB GPU without Aggressive Offload
```bash
# Only if ENABLE_AGGRESSIVE_OFFLOAD=false
{
  "prompt": "Your prompt",
  "width": 320,      # Ultra-safe
  "height": 512,     # Ultra-safe
  "num_frames": 33,  # ~1.3 seconds at 25fps
  "duration": 1.3,
  "fps": 25
}
```

**Memory usage**: ~18-20GB (leaves 4-6GB buffer)

### For 32GB+ GPU (More Quality)
```bash
{
  "prompt": "Your prompt",
  "width": 512,
  "height": 768,
  "num_frames": 75,  # 3 seconds
  "duration": 3.0,
  "fps": 25
}
```

**Memory usage**: ~28-30GB

## Memory Usage by Video Configuration

Approximate GPU memory required (with FP8 enabled):

### With Aggressive Offload (default)

| Resolution | Frames | Duration | Peak Memory | Status on 24GB |
|------------|--------|----------|-------------|----------------|
| 320x512    | 33     | 1.3s     | ~12GB       | ✅ **Safe**    |
| 384x576    | 49     | 2.0s     | ~13GB       | ✅ **Safe**    |
| 512x768    | 33     | 1.3s     | ~14GB       | ✅ **Safe**    |
| 512x768    | 75     | 3.0s     | ~16GB       | ✅ **Safe**    |
| 512x768    | 121    | 4.8s     | ~18GB       | ⚠️  Risky      |
| 768x1280   | 121    | 4.8s     | ~30GB+      | ❌ OOM         |

### Without Aggressive Offload

| Resolution | Frames | Duration | Peak Memory | Status on 24GB |
|------------|--------|----------|-------------|----------------|
| 320x512    | 33     | 1.3s     | ~19GB       | ✅ **Safe**    |
| 384x576    | 49     | 2.0s     | ~22GB       | ⚠️  Risky      |
| 512x768    | 75     | 3.0s     | ~28GB       | ❌ OOM         |
| 512x768    | 121    | 4.8s     | ~32GB       | ❌ OOM         |
| 768x1280   | 121    | 4.8s     | ~50GB+      | ❌ OOM         |

**Note**: The Text Encoder (Gemma) alone uses ~15-23GB when loading!

## Troubleshooting

### Still Getting OOM Errors?

1. **Enable aggressive offload**: `ENABLE_AGGRESSIVE_OFFLOAD=true` (default)
2. **Reduce resolution**: Use smaller width/height
3. **Reduce frames**: Generate shorter videos
4. **Restart API**: Clear all cached models
   ```bash
   docker-compose restart
   ```

5. **Check GPU usage before generation**:
   ```bash
   nvidia-smi
   # Make sure no other processes are using GPU
   ```

### Kill Other GPU Processes
```bash
# Find processes using GPU
nvidia-smi

# Kill specific process
kill -9 <PID>
```

### Monitor Memory During Generation
```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Best Practices

1. **Start small**: Test with small resolutions first
2. **Single request at a time**: Don't queue multiple requests
3. **Monitor logs**: Check memory usage in API logs
4. **Use FP8**: Keep `enable_fp8=True` (default)
5. **Avoid concurrent requests**: Wait for completion before next request

## Environment Variables Reference

```bash
# Memory Optimization
ENABLE_AGGRESSIVE_OFFLOAD=true            # Sequential CPU↔GPU offloading (recommended for 24GB GPUs)
ENABLE_CPU_OFFLOAD=false                  # Legacy CPU offloading (limited with DistilledPipeline)
CLEAR_CACHE_BEFORE_GENERATION=true        # Clear GPU cache before each generation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Set automatically

# Model Settings
MODELS_DIR=/workspace/models               # Directory containing model files

# Performance
ENABLE_FP8=true                           # Use FP8 precision (recommended for memory efficiency)
```

## Example API Requests

### Ultra-Safe (Guaranteed for 24GB) ✅
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene sunset over the ocean",
    "width": 320,
    "height": 512,
    "num_frames": 33,
    "fps": 25,
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "seed": 42
  }'
```

### Higher Quality (Requires 32GB+)
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape with mountains",
    "width": 512,
    "height": 768,
    "num_frames": 75,
    "fps": 25,
    "num_inference_steps": 40,
    "guidance_scale": 3.0
  }'
```

## Additional Resources

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [LTX-2 Repository](https://github.com/Lightricks/LTX-Video)

## Summary

The API now includes aggressive memory management that reduces peak GPU usage from ~23GB to ~12-14GB, making it reliable on 24GB GPUs like the RTX 4090.

1. ✅ **Aggressive offload** (default): Sequential CPU↔GPU model transfers reduce peak memory by ~40%
2. ✅ **Already enabled**: Cache clearing and memory fragment reduction
3. 🔧 **Try reducing**: Video resolution and frame count if still hitting OOM
4. 🔧 **For 32GB+ GPUs**: Disable aggressive offload for faster generation (`ENABLE_AGGRESSIVE_OFFLOAD=false`)

These changes allow you to generate videos successfully on 24GB GPUs without needing more GPU memory.
