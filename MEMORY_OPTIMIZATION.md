# Memory Optimization Guide for LTX-2 Video Generation

## The Problem
LTX-2 is a large model that requires significant GPU memory (~23GB). When generating videos, you may encounter CUDA out of memory errors.

## Solutions Implemented

### 1. **Automatic GPU Cache Clearing** ✅
The API now automatically clears GPU cache before each generation, freeing fragmented memory.

- **Enabled by default**
- To disable: Set `CLEAR_CACHE_BEFORE_GENERATION=false` in environment

### 2. **Expandable Memory Segments** ✅
PyTorch memory allocator is now configured to use expandable segments, reducing fragmentation.

- **Automatically configured**
- Uses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 3. **CPU Offloading** (Optional)
Move model components to CPU when not in use, reducing peak GPU memory usage.

⚠️ **Note**: This may slow down generation but allows larger models to run on limited GPU memory.

To enable:
```bash
export ENABLE_CPU_OFFLOAD=true
```

Or in `.env`:
```
ENABLE_CPU_OFFLOAD=true
```

### 4. **Memory Usage Logging** ✅
The API now logs GPU memory usage at key stages:
- Before cache clearing
- After cache clearing
- After model loading
- During generation

## Recommended Settings for Limited GPU Memory

### For 24GB GPU (Current Issue)
```bash
# .env or environment variables
ENABLE_CPU_OFFLOAD=false  # Try without first
CLEAR_CACHE_BEFORE_GENERATION=true

# In API request, use smaller dimensions:
{
  "prompt": "Your prompt",
  "width": 384,      # Reduced from 512
  "height": 576,     # Reduced from 768
  "num_frames": 49,  # Reduced from 75-121
  "duration": 2.0,   # Shorter duration
  "fps": 25
}
```

### For 16GB GPU
```bash
ENABLE_CPU_OFFLOAD=true  # Required
CLEAR_CACHE_BEFORE_GENERATION=true

# Use minimal dimensions:
{
  "width": 320,
  "height": 512,
  "num_frames": 25,
  "duration": 1.0,
  "fps": 25
}
```

## Memory Usage by Video Configuration

Approximate GPU memory required (with FP8 enabled):

| Resolution | Frames | Memory |
|------------|--------|--------|
| 320x512    | 25     | ~12GB  |
| 384x576    | 49     | ~16GB  |
| 512x768    | 75     | ~20GB  |
| 512x768    | 121    | ~23GB  |
| 768x1280   | 121    | ~40GB+ |

## Troubleshooting

### Still Getting OOM Errors?

1. **Reduce resolution**: Use smaller width/height
2. **Reduce frames**: Generate shorter videos
3. **Enable CPU offload**: `ENABLE_CPU_OFFLOAD=true`
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
ENABLE_CPU_OFFLOAD=false              # Enable CPU offloading (slower but uses less GPU memory)
CLEAR_CACHE_BEFORE_GENERATION=true    # Clear GPU cache before each generation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Set automatically

# Model Settings
MODELS_DIR=/workspace/models           # Directory containing model files

# Performance
ENABLE_FP8=true                       # Use FP8 precision (recommended for memory efficiency)
```

## Example API Requests

### Conservative (Safe for 24GB)
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene sunset over the ocean",
    "width": 384,
    "height": 576,
    "num_frames": 49,
    "fps": 25,
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "seed": 42
  }'
```

### Aggressive (Needs 24GB+ or CPU offload)
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape with mountains",
    "width": 512,
    "height": 768,
    "num_frames": 121,
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

The API now includes automatic memory management improvements. For immediate relief from OOM errors:

1. ✅ **Already enabled**: Cache clearing and memory fragment reduction
2. 🔧 **Try reducing**: Video resolution and frame count in your requests
3. 🔧 **Optional**: Enable CPU offload if needed (`ENABLE_CPU_OFFLOAD=true`)

These changes should allow you to generate videos successfully on 24GB GPUs.
