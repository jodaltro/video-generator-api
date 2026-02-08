# HuggingFace Token Setup - REQUIRED

## The Problem

The Gemma 3 model (`google/gemma-3-12b-it`) is a **gated model** on HuggingFace, which means it requires:
1. **User authentication** via a HuggingFace token
2. **Access approval** for the model (usually instant)

Without these, you'll see the error:
```
401 Client Error: Cannot access gated repo for url https://huggingface.co/google/gemma-3-12b-it/...
Access to model google/gemma-3-12b-it is restricted.
```

## Solution - 3 Simple Steps

### Step 1: Request Access to Gemma Model

1. Go to: https://huggingface.co/google/gemma-3-12b-it
2. Click the "Agree and access repository" button
3. Wait for approval (usually instant, but can take a few minutes)

### Step 2: Create HuggingFace Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "video-generator-api")
4. Select **"Read"** permission (sufficient for downloading models)
5. Click "Generate token"
6. **Copy the token** (starts with `hf_...`)

### Step 3: Set the Token in Your Environment

Choose the method that fits your deployment:

#### For Docker Run:
```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=hf_xxxxxxxxxxxx \
  -v workspace-data:/workspace \
  video-generator-api:latest
```

#### For Docker Compose:
```bash
# Option 1: Set in environment
export HF_TOKEN=hf_xxxxxxxxxxxx
HF_TOKEN=$HF_TOKEN docker-compose up

# Option 2: Create .env file (recommended)
echo "HF_TOKEN=hf_xxxxxxxxxxxx" > .env
docker-compose up
```

#### For RunPod:
Add this to your template's environment variables:
```
HF_TOKEN=hf_xxxxxxxxxxxx
```

#### For Local Development:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
python download_models.py
```

## What Changed

The following files have been updated to handle gated models properly:

1. **download_models.py** - Now checks for and uses `HF_TOKEN` for all downloads
2. **README.md** - Added warning and instructions about HF_TOKEN requirement
3. **RUNPOD_DEPLOYMENT.md** - Added token setup in template configuration
4. **QUICKSTART.md** - Added pre-requisite warning and token instructions
5. **docker-compose.yml** - Added HF_TOKEN environment variable
6. **.env.example** - Added HF_TOKEN with instructions

## Testing Your Setup

After setting the token, restart your container or run:

```bash
# Test the download script
export HF_TOKEN=hf_xxxxxxxxxxxx
python download_models.py
```

You should see:
```
✅ DONE tokenizer.model (took X.Xs)
✅ Model check completed successfully!
```

## Troubleshooting

### "401 Client Error" still appears
- Double-check your token is correct (copy/paste again)
- Ensure you requested access to the Gemma model
- Wait a few minutes if you just requested access

### "No HF_TOKEN found" warning
- Verify the environment variable is set: `echo $HF_TOKEN`
- Check the variable is passed to the container (use `docker inspect`)
- For RunPod, check template environment variables

### Token not working
- Ensure token has "Read" permission
- Try generating a new token
- Check token hasn't expired

## Security Note

**Never commit your HF_TOKEN to git!** 

The `.env` file and any files containing `HF_TOKEN` are automatically ignored by git (check `.gitignore`).

## Next Steps

Once you've set up the token:

1. **Rebuild if needed** (only if you modified Dockerfile):
   ```bash
   ./build_and_push.sh --push --username jodaltrorc
   ```

2. **Run the container** with the token:
   ```bash
   docker run --gpus all -p 8000:8000 \
     -e HF_TOKEN=hf_xxxxxxxxxxxx \
     -v workspace-data:/workspace \
     jodaltrorc/video-generator-api:latest
   ```

3. **Wait for model download** (~30GB, 10-30 minutes)

4. **Access the API** at http://localhost:8000

## Questions?

Check the updated documentation:
- [README.md](README.md) - General overview
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) - RunPod-specific guide
