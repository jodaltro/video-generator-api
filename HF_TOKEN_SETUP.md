# HuggingFace Token Setup

## No Token Required! ✅

The Wan2.1-T2V-1.3B model is a **public model** on HuggingFace and does **not** require authentication.

The previous LTX-2 setup required a HuggingFace token because it used the gated Gemma-3 12B text encoder. With the switch to Wan2.1, this is no longer needed.

## Optional: HuggingFace Token

If you want to use a HuggingFace token for other purposes (e.g., faster downloads, access to other models), you can still set one:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
```

### How to Get a Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "video-generator-api")
4. Select **"Read"** permission
5. Click "Generate token"
6. Copy the token (starts with `hf_...`)

## Security Note

**Never commit your HF_TOKEN to git!**

The `.env` file and any files containing `HF_TOKEN` are automatically ignored by git (check `.gitignore`).
