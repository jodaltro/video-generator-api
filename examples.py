#!/usr/bin/env python3
"""
Example usage of the Video Generator API (LTX-2)
"""
import requests
import json
import time
import os


def example_basic_generation():
    """Basic video generation example"""
    print("=" * 60)
    print("Example 1: Basic Video Generation")
    print("=" * 60)

    url = "http://localhost:8000/generate"

    payload = {
        "prompt": "A beautiful sunset over mountains with birds flying across the golden sky, warm light illuminating the peaks",
        "duration": 3.0,
        "width": 512,
        "height": 768,
    }

    print(f"\nRequest payload:")
    print(json.dumps(payload, indent=2))

    print("\nGenerating video (this may take 1-3 minutes)...")
    start = time.time()

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        filename = "example_basic.mp4"
        with open(filename, "wb") as f:
            f.write(response.content)
        elapsed = time.time() - start
        print(f"✓ Video saved to: {filename}")
        print(f"  Time taken: {elapsed:.1f} seconds")
        print(f"  File size: {len(response.content) / 1024 / 1024:.2f} MB")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


def example_custom_parameters():
    """Advanced video generation with custom parameters"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Parameters")
    print("=" * 60)

    url = "http://localhost:8000/generate"

    payload = {
        "prompt": "A cat playing with a ball of yarn in a cozy living room, soft afternoon light streaming through the window, the cat pounces and rolls with the yarn",
        "duration": 2.0,
        "fps": 25,
        "width": 512,
        "height": 768,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "seed": 12345  # For reproducibility
    }

    print(f"\nRequest payload:")
    print(json.dumps(payload, indent=2))

    print("\nGenerating video...")
    start = time.time()

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        filename = "example_custom.mp4"
        with open(filename, "wb") as f:
            f.write(response.content)
        elapsed = time.time() - start
        print(f"✓ Video saved to: {filename}")
        print(f"  Time taken: {elapsed:.1f} seconds")
        print(f"  File size: {len(response.content) / 1024 / 1024:.2f} MB")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


def example_multiple_videos():
    """Generate multiple videos with different prompts"""
    print("\n" + "=" * 60)
    print("Example 3: Multiple Videos")
    print("=" * 60)

    url = "http://localhost:8000/generate"

    prompts = [
        "A peaceful lake with mountains reflecting in the still water, gentle morning mist rising",
        "City traffic at night with headlight trails streaking across the intersection, neon signs glowing",
        "Ocean waves crashing on a rocky beach, white foam spraying into the air against a blue sky"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Generating: {prompt[:60]}...")

        payload = {
            "prompt": prompt,
            "duration": 2.0,
            "width": 512,
            "height": 768,
            "num_inference_steps": 40
        }

        start = time.time()
        response = requests.post(url, json=payload)
        elapsed = time.time() - start

        if response.status_code == 200:
            filename = f"example_multi_{i}.mp4"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"  ✓ Saved to {filename} ({elapsed:.1f}s)")
        else:
            print(f"  ✗ Failed: {response.status_code}")


def check_api_status():
    """Check if API is running"""
    print("Checking API status...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is running")
            print(f"  Status: {data['status']}")
            print(f"  Model loaded: {data['model_loaded']}")
            print(f"  Device: {data['device']}")
            print(f"  Models dir: {data['models_dir']}")
            return True
        else:
            print(f"✗ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is it running on http://localhost:8000?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all examples"""
    print("Video Generator API (LTX-2) - Usage Examples")
    print("=" * 60)
    print()

    # Check API status first
    if not check_api_status():
        print("\nPlease start the API first:")
        print("  docker run --gpus all -p 8000:8000 -v models-cache:/workspace/models video-generator-api:latest")
        print("\nOr using Docker Compose:")
        print("  docker-compose up")
        return 1

    print("\nNote: First video generation will be slower as the model loads.")
    print("Subsequent generations will be much faster.\n")

    try:
        # Run examples
        example_basic_generation()
        example_custom_parameters()
        example_multiple_videos()

        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        print("\nGenerated files:")
        for f in os.listdir("."):
            if f.startswith("example_") and f.endswith(".mp4"):
                size = os.path.getsize(f) / 1024 / 1024
                print(f"  - {f} ({size:.2f} MB)")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
