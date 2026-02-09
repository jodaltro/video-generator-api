#!/usr/bin/env python3
"""
Test script for Video Generator API (Wan2.1)
"""
import requests
import sys
import time


def test_health_check(base_url: str = "http://localhost:8000"):
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"✓ Health check passed: {data}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_generate_video(base_url: str = "http://localhost:8000"):
    """Test video generation endpoint"""
    print("\n🎬 Testing video generation endpoint...")
    print("Note: First generation will take longer as models are loaded...")

    payload = {
        "prompt": "A serene sunset over the ocean with waves gently rolling, golden light reflecting on the water surface",
        "duration": 2.0,
        "fps": 16,
        "width": 480,
        "height": 320,
        "num_inference_steps": 25,
        "guidance_scale": 5.0,
        "seed": 42
    }

    try:
        print(f"Sending request with payload: {payload}")
        start_time = time.time()

        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            timeout=600  # 10 minutes timeout for model loading + generation
        )
        response.raise_for_status()

        elapsed_time = time.time() - start_time

        # Save the video
        output_file = "test_output.mp4"
        with open(output_file, "wb") as f:
            f.write(response.content)

        file_size = len(response.content) / 1024 / 1024  # MB
        print(f"✓ Video generated successfully!")
        print(f"  - Time taken: {elapsed_time:.2f} seconds")
        print(f"  - File size: {file_size:.2f} MB")
        print(f"  - Saved to: {output_file}")
        return True

    except requests.exceptions.Timeout:
        print("✗ Request timed out. This might happen on first run or slow hardware.")
        print("  Try increasing the timeout or check the server logs.")
        return False
    except Exception as e:
        print(f"✗ Video generation failed: {e}")
        return False


def main():
    """Run all tests"""
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"Testing Video Generator API (Wan2.1) at {base_url}\n")
    print("=" * 60)

    # Test health check
    if not test_health_check(base_url):
        print("\n❌ Health check failed. Is the server running?")
        return 1

    # Test video generation
    if not test_generate_video(base_url):
        print("\n❌ Video generation test failed.")
        return 1

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
