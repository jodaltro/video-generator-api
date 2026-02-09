#!/usr/bin/env bash
#
# Build and Push Docker Image for Video Generator API (LTX-2)
#
# Usage:
#   ./build_and_push.sh                     # Build only
#   ./build_and_push.sh --push              # Build and push
#   ./build_and_push.sh --push --tag v2.0   # Build, tag and push
#
# Environment variables:
#   DOCKER_USERNAME  - Docker Hub username (required for push)
#   IMAGE_NAME       - Image name (default: video-generator-api)
#   TAG              - Image tag (default: latest)
#

set -e

# Defaults
IMAGE_NAME="${IMAGE_NAME:-video-generator-api}"
TAG="${TAG:-latest}"
PUSH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --username)
            DOCKER_USERNAME="$2"
            shift 2
            ;;
        --name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --push              Push image to Docker Hub after building"
            echo "  --tag TAG           Image tag (default: latest)"
            echo "  --username USER     Docker Hub username"
            echo "  --name NAME         Image name (default: video-generator-api)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  DOCKER_USERNAME     Docker Hub username (required for push)"
            echo "  IMAGE_NAME          Image name (default: video-generator-api)"
            echo "  TAG                 Image tag (default: latest)"
            echo ""
            echo "Examples:"
            echo "  $0                                  # Build only"
            echo "  $0 --push --username myuser          # Build and push"
            echo "  $0 --push --tag v2.0 --username myuser  # Build with tag and push"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  Video Generator API - Docker Build"
echo "=========================================="
echo ""
echo "Image: ${IMAGE_NAME}:${TAG}"

# Build the Docker image
echo ""
echo "🔨 Building Docker image..."
docker build -t "${IMAGE_NAME}:${TAG}" .

echo ""
echo "✅ Build completed: ${IMAGE_NAME}:${TAG}"

# Push if requested
if [ "$PUSH" = true ]; then
    if [ -z "$DOCKER_USERNAME" ]; then
        echo ""
        echo "❌ Error: DOCKER_USERNAME is required for push"
        echo "   Set it with: export DOCKER_USERNAME=your-username"
        echo "   Or use: $0 --push --username your-username"
        exit 1
    fi

    FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

    echo ""
    echo "🏷️  Tagging: ${FULL_IMAGE}"
    docker tag "${IMAGE_NAME}:${TAG}" "${FULL_IMAGE}"

    echo ""
    echo "🔐 Logging in to Docker Hub..."
    docker login

    echo ""
    echo "📤 Pushing: ${FULL_IMAGE}"
    docker push "${FULL_IMAGE}"

    # Also push as latest if tag is not already latest
    if [ "$TAG" != "latest" ]; then
        LATEST_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
        echo ""
        echo "🏷️  Also tagging as latest: ${LATEST_IMAGE}"
        docker tag "${IMAGE_NAME}:${TAG}" "${LATEST_IMAGE}"
        docker push "${LATEST_IMAGE}"
    fi

    echo ""
    echo "✅ Push completed: ${FULL_IMAGE}"
fi

echo ""
echo "=========================================="
echo "  Done!"
echo "=========================================="
echo ""
echo "To run locally (volumes persist dependencies and models across restarts):"
echo "  docker run --gpus all -p 8000:8000 -v workspace-data:/workspace -v pip-packages:/usr/local/lib/python3.10/dist-packages -v pip-bin:/usr/local/bin ${IMAGE_NAME}:${TAG}"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose up"
echo ""
