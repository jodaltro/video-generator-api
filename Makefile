.PHONY: help install run test docker-build docker-run docker-push clean download-models

help:
	@echo "Video Generator API (LTX-2) - Available commands:"
	@echo ""
	@echo "  make install            - Install Python dependencies"
	@echo "  make run                - Run the API locally"
	@echo "  make test               - Run tests"
	@echo "  make examples           - Run usage examples"
	@echo "  make docker-build       - Build Docker image"
	@echo "  make docker-run         - Run Docker container (CPU)"
	@echo "  make docker-run-gpu     - Run Docker container (GPU)"
	@echo "  make docker-push        - Build and push Docker image"
	@echo "  make docker-compose-up  - Start with Docker Compose"
	@echo "  make docker-compose-down - Stop Docker Compose"
	@echo "  make download-models    - Download LTX-2 models"
	@echo "  make clean              - Clean generated files"
	@echo ""

install:
	pip install -r requirements.txt

run:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py

test:
	python test_api.py

examples:
	python examples.py

docker-build:
	./build_and_push.sh

docker-run:
	docker run -p 8000:8000 -v models-cache:/workspace/models video-generator-api:latest

docker-run-gpu:
	docker run --gpus all -p 8000:8000 -v models-cache:/workspace/models video-generator-api:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

docker-push:
	@echo "Usage: ./build_and_push.sh --push --username YOUR_DOCKER_USERNAME"
	@echo ""
	@echo "Or set DOCKER_USERNAME and run:"
	@echo "  ./build_and_push.sh --push"

clean:
	rm -f *.mp4 *.avi *.mov
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

download-models:
	python download_models.py
