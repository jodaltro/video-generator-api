.PHONY: help install run test docker-build docker-run docker-push clean

help:
	@echo "Video Generator API - Available commands:"
	@echo ""
	@echo "  make install        - Install Python dependencies"
	@echo "  make run            - Run the API locally"
	@echo "  make test           - Run tests"
	@echo "  make examples       - Run usage examples"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo "  make docker-push    - Push Docker image to registry"
	@echo "  make clean          - Clean generated files"
	@echo ""

install:
	pip install -r requirements.txt

run:
	python main.py

test:
	python test_api.py

examples:
	python examples.py

docker-build:
	docker build -t video-generator-api:latest .

docker-run:
	docker run -p 8000:8000 video-generator-api:latest

docker-run-gpu:
	docker run --gpus all -p 8000:8000 video-generator-api:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

docker-push:
	@echo "Please tag and push manually:"
	@echo "  docker tag video-generator-api:latest your-username/video-generator-api:latest"
	@echo "  docker push your-username/video-generator-api:latest"

clean:
	rm -f *.mp4 *.avi *.mov
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

download-models:
	python download_models.py
