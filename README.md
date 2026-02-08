# Video Generator API (LTX-2)

API para geração de vídeos a partir de texto usando o modelo **LTX-2** da Lightricks.

## 🎥 Características

- **Modelo LTX-2**: Usa o modelo de última geração LTX-2 (19B parâmetros) da Lightricks
- **DistilledPipeline**: Pipeline otimizado para inferência rápida (8 sigmas predefinidos)
- **FP8 Support**: Suporte a FP8 para menor uso de memória VRAM
- **Download automático**: Modelos baixados automaticamente na primeira inicialização do pod
- **Documentação Swagger**: Interface interativa em `/` (raiz)
- **Pronto para RunPod/Pod GPU**: Otimizado para deployment em pods GPU

## 🚀 Início Rápido

### Opção 1: Docker (Recomendado)

```bash
# Build da imagem
./build_and_push.sh

# Executar a API (dependências e modelos são baixados automaticamente na primeira execução)
docker run --gpus all -p 8000:8000 \
  -v workspace-data:/workspace \
  -v pip-packages:/usr/local/lib/python3.10/dist-packages \
  -v pip-bin:/usr/local/bin \
  -v ltx2-repo:/opt/LTX-2 \
  video-generator-api:latest
```

### Opção 2: Build e Push para Docker Hub

```bash
# Build e push
./build_and_push.sh --push --username jodaltrorc

# Ou com tag específica
./build_and_push.sh --push --username seu-usuario --tag v2.0
```

### Opção 3: Docker Compose

```bash
docker-compose up -d
```

## 📦 Modelos Necessários (LTX-2)

O script `download_models.py` baixa automaticamente os seguintes modelos do HuggingFace:

| Modelo | Descrição | Tamanho |
|--------|-----------|---------|
| `ltx-2-19b-distilled-fp8.safetensors` | Checkpoint LTX-2 (distilled FP8) | ~10 GB |
| `ltx-2-spatial-upscaler-x2-1.0.safetensors` | Upscaler espacial 2x | ~2 GB |
| `ltx-2-19b-distilled-lora-384.safetensors` | LoRA destilado | ~1 GB |
| `gemma-3-12b-it-qat-q4_0-unquantized/` | Encoder de texto Gemma 3 (⚠️ **gated model**) | ~15 GB |

> ⚠️ **IMPORTANTE**: O modelo Gemma 3 é um **gated model** no HuggingFace que requer autenticação. Você DEVE:
>
> 1. **Solicitar acesso**: https://huggingface.co/google/gemma-3-12b-it (aprovação geralmente instantânea)
> 2. **Criar token**: https://huggingface.co/settings/tokens (permissão "Read")
> 3. **Configurar variável de ambiente**: `HF_TOKEN=hf_xxxxxxxxxxxx`

### Download Manual

```bash
# Configurar token antes
export HF_TOKEN=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

# Usando o script (dentro do container)
python download_models.py
```

### Executar com Token HuggingFace

```bash
# Docker
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=hf_xxxxxxxxxxxx \
  -v workspace-data:/workspace \
  video-generator-api:latest

# Docker Compose (adicione ao .env ou docker-compose.yml)
HF_TOKEN=hf_xxxxxxxxxxxx docker-compose up -d
```

## 📡 API Endpoints

### Gerar Vídeo

**POST** `/generate`

Gera um vídeo a partir de uma descrição em texto.

#### Request Body

```json
{
  "prompt": "A serene sunset over the ocean with waves gently rolling",
  "duration": 3.0,
  "fps": 25,
  "width": 512,
  "height": 768,
  "num_inference_steps": 40,
  "guidance_scale": 3.0,
  "seed": 42
}
```

#### Parâmetros

| Parâmetro | Tipo | Obrigatório | Padrão | Descrição |
|-----------|------|-------------|---------|-----------|
| `prompt` | string | Sim | - | Descrição textual do vídeo |
| `duration` | float | Não | 3.0 | Duração em segundos (1-10) |
| `num_frames` | int | Não | auto | Número de frames (8-257) |
| `fps` | int | Não | 25 | Frames por segundo (8-60) |
| `width` | int | Não | 512 | Largura em pixels (256-1280, divisível por 8) |
| `height` | int | Não | 768 | Altura em pixels (256-1280, divisível por 8) |
| `num_inference_steps` | int | Não | 40 | Steps de denoising (4-100) |
| `guidance_scale` | float | Não | 3.0 | Força de aderência ao prompt (1-20) |
| `seed` | int | Não | random | Seed para reprodutibilidade |

#### Response

Retorna um arquivo MP4 diretamente (download).

### Health Check

**GET** `/health`

Verifica o status da API e se o modelo está carregado.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "models_dir": "/workspace/models",
  "message": "Video Generator API (LTX-2) is running"
}
```

## 📚 Swagger/OpenAPI

Acesse a documentação interativa:

- **Swagger UI**: http://localhost:8000/
- **ReDoc**: http://localhost:8000/redoc

## 🐳 Build e Push Local

O projeto usa um script de build local (sem GitHub Actions):

```bash
# Apenas build
./build_and_push.sh

# Build e push
./build_and_push.sh --push --username seu-usuario

# Build com tag e push
./build_and_push.sh --push --username seu-usuario --tag v2.0

# Ver opções
./build_and_push.sh --help
```

## ☁️ Deployment em Pod GPU (RunPod)

### 1. Build e Push

```bash
./build_and_push.sh --push --username seu-usuario
```

### 2. Configuração do Pod

- **Imagem Docker**: `seu-usuario/video-generator-api:latest`
- **GPU**: Recomendado NVIDIA A40 ou superior (48GB+ VRAM)
- **Porta**: 8000
- **Volume**: Montar volume em `/workspace/models`

### 3. Variáveis de Ambiente

```bash
PORT=8000
MODELS_DIR=/workspace/models
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 4. Primeira Execução

Os modelos e dependências Python são baixados automaticamente na primeira inicialização do pod. Basta criar o pod com a imagem Docker e ele começará a instalar as dependências, baixar os modelos e iniciar a API.

Modelos baixados automaticamente:
1. Checkpoint LTX-2 (~10 GB)
2. Spatial Upscaler (~2 GB)
3. Distilled LoRA (~1 GB)
4. Gemma 3 Text Encoder (~15 GB)

**Tempo estimado**: 10-30 minutos dependendo da velocidade da rede.

> **Nota**: Se os modelos já estiverem presentes no volume persistente, o download é ignorado e a API inicia imediatamente.

## 🔧 Configuração Avançada

### Escolha de Checkpoint

O LTX-2 oferece várias opções de checkpoint:

| Checkpoint | Descrição | VRAM |
|------------|-----------|------|
| `ltx-2-19b-distilled-fp8` | Mais rápido, menor memória (padrão) | ~20 GB |
| `ltx-2-19b-distilled` | Rápido, FP32 | ~40 GB |
| `ltx-2-19b-dev-fp8` | Maior qualidade, FP8 | ~20 GB |
| `ltx-2-19b-dev` | Maior qualidade, FP32 | ~40 GB |

Para trocar o checkpoint, altere a variável `LTX2_CHECKPOINT` no `.env`.

### Estrutura dos Modelos

```
/workspace/models/
├── ltx-2-19b-distilled-fp8.safetensors
├── ltx-2-spatial-upscaler-x2-1.0.safetensors
├── ltx-2-19b-distilled-lora-384.safetensors
└── gemma-3-12b-it-qat-q4_0-unquantized/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

## 📝 Exemplos de Uso

### cURL

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing with a ball of yarn in a cozy living room, warm afternoon light streaming through the window",
    "duration": 3.0,
    "fps": 25,
    "width": 512,
    "height": 768,
    "num_inference_steps": 40,
    "guidance_scale": 3.0
  }' \
  --output video.mp4
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "A cat playing with a ball of yarn in a cozy living room",
        "duration": 3.0,
        "fps": 25,
        "width": 512,
        "height": 768,
        "num_inference_steps": 40,
        "guidance_scale": 3.0
    }
)

with open("video.mp4", "wb") as f:
    f.write(response.content)
```

## ⚙️ Requisitos de Sistema

### Mínimo

- **GPU**: NVIDIA com 24 GB VRAM (RTX 3090/4090)
- **RAM**: 32 GB
- **Storage**: 50 GB (para modelos)
- **CUDA**: 12.1+

### Recomendado (RunPod/Pod)

- **GPU**: NVIDIA A40 (48 GB) ou A100 (80 GB)
- **RAM**: 64 GB+
- **Storage**: 100 GB SSD

## 🐛 Troubleshooting

### Erro: "CUDA out of memory"

- Use checkpoint FP8 (`ltx-2-19b-distilled-fp8`)
- Reduza resolução: tente 384x512
- Reduza `num_frames`: 60-80

### Modelos não encontrados

```bash
# Baixe os modelos primeiro
python download_models.py
```

### Vídeos muito lentos

- Certifique-se de usar GPU NVIDIA com CUDA
- Use a DistilledPipeline (padrão) para inferência mais rápida
- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 📄 Licença

Este projeto usa:
- **LTX-2**: Apache 2.0 License (Lightricks)
- **Gemma 3**: Google Terms of Service
- **FastAPI**: MIT License

## 📞 Suporte

Para problemas ou dúvidas, abra uma issue no GitHub.
