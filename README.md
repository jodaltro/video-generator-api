# Video Generator API (Wan2.1)

API para geração de vídeos a partir de texto usando o modelo leve **Wan2.1-T2V-1.3B**.

## 🎥 Características

- **Modelo Wan2.1-T2V-1.3B**: Modelo leve de texto-para-vídeo (1.3B parâmetros) da Wan-AI
- **Baixo consumo de memória**: ~4GB VRAM (vs ~30GB do modelo anterior LTX-2)
- **Sem necessidade de token HF**: Modelo público, sem autenticação necessária
- **Sem geração de áudio**: Apenas vídeo, otimizando recursos
- **Diffusers**: Usa a biblioteca HuggingFace diffusers (sem dependências customizadas)
- **Gerenciamento de Memória**: Cache clearing automático e fragmentação reduzida
- **CPU Offloading**: Opção para offload de modelos para CPU (para GPUs com memória limitada)
- **Download automático**: Modelo baixado automaticamente na primeira inicialização do pod
- **Documentação Swagger**: Interface interativa em `/` (raiz)
- **Pronto para RunPod/Pod GPU**: Otimizado para deployment em pods GPU

## ⚠️ Requisitos de GPU

- **Mínimo**: 8GB VRAM (RTX 3060, RTX 4060)
  - Resolução padrão: `480x320` com ~33 frames (~2s)
- **Recomendado**: 12GB+ VRAM (RTX 3060 Ti, RTX 4070)
  - Permite resoluções maiores e mais frames

## 🚀 Início Rápido

### Opção 1: Docker (Recomendado)

```bash
# Build da imagem
./build_and_push.sh

# Executar a API (dependências e modelo são baixados automaticamente na primeira execução)
docker run --gpus all -p 8000:8000 \
  -v workspace-data:/workspace \
  -v pip-packages:/usr/local/lib/python3.10/dist-packages \
  -v pip-bin:/usr/local/bin \
  video-generator-api:latest
```

### Opção 2: Build e Push para Docker Hub

```bash
# Build e push
./build_and_push.sh --push --username jodaltrorc

# Ou com tag específica
./build_and_push.sh --push --username seu-usuario --tag v3.0
```

### Opção 3: Docker Compose

```bash
docker-compose up -d
```

## 📦 Modelo Necessário (Wan2.1)

O script `download_models.py` baixa automaticamente o modelo do HuggingFace:

| Modelo | Descrição | Tamanho |
|--------|-----------|---------|
| `Wan2.1-T2V-1.3B-Diffusers/` | Modelo texto-para-vídeo Wan2.1 (1.3B parâmetros) | ~3-4 GB |

> ✅ **Não requer token HuggingFace** - o modelo Wan2.1 é público.

## 📡 API Endpoints

### Gerar Vídeo

**POST** `/generate`

Gera um vídeo a partir de uma descrição em texto.

#### Request Body

```json
{
  "prompt": "A serene sunset over the ocean with waves gently rolling",
  "duration": 2.0,
  "fps": 16,
  "width": 480,
  "height": 320,
  "num_inference_steps": 25,
  "guidance_scale": 5.0,
  "seed": 42
}
```

#### Parâmetros

| Parâmetro | Tipo | Obrigatório | Padrão | Descrição |
|-----------|------|-------------|---------|-----------|
| `prompt` | string | Sim | - | Descrição textual do vídeo |
| `duration` | float | Não | 2.0 | Duração em segundos (1-10) |
| `num_frames` | int | Não | auto | Número de frames (8-81) |
| `fps` | int | Não | 16 | Frames por segundo (8-30) |
| `width` | int | Não | 480 | Largura em pixels (256-1280, divisível por 8) |
| `height` | int | Não | 320 | Altura em pixels (256-1280, divisível por 8) |
| `num_inference_steps` | int | Não | 25 | Steps de denoising (4-100) |
| `guidance_scale` | float | Não | 5.0 | Força de aderência ao prompt (1-20) |
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
  "message": "Video Generator API (Wan2.1) is running"
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
./build_and_push.sh --push --username seu-usuario --tag v3.0

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
- **GPU**: Qualquer NVIDIA com 8GB+ VRAM
- **Porta**: 8000
- **Volume**: Montar volume em `/workspace/models`

### 3. Variáveis de Ambiente

```bash
PORT=8000
MODELS_DIR=/workspace/models
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 4. Primeira Execução

O modelo e dependências Python são baixados automaticamente na primeira inicialização do pod.

Modelo baixado automaticamente:
1. Wan2.1-T2V-1.3B-Diffusers (~3-4 GB)

**Tempo estimado**: 2-5 minutos dependendo da velocidade da rede.

> **Nota**: Se o modelo já estiver presente no volume persistente, o download é ignorado e a API inicia imediatamente.

## 📝 Exemplos de Uso

### cURL

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing with a ball of yarn in a cozy living room, warm afternoon light streaming through the window",
    "duration": 2.0,
    "fps": 16,
    "width": 480,
    "height": 320,
    "num_inference_steps": 25,
    "guidance_scale": 5.0
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
        "duration": 2.0,
        "fps": 16,
        "width": 480,
        "height": 320,
        "num_inference_steps": 25,
        "guidance_scale": 5.0
    }
)

with open("video.mp4", "wb") as f:
    f.write(response.content)
```

## ⚙️ Requisitos de Sistema

### Mínimo

- **GPU**: NVIDIA com 8 GB VRAM (RTX 3060/4060)
- **RAM**: 16 GB
- **Storage**: 10 GB (para modelo)
- **CUDA**: 12.1+

### Recomendado (RunPod/Pod)

- **GPU**: NVIDIA RTX 4070 (12 GB) ou superior
- **RAM**: 32 GB+
- **Storage**: 20 GB SSD

## 🐛 Troubleshooting

### Erro: "CUDA out of memory"

- Reduza resolução: tente 320x256
- Reduza `num_frames`: 16-24
- Ative CPU offload: `ENABLE_CPU_OFFLOAD=true`

### Modelos não encontrados

```bash
# Baixe o modelo primeiro
python download_models.py
```

### Vídeos muito lentos

- Certifique-se de usar GPU NVIDIA com CUDA
- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 📄 Licença

Este projeto usa:
- **Wan2.1**: Apache 2.0 License (Wan-AI)
- **FastAPI**: MIT License

## 📞 Suporte

Para problemas ou dúvidas, abra uma issue no GitHub.
