# Video Generator API

API para geração de vídeos a partir de texto usando o modelo LTX-Video da Lightricks.

## 🎥 Características

- **Endpoint único**: `/generate` - Gera vídeos a partir de texto
- **Modelo moderno**: Usa LTX-Video (open-source) da Lightricks
- **Download sob demanda**: Modelos são baixados na primeira execução (imagem Docker leve)
- **Documentação Swagger**: Interface interativa em `/` (raiz)
- **Pronto para RunPod**: Otimizado para deployment em pods GPU

## 🚀 Início Rápido

### Opção 1: Docker (Recomendado)

```bash
# Build da imagem
docker build -t video-generator-api .

# Run do container
docker run -p 8000:8000 video-generator-api
```

### Opção 2: Local (Python)

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar API
python main.py
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
  "height": 512,
  "num_inference_steps": 30,
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
| `width` | int | Não | 512 | Largura em pixels (256-1024, divisível por 8) |
| `height` | int | Não | 512 | Altura em pixels (256-1024, divisível por 8) |
| `num_inference_steps` | int | Não | 30 | Steps de denoising (10-100) |
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
  "message": "Video Generator API is running"
}
```

## 📚 Swagger/OpenAPI

Acesse a documentação interativa:

- **Swagger UI**: http://localhost:8000/
- **ReDoc**: http://localhost:8000/redoc

## 🐳 Docker Hub

### Build e Push

```bash
# Build
docker build -t seu-usuario/video-generator-api:latest .

# Tag
docker tag video-generator-api seu-usuario/video-generator-api:latest

# Push
docker push seu-usuario/video-generator-api:latest
```

## ☁️ Deployment no RunPod

### 1. Configuração do Pod

- **Imagem Docker**: `seu-usuario/video-generator-api:latest`
- **GPU**: Recomendado NVIDIA A40 ou superior
- **Porta**: 8000
- **Volumes**: Opcional - montar volume para cache de modelos

### 2. Variáveis de Ambiente

```bash
PORT=8000
HF_HOME=/workspace/.cache/huggingface
```

### 3. Primeira Execução

Na primeira execução, o pod irá:
1. Baixar os modelos LTX-Video do HuggingFace (~10-15 GB)
2. Cachear os modelos no volume (se configurado)
3. Estar pronto para gerar vídeos

**Tempo estimado da primeira execução**: 10-20 minutos (dependendo da velocidade da rede)

## 🔧 Configuração Avançada

### Otimizações de Memória

Para GPUs com menos VRAM, o código já inclui:
- `enable_model_cpu_offload()`: Offload automático para CPU
- `enable_vae_slicing()`: Reduz uso de memória do VAE

### Cache de Modelos

Os modelos são baixados para:
- Local: `~/.cache/huggingface/`
- Docker: `/root/.cache/huggingface/`
- RunPod: `/workspace/.cache/huggingface/` (configurável)

## 📝 Exemplos de Uso

### cURL

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing with a ball of yarn",
    "duration": 3.0,
    "fps": 25,
    "width": 512,
    "height": 512,
    "num_inference_steps": 30,
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
        "prompt": "A cat playing with a ball of yarn",
        "duration": 3.0,
        "fps": 25,
        "width": 512,
        "height": 512,
        "num_inference_steps": 30,
        "guidance_scale": 3.0
    }
)

with open("video.mp4", "wb") as f:
    f.write(response.content)
```

### JavaScript (Node.js)

```javascript
const fs = require('fs');
const axios = require('axios');

async function generateVideo() {
  const response = await axios.post(
    'http://localhost:8000/generate',
    {
      prompt: 'A cat playing with a ball of yarn',
      duration: 3.0,
      fps: 25,
      width: 512,
      height: 512,
      num_inference_steps: 30,
      guidance_scale: 3.0
    },
    { responseType: 'arraybuffer' }
  );
  
  fs.writeFileSync('video.mp4', response.data);
}

generateVideo();
```

## ⚙️ Requisitos de Sistema

### Mínimo

- **CPU**: 4 cores
- **RAM**: 16 GB
- **GPU**: NVIDIA com 8 GB VRAM (opcional, mas recomendado)
- **Storage**: 20 GB (para modelos e cache)

### Recomendado (RunPod)

- **GPU**: NVIDIA A40, A100, ou RTX 4090
- **RAM**: 32 GB+
- **Storage**: 50 GB SSD

## 🐛 Troubleshooting

### Erro: "CUDA out of memory"

Reduza os parâmetros:
- `width` e `height`: tente 256x256
- `num_frames`: reduza para 40-60
- `num_inference_steps`: reduza para 20

### Modelo não carrega

Verifique:
1. Conexão com internet (primeira execução)
2. Espaço em disco disponível (mínimo 20 GB)
3. Logs do container: `docker logs <container-id>`

### Vídeos muito lentos para gerar

- Use GPU (tempo: ~1-2 min com A40)
- CPU será muito lento (tempo: ~20-30 min)

## 📄 Licença

Este projeto usa:
- **LTX-Video**: Apache 2.0 License (Lightricks)
- **FastAPI**: MIT License
- **Diffusers**: Apache 2.0 License

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:
1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📞 Suporte

Para problemas ou dúvidas, abra uma issue no GitHub.
