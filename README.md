# LLM Hallucination Detector

Production-ready service that detects hallucinations in LLM answers by extracting atomic claims, retrieving evidence, and verifying each claim with NLI.

## What This Provides
- Claim extraction with a deterministic seq2seq model
- Evidence retrieval from Wikipedia and GDELT news
- Claim verification with NLI and a transparent score
- Persistent vector index with configurable caching
- Separate endpoints for claims, evidence, verification, and full detection

## Quickstart (Local)
```bash
pip install -r requirements.txt
set LHD_CONFIG_PATH=config/models.yaml
uvicorn llm_hallucination_detector.app:create_app --factory --host 0.0.0.0 --port 8000
```

## Docker
```bash
docker build -t llm-hallucination-detector .
docker run -p 8000:8000 -v %cd%/data:/app/data llm-hallucination-detector
```

## API Endpoints
- `GET /health`
- `POST /claims`
- `POST /evidence`
- `POST /verify`
- `POST /detect`

## Example Request
```bash
curl -X POST http://localhost:8000/detect \
	-H "Content-Type: application/json" \
	-d '{"question":"Who won the 2018 FIFA World Cup?","answer":"France won the 2018 FIFA World Cup, defeating Croatia 4-2 in Russia."}'
```

## Configuration
All runtime configuration lives in [config/models.yaml](config/models.yaml). You can override any field with env vars using `LHD__SECTION__KEY` (for example `LHD__SERVICE__PORT=8080`).
