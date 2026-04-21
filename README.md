# LLM Hallucination Detector

Production-ready service that detects hallucinations in LLM answers by extracting atomic claims, retrieving evidence, and verifying each claim with NLI. It is built for product use: deterministic extraction, auditable evidence, and configurable persistence.

## Why This Exists
LLMs can sound confident while being wrong. This service makes hallucinations measurable by turning an answer into atomic claims, finding evidence, and scoring each claim with NLI. The output is a transparent, claim-level verdict plus a single hallucination score.

## Core Features
- Deterministic claim extraction with strict JSON output
- Evidence retrieval from Wikipedia and GDELT news
- NLI-based verification with a confidence score per claim
- Persistent vector index with configurable caching
- Separate endpoints for claims, evidence, verification, and full detection
- Fully local, open-source models by default

## How It Works
1. **Extract claims** from the answer using a seq2seq model with a strict prompt.
2. **Retrieve evidence** by combining BM25 lexical search and semantic embeddings.
3. **Verify claims** with an NLI model against the best evidence snippets.
4. **Score** the answer based on claim-level verdicts.

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

## API Examples

### Detect Hallucinations
```bash
curl -X POST http://localhost:8000/detect \
	-H "Content-Type: application/json" \
	-d '{"question":"Who won the 2018 FIFA World Cup?","answer":"France won the 2018 FIFA World Cup, defeating Croatia 4-2 in Russia."}'
```

Example response:
```json
{
	"hallucination_score": 0.0,
	"claims": [
		{
			"claim": "France won the 2018 FIFA World Cup.",
			"label": "entailed",
			"score": 0.92,
			"evidence": "France defeated Croatia 4-2 in the 2018 World Cup final..."
		}
	]
}
```

### Claim Extraction Only
```bash
curl -X POST http://localhost:8000/claims \
	-H "Content-Type: application/json" \
	-d '{"question":"Who won the 2018 FIFA World Cup?","answer":"France won the 2018 FIFA World Cup, defeating Croatia 4-2 in Russia."}'
```

### Evidence Retrieval Only
```bash
curl -X POST http://localhost:8000/evidence \
	-H "Content-Type: application/json" \
	-d '{"claim":"France won the 2018 FIFA World Cup.","top_k":5}'
```

### Verification Only
```bash
curl -X POST http://localhost:8000/verify \
	-H "Content-Type: application/json" \
	-d '{"claim":"France won the 2018 FIFA World Cup.","evidence":["France defeated Croatia 4-2 in the 2018 World Cup final."]}'
```

## Configuration
All runtime configuration lives in [config/models.yaml](config/models.yaml). You can override any field with env vars using `LHD__SECTION__KEY` (example: `LHD__SERVICE__PORT=8080`).

Key sections:
- `service`: API host, port, and log level
- `models`: claim extractor, verifier, and embeddings
- `retrieval`: BM25 and vector settings, chunk size
- `sources`: Wikipedia and GDELT settings
- `router`: topic-based source routing
- `index`: persistence and vector index behavior
- `cache`: evidence cache policy and auto-clear behavior
- `scoring`: how neutral or insufficient evidence affects score

## Persistence and Caching
- Evidence caches are disabled by default to avoid disk growth.
- If you enable disk caching, you can set `cache.clear_on_response=true` to wipe evidence caches after each response.
- The vector index is controlled separately under `index` and can still persist for warm restarts.

## Data and Privacy
This service does not send your data to paid APIs by default. Retrieval pulls from public sources; caches are stored on disk if enabled. Disable caching if you need zero retention.

## Performance Notes
- GPU is recommended for the claim extractor and NLI verifier.
- CPU mode is supported but slower for large prompts.

## Model Downloads and Cache Size
- Default models download on first run and are cached by Hugging Face.
- Expect roughly 4 to 6 GB for the default model set (claim extractor, NLI verifier, embeddings).
- Evidence caches (Wikipedia/news), if enabled, and the persistent vector index can add anywhere from hundreds of MB to many GB depending on usage.
- You can change model sizes and disable caching in [config/models.yaml](config/models.yaml).

## Project Layout
- `src/llm_hallucination_detector`: core library and service
- `config/models.yaml`: runtime configuration
- `experiments/claim_extraction/prompts`: extraction prompts
- `tests/`: unit tests

## Roadmap
- Web UI for paste-in prompts and evidence visualization
- Source connectors for custom document stores
- Cross-encoder re-ranking for higher precision

## Development
```bash
pip install -r requirements.txt
pytest
```
