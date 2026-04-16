import logging

from llm_hallucination_detector.settings import Settings


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.service.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
