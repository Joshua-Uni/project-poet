import logging
import sys

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging once at app startup."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),  # print to terminal
    ]

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
