import logging
from pathlib import Path
import time
from typing import Dict, Any
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("mix_analyzer")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(
        log_dir / f"mix_analyzer_{int(time.time())}.log"
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    # Console handler with tqdm support
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s: %(message)s')
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger