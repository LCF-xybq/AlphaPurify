#!/usr/bin/env python3
"""
One-shot sync of all index data (d/w/m frequencies).
"""
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.data.index import sync_index

if __name__ == "__main__":
    logger.info("Starting index data sync...")
    for freq in ["d", "w", "m"]:
        try:
            count = sync_index(freq)
            logger.info(f"Index sync ({freq}) completed: {count} rows")
        except Exception as e:
            logger.exception(f"Index sync ({freq}) failed: {e}")
    logger.info("All index data sync completed")
