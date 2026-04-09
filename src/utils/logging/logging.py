"""Singleton logger with file and console output."""

import logging
import os
import datetime
from typing import Optional

from src.utils.base import Singleton
from src.settings import Settings


class Logger(metaclass=Singleton):
    """Application-wide logger that writes to both file and console.

    Uses the Singleton pattern to ensure a single logger instance per name.
    Log files are organized by month under the LOGS_DIR directory.
    """

    def __init__(self) -> None:
        self._loggers: dict[str, Optional[logging.Logger]] = {
            Settings.loggers.DAILY: None,
            Settings.loggers.BACKFILL: None,
        }

    def init_logger(self, logger_name: str) -> None:
        """Initialize a named logger with file and console handlers.

        Parameters
        ----------
        logger_name : str
            Name for the logger (used for the log filename).
        """
        log = logging.getLogger(logger_name)
        log_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        log_directory = Settings.LOGS_DIR / (
            f"{str.lower(datetime.date.today().strftime('%b'))}_"
            f"{datetime.date.today().strftime('%Y')}"
        )
        log_file_name = log_directory / f"{logger_name}.log"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        file_handler = logging.FileHandler(log_file_name, mode="a+")
        file_handler.setFormatter(log_formatter)
        console_handler = logging.StreamHandler()
        log.addHandler(file_handler)
        log.addHandler(console_handler)
        log.setLevel(logging.DEBUG)
        self._loggers[logger_name] = log

    def get_logger(self, logger_name: str) -> logging.Logger:
        """Get a logger by name, initializing it if needed.

        Parameters
        ----------
        logger_name : str
            Name of the logger to retrieve.

        Returns
        -------
        logging.Logger
            The configured logger instance.
        """
        if self._loggers.get(logger_name) is None:
            self.init_logger(logger_name)
        return self._loggers.get(logger_name)
