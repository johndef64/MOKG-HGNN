"""Simple reusable logger helper.

Usage:
	from multiomics_gnn.utils.logger import get_logger
	logger = get_logger(__name__, level=logging.INFO, log_to_file='app.log')
"""
from typing import Optional
import logging


DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: Optional[str] = None,
			   level: int = logging.INFO,
			   log_to_console: bool = True,
			   log_to_file: Optional[str] = None,
			   fmt: Optional[str] = None) -> logging.Logger:
	"""Return a configured logger instance.

	- Ensures handlers are not duplicated on repeated calls.
	- By default logs to console at INFO level.

	Args:
		name: logger name (defaults to root if None).
		level: logging level (e.g., logging.DEBUG).
		log_to_console: whether to add a StreamHandler.
		log_to_file: optional file path to log to.
		fmt: optional log message format.

	Returns:
		Configured ``logging.Logger`` instance.
	"""
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.propagate = False

	# Avoid adding handlers multiple times
	if logger.handlers:
		return logger

	formatter = logging.Formatter(fmt or DEFAULT_FORMAT)

	if log_to_console:
		ch = logging.StreamHandler()
		ch.setLevel(level)
		ch.setFormatter(formatter)
		logger.addHandler(ch)

	if log_to_file:
		fh = logging.FileHandler(log_to_file)
		fh.setLevel(level)
		fh.setFormatter(formatter)
		logger.addHandler(fh)

	return logger
def set_log_to_file(logger: logging.Logger, log_file: str):
	"""Add a file handler to an existing logger."""
	fh = logging.FileHandler(log_file)
	fh.setLevel(logger.level)
	formatter = logging.Formatter(DEFAULT_FORMAT)
	fh.setFormatter(formatter)
	logger.addHandler(fh)
