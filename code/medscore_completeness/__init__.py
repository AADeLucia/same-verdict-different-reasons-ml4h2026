# Initialization code for the package
import logging
import os
import typing

# Prevent "No handler found" warnings for library users
logging.getLogger(__name__).addHandler(logging.NullHandler())


def configure_logging(
        log_level: int = logging.INFO,
        fmt: str = "%(asctime)s : %(filename)s : %(levelname)s : %(message)s",
        log_file: typing.Optional[str] = None,
        filemode: str = "a",
        also_console: bool = True,
) -> None:
    """
    Idempotent logging setup:
    - Does not add duplicate StreamHandlers or FileHandlers.
    - Updates format and level on existing handlers.
    """
    root = logging.getLogger()
    root.setLevel(log_level)
    formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Detect existing handlers
    abs_log_file = os.path.abspath(log_file) if log_file else None
    existing_stream = None
    existing_file = None

    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                if abs_log_file and os.path.abspath(getattr(h, "baseFilename", "")) == abs_log_file:
                    existing_file = h
            except Exception:
                pass
        elif isinstance(h, logging.StreamHandler):
            # Treat any non-FileHandler StreamHandler as the console handler
            existing_stream = existing_stream or h

    # Ensure file handler (avoid duplicates)
    if log_file:
        if existing_file is None:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            fh = logging.FileHandler(log_file, mode=filemode, encoding="utf-8")
            fh.setFormatter(formatter)
            fh.setLevel(log_level)
            root.addHandler(fh)
        else:
            existing_file.setFormatter(formatter)
            existing_file.setLevel(log_level)

    # Ensure console handler (avoid duplicates)
    if also_console:
        if existing_stream is None:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            sh.setLevel(log_level)
            root.addHandler(sh)
        else:
            existing_stream.setFormatter(formatter)
            existing_stream.setLevel(log_level)

    # Silence noisy third-party libraries by default for applications using this helper
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


__all__ = ["configure_logging"]