import logging
import sys
import traceback
from pathlib import Path
from typing import Optional, TextIO, Any
from contextlib import contextmanager
import threading

class _TeeStream:
    """Thread-safe stream that writes to both terminal and log file."""
    
    def __init__(self, original_stream: TextIO, log_file: TextIO):
        self.original_stream = original_stream
        self.log_file = log_file
        self._lock = threading.Lock()
    
    def write(self, text: str) -> int:
        with self._lock:
            # Write to original stream (terminal)
            result = self.original_stream.write(text)
            self.original_stream.flush()
            
            # Write to log file
            try:
                self.log_file.write(text)
                self.log_file.flush()
            except (ValueError, OSError):
                # Log file might be closed, ignore
                pass
            
            return result
    
    def flush(self):
        with self._lock:
            try:
                self.original_stream.flush()
            except (ValueError, OSError):
                pass
            try:
                self.log_file.flush()
            except (ValueError, OSError):
                pass
    
    def fileno(self):
        """Delegate fileno() to the original stream for compatibility with libraries like alive-progress."""
        return self.original_stream.fileno()
    
    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to the original stream."""
        return getattr(self.original_stream, name)

class MotifLogger:
    """Logger for motif-specific logs with stdout capture."""

    def __init__(self, motif_dir: Path, motif_id: str, log_to_terminal: bool = False, capture_stdout: bool = True):
        self.motif_dir = motif_dir
        self.motif_id = motif_id
        self.log_path = motif_dir / "motif.log"
        self.log_to_terminal = log_to_terminal
        self.capture_stdout = capture_stdout
        self.logger: Optional[logging.Logger] = None
        self.log_file: Optional[TextIO] = None
        self.file_handler: Optional[logging.FileHandler] = None

        # Stream redirection - store thread-local copies
        import threading
        self._thread_local = threading.local()
        self._thread_local.original_stdout = None
        self._thread_local.original_stderr = None
        self._thread_local.tee_stdout = None
        self._thread_local.tee_stderr = None

        motif_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        """Set up the logger and stream redirection when entering context."""
        try:
            # Open log file
            self.log_file = open(self.log_path, 'a', encoding='utf-8')
            
            # Set up structured logging
            logger_name = f"motif.{self.motif_id}.{id(self)}"
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.DEBUG)
            self.logger.handlers.clear()
            
            # Create file handler for structured logs
            self.file_handler = logging.FileHandler(self.log_path, mode='a', encoding='utf-8')
            self.file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                fmt='[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)
            
            # Add terminal handler if requested
            if self.log_to_terminal:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                console_formatter = logging.Formatter(f'[{self.motif_id}] %(message)s')
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
            
            self.logger.propagate = False
            
            # Set up stdout/stderr capture if requested
            if self.capture_stdout:
                # Store original streams in thread-local storage
                self._thread_local.original_stdout = sys.stdout
                self._thread_local.original_stderr = sys.stderr

                # Create tee streams that delegate fileno() and other methods to original streams
                self._thread_local.tee_stdout = _TeeStream(self._thread_local.original_stdout, self.log_file)
                self._thread_local.tee_stderr = _TeeStream(self._thread_local.original_stderr, self.log_file)

                # Only redirect if we haven't been redirected by another logger in this thread
                if not hasattr(sys.stdout, '_motif_logger_id') or sys.stdout._motif_logger_id != id(self):
                    sys.stdout = self._thread_local.tee_stdout
                    sys.stderr = self._thread_local.tee_stderr
                    # Mark the streams with this logger's ID
                    sys.stdout._motif_logger_id = id(self)
                    sys.stderr._motif_logger_id = id(self)
            
            self.logger.info(f"Starting motif {self.motif_id} processing")
            return self
            
        except Exception as e:
            # Use print as fallback since logging might not be set up
            print(f"Failed to set up motif logger: {e}", file=sys.__stderr__)
            self._cleanup()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up logger and streams when exiting context."""
        try:
            # Log exception if occurred
            if self.logger and exc_type:
                self.logger.error(f"Exception occurred during motif processing: {exc_val}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            if self.logger:
                self.logger.info(f"Finished processing motif {self.motif_id}")
            
        finally:
            self._cleanup()
            # Don't suppress exceptions
            return False

    def _cleanup(self):
        """Clean up resources and restore original streams."""
        try:
            # Restore original streams first - only if this logger owns the current redirection
            if self.capture_stdout:
                if (hasattr(sys.stdout, '_motif_logger_id') and
                    sys.stdout._motif_logger_id == id(self)):
                    if self._thread_local.original_stdout is not None:
                        sys.stdout = self._thread_local.original_stdout
                    if self._thread_local.original_stderr is not None:
                        sys.stderr = self._thread_local.original_stderr

            # Clean up handlers
            if self.logger:
                for handler in self.logger.handlers[:]:
                    try:
                        handler.close()
                        self.logger.removeHandler(handler)
                    except (ValueError, OSError):
                        pass

            # Close log file
            if self.log_file is not None:
                try:
                    self.log_file.close()
                except (ValueError, OSError):
                    pass
                self.log_file = None

            # Reset thread-local references
            self._thread_local.tee_stdout = None
            self._thread_local.tee_stderr = None
            self._thread_local.original_stdout = None
            self._thread_local.original_stderr = None
            self.file_handler = None
            
        except Exception as e:
            print(f"Error during MotifLogger cleanup: {e}", file=sys.__stderr__)

    def _log_if_available(self, level: str, message: str) -> None:
        """Helper to log if logger is available."""
        if self.logger:
            getattr(self.logger, level)(message)

    def info(self, message: str) -> None:
        self._log_if_available('info', message)

    def warning(self, message: str) -> None:
        self._log_if_available('warning', message)

    def error(self, message: str) -> None:
        self._log_if_available('error', message)

    def debug(self, message: str) -> None:
        self._log_if_available('debug', message)

    def write(self, message: str) -> None:
        if self.logger:
            self.logger.info(message.strip())


@contextmanager
def motif_logging_context(motif_id: str, log_path: Path, log_to_terminal: bool = False, capture_stdout: bool = True):
    """Context manager for motif logging"""
    motif_dir = log_path.parent
    logger_instance = MotifLogger(motif_dir, motif_id, log_to_terminal, capture_stdout)
    try:
        with logger_instance as logger:
            yield logger
    except Exception as e:
        print(f"Error in motif logging context: {e}", file=sys.__stderr__)
        raise 