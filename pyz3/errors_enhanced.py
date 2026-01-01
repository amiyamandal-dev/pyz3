"""Enhanced error handling with detailed stack traces for pyz3.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StackFrame:
    """Represents a single frame in the stack trace."""

    file: str
    line: int
    function: str
    code: str | None = None
    is_zig: bool = False
    module: str | None = None


@dataclass
class ErrorInfo:
    """Enhanced error information with stack trace."""

    error_type: str
    message: str
    frames: list[StackFrame]
    cause: "ErrorInfo | None" = None

    def format(self, show_code: bool = True, colorize: bool = False) -> str:
        """Format the error for display."""
        lines = []

        # Header
        lines.append(f"\n{'=' * 60}")
        lines.append(f"Error: {self.error_type}")
        lines.append(f"Message: {self.message}")
        lines.append(f"{'=' * 60}\n")

        # Stack trace
        lines.append("Stack Trace (most recent call last):")
        lines.append("-" * 40)

        for i, frame in enumerate(self.frames):
            prefix = "→ " if i == len(self.frames) - 1 else "  "
            lang = "[Zig]" if frame.is_zig else "[Py] "

            lines.append(f"{prefix}{lang} {frame.file}:{frame.line}")
            lines.append(f"         in {frame.function}")

            if show_code and frame.code:
                lines.append(f"         > {frame.code.strip()}")

            lines.append("")

        # Cause chain
        if self.cause:
            lines.append("\nCaused by:")
            lines.append(self.cause.format(show_code=show_code, colorize=colorize))

        return "\n".join(lines)


class PyError(Exception):
    """Base class for enhanced Python errors from Zig extensions.

    This provides better stack trace integration between Zig and Python code.
    """

    def __init__(
        self,
        message: str,
        zig_source: str | None = None,
        zig_line: int | None = None,
        zig_function: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.zig_source = zig_source
        self.zig_line = zig_line
        self.zig_function = zig_function
        self._error_info: ErrorInfo | None = None

    @property
    def error_info(self) -> ErrorInfo:
        """Get detailed error information."""
        if self._error_info is None:
            self._error_info = self._build_error_info()
        return self._error_info

    def _build_error_info(self) -> ErrorInfo:
        """Build error info from current exception state."""
        frames = []

        # Add Zig frame if available
        if self.zig_source and self.zig_line:
            frames.append(
                StackFrame(
                    file=self.zig_source,
                    line=self.zig_line,
                    function=self.zig_function or "<unknown>",
                    is_zig=True,
                )
            )

        # Add Python frames
        _, _, tb = sys.exc_info()
        if tb:
            for frame_info in traceback.extract_tb(tb):
                frames.append(
                    StackFrame(
                        file=frame_info.filename,
                        line=frame_info.lineno,
                        function=frame_info.name,
                        code=frame_info.line,
                        is_zig=False,
                    )
                )

        return ErrorInfo(
            error_type=self.__class__.__name__,
            message=self.message,
            frames=frames,
        )

    def format_trace(self, show_code: bool = True) -> str:
        """Format the error with full stack trace."""
        return self.error_info.format(show_code=show_code)


class ZigValueError(PyError, ValueError):
    """ValueError originating from Zig code."""

    pass


class ZigTypeError(PyError, TypeError):
    """TypeError originating from Zig code."""

    pass


class ZigRuntimeError(PyError, RuntimeError):
    """RuntimeError originating from Zig code."""

    pass


class ZigIndexError(PyError, IndexError):
    """IndexError originating from Zig code."""

    pass


class ZigKeyError(PyError, KeyError):
    """KeyError originating from Zig code."""

    pass


class ZigAttributeError(PyError, AttributeError):
    """AttributeError originating from Zig code."""

    pass


class ZigOverflowError(PyError, OverflowError):
    """OverflowError originating from Zig code."""

    pass


class ZigMemoryError(PyError, MemoryError):
    """MemoryError originating from Zig code."""

    pass


class ZigZeroDivisionError(PyError, ZeroDivisionError):
    """ZeroDivisionError originating from Zig code."""

    pass


# Error type mapping
ERROR_TYPES = {
    "ValueError": ZigValueError,
    "TypeError": ZigTypeError,
    "RuntimeError": ZigRuntimeError,
    "IndexError": ZigIndexError,
    "KeyError": ZigKeyError,
    "AttributeError": ZigAttributeError,
    "OverflowError": ZigOverflowError,
    "MemoryError": ZigMemoryError,
    "ZeroDivisionError": ZigZeroDivisionError,
}


def create_error(
    error_type: str,
    message: str,
    zig_source: str | None = None,
    zig_line: int | None = None,
    zig_function: str | None = None,
) -> PyError:
    """Create an appropriate enhanced error based on type name.

    Args:
        error_type: Name of the Python exception type
        message: Error message
        zig_source: Source file in Zig code
        zig_line: Line number in Zig code
        zig_function: Function name in Zig code

    Returns:
        Appropriate PyError subclass instance
    """
    error_class = ERROR_TYPES.get(error_type, PyError)
    return error_class(
        message=message,
        zig_source=zig_source,
        zig_line=zig_line,
        zig_function=zig_function,
    )


def get_current_error_info() -> ErrorInfo | None:
    """Get enhanced error info for the current exception."""
    exc_type, exc_value, exc_tb = sys.exc_info()

    if exc_type is None:
        return None

    frames = []
    if exc_tb:
        for frame_info in traceback.extract_tb(exc_tb):
            frames.append(
                StackFrame(
                    file=frame_info.filename,
                    line=frame_info.lineno,
                    function=frame_info.name,
                    code=frame_info.line,
                    is_zig=_is_zig_frame(frame_info.filename),
                )
            )

    # Handle exception chaining
    cause = None
    if exc_value and exc_value.__cause__:
        cause_exc = exc_value.__cause__
        cause_frames = []
        if cause_exc.__traceback__:
            for frame_info in traceback.extract_tb(cause_exc.__traceback__):
                cause_frames.append(
                    StackFrame(
                        file=frame_info.filename,
                        line=frame_info.lineno,
                        function=frame_info.name,
                        code=frame_info.line,
                        is_zig=_is_zig_frame(frame_info.filename),
                    )
                )
        cause = ErrorInfo(
            error_type=type(cause_exc).__name__,
            message=str(cause_exc),
            frames=cause_frames,
        )

    return ErrorInfo(
        error_type=exc_type.__name__,
        message=str(exc_value) if exc_value else "",
        frames=frames,
        cause=cause,
    )


def _is_zig_frame(filename: str) -> bool:
    """Check if a frame is from Zig code (compiled extension)."""
    path = Path(filename)
    # Check for common extension patterns
    if path.suffix in (".zig", ".so", ".pyd", ".dylib"):
        return True
    # Check for abi3 extensions
    if ".abi3." in path.name:
        return True
    return False


def format_exception_with_zig_trace(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: type[BaseException] | None,
) -> str:
    """Format an exception with enhanced Zig/Python stack trace.

    This can be used as a replacement for traceback.format_exception.
    """
    lines = []

    lines.append("Traceback (most recent call last):")

    if exc_tb:
        for frame_info in traceback.extract_tb(exc_tb):
            lines.append(f'  File "{frame_info.filename}", line {frame_info.lineno}, in {frame_info.name}')
            if frame_info.line:
                lines.append(f"    {frame_info.line}")

    lines.append(f"{exc_type.__name__}: {exc_value}")

    return "\n".join(lines)


# Store original hook for restoration
_original_excepthook = None


def install_enhanced_hook() -> None:
    """Install enhanced exception hook for better Zig stack traces.

    Call this at module initialization to get enhanced stack traces
    throughout your application.
    """
    global _original_excepthook
    _original_excepthook = sys.excepthook

    def enhanced_hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: type[BaseException] | None,
    ) -> None:
        # Check if this is a PyError with enhanced info
        if isinstance(exc_value, PyError):
            print(exc_value.format_trace(), file=sys.stderr)
        else:
            # Use enhanced formatting for all exceptions
            print(
                format_exception_with_zig_trace(exc_type, exc_value, exc_tb),
                file=sys.stderr,
            )

    sys.excepthook = enhanced_hook


def uninstall_enhanced_hook() -> None:
    """Restore the original exception hook."""
    global _original_excepthook
    if _original_excepthook is not None:
        sys.excepthook = _original_excepthook
        _original_excepthook = None
