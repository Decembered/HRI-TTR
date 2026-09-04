"""Process-signal boundary for recoverable training interruption."""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import FrameType


@dataclass(slots=True)
class StopController:
    """Mutable stop request sampled only at optimizer-step boundaries."""

    requested: bool = False
    signal_number: int | None = None

    @property
    def exit_code(self) -> int:
        """Return the conventional nonzero process status for the signal."""
        number = self.signal_number if self.signal_number is not None else signal.SIGINT
        return 128 + int(number)

    def request(self, signal_number: int) -> None:
        """Record the first signal without throwing inside the active step."""
        if not self.requested:
            self.requested = True
            self.signal_number = signal_number

    def handle(self, signal_number: int, _frame: FrameType | None) -> None:
        """Adapt a Python signal callback to a boundary stop request."""
        self.request(signal_number)


@contextmanager
def installed_stop_controller() -> Generator[StopController, None, None]:
    """Install reversible SIGTERM/SIGINT handlers on the main thread."""
    controller = StopController()
    if threading.current_thread() is not threading.main_thread():
        yield controller
        return
    previous_term = signal.signal(signal.SIGTERM, controller.handle)
    previous_int = signal.signal(signal.SIGINT, controller.handle)
    try:
        yield controller
    finally:
        _ = signal.signal(signal.SIGTERM, previous_term)
        _ = signal.signal(signal.SIGINT, previous_int)
