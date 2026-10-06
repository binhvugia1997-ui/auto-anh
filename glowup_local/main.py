"""Application entry point."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from .services.logging_setup import configure_logging
from .ui.main_window import MainWindow


def main() -> int:
    try:
        log_path = configure_logging()
    except OSError:
        # Logging should degrade gracefully on a read-only profile; image processing remains local.
        logging.basicConfig(level=logging.INFO)
        log_path = None
    logger = logging.getLogger(__name__)
    logger.info("Application entry point; log=%s", log_path or "stderr")
    app = QApplication(sys.argv)
    app.setApplicationName("GlowUp Local")
    app.setOrganizationName("GlowUp Local")
    app.setApplicationDisplayName("GlowUp Local")
    try:
        window = MainWindow()
        window.show()
        return app.exec()
    except Exception as exc:
        logger.exception("Application startup failed")
        QMessageBox.critical(
            None, "GlowUp Local", f"The application could not start.\n\n{exc}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
