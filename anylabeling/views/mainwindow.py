"""This module defines the main application window"""

from PyQt5.QtWidgets import QMainWindow, QStatusBar, QVBoxLayout, QWidget
import shutil

from ..app_info import __appdescription__, __appname__, __version__
from .labeling.label_wrapper import LabelingWrapper
from .labeling.logger import logger


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(
        self,
        app,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        super().__init__()
        self.app = app
        self.config = config

        self.setContentsMargins(0, 0, 0, 0)
        self.setWindowTitle(__appname__)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.labeling_widget = LabelingWrapper(
            self,
            config=config,
            filename=filename,
            output=output,
            output_file=output_file,
            output_dir=output_dir,
        )
        main_layout.addWidget(self.labeling_widget)
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        status_bar = QStatusBar()
        status_bar.showMessage(
            f"{__appname__} v{__version__} - {__appdescription__}"
        )
        self.setStatusBar(status_bar)

        # Check if ffmpeg is available via imageio-ffmpeg
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            logger.info(f"ffmpeg is available via imageio-ffmpeg: {ffmpeg_path}")
        except Exception as e:
            logger.warning(f"ffmpeg check failed: {e}. Video loading using ffmpeg might not work.")

    def closeEvent(self, event):
        self.labeling_widget.closeEvent(event)
