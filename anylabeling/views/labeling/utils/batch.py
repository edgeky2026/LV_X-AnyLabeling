import base64
import json
import os.path as osp
from PIL import Image

from datetime import datetime

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, QThread
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QProgressDialog,
    QDialog,
    QLabel,
    QLineEdit,
    QDialogButtonBox,
    QApplication,
    QButtonGroup,
    QRadioButton,
)

from anylabeling.app_info import __version__
from anylabeling.services.auto_labeling import (
    _BATCH_PROCESSING_INVALID_MODELS,
    _BATCH_PROCESSING_TEXT_PROMPT_MODELS,
    _BATCH_PROCESSING_VIDEO_MODELS,
    _SKIP_DET_MODELS,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils._io import io_open
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import get_msg_box_style
from anylabeling.views.labeling.widgets.popup import Popup


__all__ = ["run_all_images"]


class TextInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Enter Text Prompt"))
        self.setFixedSize(400, 180)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        prompt_label = QLabel(self.tr("Please enter your text prompt:"))
        prompt_label.setStyleSheet(
            "font-size: 13px; color: #1d1d1f; font-weight: 500;"
        )
        layout.addWidget(prompt_label)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText(self.tr("Enter prompt here..."))
        layout.addWidget(self.text_input)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #ffffff;
                border-radius: 10px;
            }
            
            QLineEdit {
                border: 1px solid #E5E5E5;
                border-radius: 8px;
                background-color: #F9F9F9;
                font-size: 13px;
                height: 36px;
                padding: 0 12px;
            }
            
            QLineEdit:hover {
                background-color: #DBDBDB;
            }
            
            QLineEdit:focus {
                border: 2px solid #0066FF;
                background-color: #F9F9F9;
            }
            
            QPushButton {
                min-width: 100px;
                height: 36px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 13px;
            }
            
            QPushButton[text="OK"] {
                background-color: #0066FF;
                color: white;
                border: none;
            }
            
            QPushButton[text="OK"]:hover {
                background-color: #0077ED;
            }
            
            QPushButton[text="OK"]:pressed {
                background-color: #0068D0;
            }
            
            QPushButton[text="Cancel"] {
                background-color: #f5f5f7;
                color: #1d1d1f;
                border: 1px solid #d2d2d7;
            }
            
            QPushButton[text="Cancel"]:hover {
                background-color: #e5e5e5;
            }
            
            QPushButton[text="Cancel"]:pressed {
                background-color: #d5d5d5;
            }
        """
        )

    def get_input_text(self):
        if self.exec_() == QDialog.Accepted:
            return self.text_input.text().strip()
        return ""


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

class VideoExportThread(QThread):
    def __init__(self, image_list, output_dir):
        super().__init__()
        self.image_list = list(image_list)
        self.output_dir = output_dir

    def run(self):
        try:
            import cv2
            import numpy as np

            if self.image_list and len(self.image_list) > 0:
                dir_path = osp.dirname(self.image_list[0])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_output_path = osp.join(dir_path, f"labeled_video_{timestamp}.mp4")

                # Read first image to get dimensions
                first_img = cv2.imread(self.image_list[0])
                if first_img is not None:
                    height, width = first_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_output_path, fourcc, 30.0, (width, height))
                    logger.info(f"Creating labeled video: {video_output_path}")

                    for image_file in self.image_list:
                        frame_img = cv2.imread(image_file)
                        if frame_img is None:
                            continue

                        # Read corresponding JSON file
                        label_file = osp.splitext(image_file)[0] + ".json"
                        if self.output_dir:
                            label_file = osp.join(self.output_dir, osp.basename(label_file))

                        if osp.exists(label_file):
                            try:
                                with io_open(label_file, "r") as f:
                                    data = json.load(f)

                                shapes = data.get("shapes", [])

                                # Batched points
                                poly_points_list = []
                                rect_points_list = []

                                for shape_data in shapes:
                                    shape_type = shape_data.get("shape_type", "polygon")
                                    points = shape_data.get("points", [])

                                    if len(points) < 3:
                                        continue

                                    pts = np.array([(int(p[0]), int(p[1])) for p in points], np.int32)
                                    pts = pts.reshape((-1, 1, 2))

                                    if shape_type == "rectangle":
                                        rect_points_list.append(pts)
                                    elif shape_type == "polygon":
                                        poly_points_list.append(pts)

                                # Batch draw rectangles
                                if rect_points_list:
                                    rect_overlay = frame_img.copy()
                                    cv2.fillPoly(rect_overlay, rect_points_list, (180, 180, 255))  # Light red (BGR)
                                    alpha_bbox = 0.2  # High transparency
                                    cv2.addWeighted(rect_overlay, alpha_bbox, frame_img, 1 - alpha_bbox, 0, frame_img)
                                    # Draw red outline
                                    cv2.polylines(frame_img, rect_points_list, True, (0, 0, 255), 2)

                                # Batch draw polygons
                                if poly_points_list:
                                    poly_overlay = frame_img.copy()
                                    cv2.fillPoly(poly_overlay, poly_points_list, (0, 128, 0))  # Dark green (BGR)
                                    alpha_mask = 0.7  # Low transparency
                                    cv2.addWeighted(poly_overlay, alpha_mask, frame_img, 1 - alpha_mask, 0, frame_img)
                            except Exception as e:
                                logger.warning(f"Error reading label file {label_file}: {e}")

                        video_writer.write(frame_img)

                    video_writer.release()
                    logger.info(f"Labeled video saved to: {video_output_path}")
        except Exception as e:
            logger.error(f"Failed to create labeled video: {e}")


def load_existing_shapes(image_file):
    """
    Loads existing shapes from the JSON file for skip detection.

    Args:
        image_file (str): The path to the image file.

    Returns:
        list: A list of Shape objects loaded from the JSON file, or None if
              the file does not exist or contains no shapes.
    """
    label_file = osp.splitext(image_file)[0] + ".json"
    if not osp.exists(label_file):
        return None

    try:
        with io_open(label_file, "r") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        if not shapes:
            return None

        existing_shapes = []
        for shape_data in shapes:
            shape = Shape()
            shape.load_from_dict(shape_data, close=False)
            if shape.shape_type in ["rectangle", "rotation", "polygon"]:
                shape.selected = True
                existing_shapes.append(shape)

        return existing_shapes if existing_shapes else None

    except Exception as e:
        logger.warning(f"Failed to load existing shapes: {e}")
        return None


def finish_processing(self, progress_dialog):
    target_index = self.current_index
    target_file = self.image_list[self.current_index]
    # --- Video Export with Burned-in Masks (Background) ---
    self.video_export_thread = VideoExportThread(self.image_list, self.output_dir)
    self.video_export_thread.start()
    # --- End Video Export ---
    
    self.import_image_folder(osp.dirname(target_file), load=False)
    self.file_list_widget.setCurrentRow(target_index)

    del self.text_prompt
    del self.run_tracker
    del self.image_index
    del self.current_index

    progress_dialog.close()

    popup = Popup(
        self.tr("Processing completed successfully!"),
        self,
        icon=new_icon_path("copy-green", "svg"),
    )
    popup.show_popup(self, position="center")


def cancel_operation(self):
    self.cancel_processing = True


def save_auto_labeling_result(self, image_file, auto_labeling_result):
    try:
        label_file = osp.splitext(image_file)[0] + ".json"
        if self.output_dir:
            label_file = osp.join(self.output_dir, osp.basename(label_file))

        if auto_labeling_result is None:
            new_shapes = []
            new_description = ""
            replace = True
        else:
            new_shapes = [
                shape.to_dict() for shape in auto_labeling_result.shapes
            ]
            new_description = auto_labeling_result.description
            replace = auto_labeling_result.replace

        if osp.exists(label_file):
            with io_open(label_file, "r") as f:
                data = json.load(f)

            if replace:
                data["shapes"] = new_shapes
                data["description"] = new_description
            else:
                data["shapes"].extend(new_shapes)
                if "description" in data:
                    data["description"] += new_description
                else:
                    data["description"] = new_description
        else:
            if self._config["store_data"]:
                with open(image_file, "rb") as f:
                    image_data = f.read()
                image_data = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data = None

            image_path = osp.basename(image_file)
            image_width, image_height = get_image_size(image_file)

            data = {
                "version": __version__,
                "flags": {},
                "shapes": new_shapes,
                "imagePath": image_path,
                "imageData": image_data,
                "imageHeight": image_height,
                "imageWidth": image_width,
                "description": new_description,
            }

        with io_open(label_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(
            f"Failed to save auto labeling result for image file '{image_file}': {str(e)}"
        )


def process_next_image(self, progress_dialog, batch=True):
    """Process images in batch mode.

    Args:
        progress_dialog: Progress dialog widget for displaying progress.
        batch: If True, results are saved directly without updating canvas.
               If False, results trigger UI updates and canvas refresh.
               Defaults to True for batch processing mode.
    """
    model_type = self.auto_labeling_widget.model_manager.loaded_model_config[
        "type"
    ]
    model = self.auto_labeling_widget.model_manager.loaded_model_config[
        "model"
    ]
    total_images = len(self.image_list)
    self._progress_dialog = progress_dialog

    try:
        while (self.image_index < total_images) and (
            not self.cancel_processing
        ):
            image_file = self.image_list[self.image_index]
            current_progress = self.image_index + 1
            progress_dialog.setValue(current_progress)
            progress_dialog.setLabelText(
                f"Progress: {current_progress}/{total_images}"
            )
            QApplication.processEvents()

            batch_processing_mode = "default"
            if model_type == "remote_server":
                batch_processing_mode = model.get_batch_processing_mode()
                if batch_processing_mode == "video":
                    model._widget = self
                    self.filename = image_file
                    self.load_file(self.filename)
                    batch = False
            elif model_type in _BATCH_PROCESSING_VIDEO_MODELS:
                self.filename = image_file
                self.load_file(self.filename)
                batch = False

            if self.text_prompt:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        text_prompt=self.text_prompt,
                        batch=batch,
                    )
                )
            elif self.run_tracker:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        run_tracker=self.run_tracker,
                        batch=batch,
                    )
                )
                if batch_processing_mode == "video":
                    logger.info("Video propagation completed, breaking loop")
                    self.image_index = total_images
                    break
            else:
                existing_shapes = None
                if (
                    model_type in _SKIP_DET_MODELS
                    and self.auto_labeling_widget.button_skip_detection.isChecked()
                ):
                    existing_shapes = load_existing_shapes(image_file)

                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        batch=batch,
                        existing_shapes=existing_shapes,
                    )
                )

            if batch:
                save_auto_labeling_result(
                    self, image_file, auto_labeling_result
                )

            self.image_index += 1

        finish_processing(self, progress_dialog)

    except Exception as e:
        progress_dialog.close()

        logger.error(f"Error occurred while processing images: {e}")
        popup = Popup(
            self.tr("Error occurred while processing images!"),
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def show_progress_dialog_and_process(self):
    self.cancel_processing = False

    progress_dialog = QProgressDialog(
        self.tr("Processing..."),
        self.tr("Cancel"),
        0,
        len(self.image_list),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Batch Processing"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)

    initial_progress = (
        self.image_index + 1
        if self.image_index < len(self.image_list)
        else len(self.image_list)
    )
    progress_dialog.setValue(initial_progress)
    progress_dialog.setLabelText(
        f"Progress: {initial_progress}/{len(self.image_list)}"
    )
    progress_bar = progress_dialog.findChild(QtWidgets.QProgressBar)

    if progress_bar:
        model_type = (
            self.auto_labeling_widget.model_manager.loaded_model_config.get(
                "type", ""
            )
        )
        batch_processing_mode = "default"
        if model_type == "remote_server":
            model = self.auto_labeling_widget.model_manager.loaded_model_config.get(
                "model"
            )
            batch_processing_mode = model.get_batch_processing_mode()

        def update_progress(value):
            if batch_processing_mode != "video" or not self.run_tracker:
                progress_dialog.setLabelText(f"{value}/{len(self.image_list)}")

        progress_bar.valueChanged.connect(update_progress)

    progress_dialog.setStyleSheet(
        """
        QProgressDialog {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            min-width: 280px;
            min-height: 120px;
            padding: 20px;
            backdrop-filter: blur(20px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08),
                        0 2px 6px rgba(0, 0, 0, 0.04);
        }
        QProgressBar {
            border: none;
            border-radius: 4px;
            background-color: rgba(0, 0, 0, 0.05);
            text-align: center;
            color: #1d1d1f;
            font-size: 13px;
            min-height: 20px;
            max-height: 20px;
            margin: 16px 0;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #0066FF,
                stop:0.5 #00A6FF,
                stop:1 #0066FF);
            border-radius: 3px;
        }
        QLabel {
            color: #1d1d1f;
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 8px;
        }
        QPushButton {
            background-color: rgba(255, 255, 255, 0.8);
            border: 0.5px solid rgba(0, 0, 0, 0.1);
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
            color: #0066FF;
            min-width: 82px;
            height: 36px;
            padding: 0 16px;
            margin-top: 16px;
        }
        QPushButton:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        QPushButton:pressed {
            background-color: rgba(0, 0, 0, 0.08);
        }
    """
    )
    progress_dialog.canceled.connect(lambda: cancel_operation(self))
    progress_dialog.show()

    QTimer.singleShot(200, lambda: process_next_image(self, progress_dialog))


class BatchModeSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = "tracking"
        self.text_prompt = ""
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Select Batch Processing Mode"))
        self.setFixedSize(450, 300)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Mode Selection Group
        self.mode_group = QButtonGroup(self)
        
        self.track_radio = QRadioButton(self.tr("Video Tracking (Propagation)"))
        self.track_radio.setChecked(True)
        self.track_radio.setToolTip(self.tr("Use existing mask to track objects across frames using video propagation."))
        self.mode_group.addButton(self.track_radio)
        layout.addWidget(self.track_radio)

        track_desc = QLabel(self.tr("Propagate the current object mask to all frames."))
        track_desc.setStyleSheet("color: #666; font-size: 11px; margin-left: 20px; margin-bottom: 10px;")
        layout.addWidget(track_desc)

        self.text_radio = QRadioButton(self.tr("Text Prompt (All Frames)"))
        self.text_radio.setToolTip(self.tr("Apply text prompt to each frame independently."))
        self.mode_group.addButton(self.text_radio)
        layout.addWidget(self.text_radio)

        text_desc = QLabel(self.tr("Run text prompt inference on every frame."))
        text_desc.setStyleSheet("color: #666; font-size: 11px; margin-left: 20px;")
        layout.addWidget(text_desc)

        # Text Input Area
        self.input_container = QtWidgets.QWidget()
        input_layout = QVBoxLayout(self.input_container)
        input_layout.setContentsMargins(20, 0, 0, 0)
        
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText(self.tr("Enter text prompt (e.g., 'black cat')..."))
        self.text_input.setEnabled(False)
        input_layout.addWidget(self.text_input)
        
        layout.addWidget(self.input_container)

        # Connect signals
        self.track_radio.toggled.connect(self.on_mode_changed)
        self.text_radio.toggled.connect(self.on_mode_changed)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.setLayout(layout)
        self.setStyleSheet(
            """
            QRadioButton { font-size: 13px; font-weight: 500; color: #1d1d1f; }
            QLineEdit {
                border: 1px solid #E5E5E5;
                border-radius: 8px;
                background-color: #F9F9F9;
                font-size: 13px;
                height: 36px;
                padding: 0 12px;
                color: #1d1d1f;
            }
            QLineEdit:disabled { background-color: #F0F0F0; color: #999; }
            QLineEdit:focus { border: 2px solid #0066FF; }
            QLabel { color: #1d1d1f; }
        """
        )

    def on_mode_changed(self):
        is_text_mode = self.text_radio.isChecked()
        self.text_input.setEnabled(is_text_mode)
        if is_text_mode:
            self.text_input.setFocus()
            self.mode = "text"
        else:
            self.mode = "tracking"

    def validate_and_accept(self):
        if self.text_radio.isChecked():
            text = self.text_input.text().strip()
            if not text:
                QtWidgets.QMessageBox.warning(
                    self, 
                    self.tr("Missing Input"), 
                    self.tr("Please enter a text prompt.")
                )
                return
            self.text_prompt = text
        self.accept()

    def get_selection(self):
        if self.exec_() == QDialog.Accepted:
            return self.mode, self.text_prompt
        return None, None


def run_all_images(self):
    if len(self.image_list) < 1:
        return

    if self.auto_labeling_widget.model_manager.loaded_model_config is None:
        self.auto_labeling_widget.model_manager.new_model_status.emit(
            self.tr("Model is not loaded. Choose a mode to continue.")
        )
        return

    if (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in _BATCH_PROCESSING_INVALID_MODELS
    ):
        logger.warning(
            f"The model `{self.auto_labeling_widget.model_manager.loaded_model_config['type']}`"
            f" is not supported for this action."
            f" Please choose a valid model to execute."
        )
        self.auto_labeling_widget.model_manager.new_model_status.emit(
            self.tr(
                "Invalid model type, please choose a valid model_type to run."
            )
        )
        return

    # Skip confirmation for Video mode because we have a dedicated dialog
    model_type = self.auto_labeling_widget.model_manager.loaded_model_config["type"]
    batch_processing_mode = "default"
    if model_type == "remote_server":
        model = self.auto_labeling_widget.model_manager.loaded_model_config["model"]
        if hasattr(model, "get_batch_processing_mode"):
            batch_processing_mode = model.get_batch_processing_mode()
        
        if batch_processing_mode is None:
            self.auto_labeling_widget.model_manager.new_model_status.emit(
                self.tr(
                    "Batch processing is not supported for the current task."
                )
            )
            return

    if batch_processing_mode != "video":
        response = QtWidgets.QMessageBox()
        response.setIcon(QtWidgets.QMessageBox.Warning)
        response.setWindowTitle(self.tr("Confirmation"))
        response.setText(self.tr("Do you want to process all images?"))
        response.setStandardButtons(
            QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
        )
        response.setStyleSheet(get_msg_box_style())
        if response.exec_() != QtWidgets.QMessageBox.Ok:
            return

    logger.info("Start running all images...")

    self.current_index = self.fn_to_index[str(self.filename)]
    self.image_index = self.current_index
    self.text_prompt = ""
    self.run_tracker = False

    if model_type == "remote_server":
        if batch_processing_mode == "video":
            dialog = BatchModeSelectionDialog(parent=self)
            mode, text = dialog.get_selection()
            
            if mode == "tracking":
                self.run_tracker = True
                show_progress_dialog_and_process(self)
            elif mode == "text":
                self.text_prompt = text
                # run_tracker must be False to allow looping in process_next_image
                self.run_tracker = False 
                show_progress_dialog_and_process(self)
        
        elif batch_processing_mode == "text_prompt":
            text_input_dialog = TextInputDialog(parent=self)
            self.text_prompt = text_input_dialog.get_input_text()
            if self.text_prompt:
                show_progress_dialog_and_process(self)
        else:
            show_progress_dialog_and_process(self)

    elif model_type in _BATCH_PROCESSING_TEXT_PROMPT_MODELS:
        text_input_dialog = TextInputDialog(parent=self)
        self.text_prompt = text_input_dialog.get_input_text()
        if self.text_prompt or model_type == "yoloe":
            show_progress_dialog_and_process(self)
            
    elif (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        == "florence2"
    ):
        self.text_prompt = self.auto_labeling_widget.edit_text.text()
        show_progress_dialog_and_process(self)
        
    elif (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in _BATCH_PROCESSING_VIDEO_MODELS
    ):
        # Fallback for local video models if any
        self.run_tracker = True
        show_progress_dialog_and_process(self)
    else:
        show_progress_dialog_and_process(self)
