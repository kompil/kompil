import torch
import sys
from typing import Optional, Tuple, List
from torch.utils.hooks import RemovableHandle
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QModelIndex, QCoreApplication, QPoint, QSize
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (
    QFrame,
    QSizePolicy,
    QWidget,
    QSlider,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QSplitter,
    QGroupBox,
    QPushButton,
    QMessageBox,
)
import plotly.graph_objects as go

from kompil.nn.models.model import VideoNet, model_load
from kompil.utils.ui.image_viewer import QImageViewer
from kompil.utils.ui.image import torch_rgb_tensor_to_qimage, torch_mono_tensor_to_qimage
from kompil.utils.ui.layer_selecter import LayerSelecter, LayerSelectItem
from kompil.utils.ui.app import QKompilApp
from kompil.utils.ui.icons import *


def _to_3d_tensor(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 1:
        t = t.unsqueeze(1)
    if len(t.shape) == 2:
        t = t.unsqueeze(1)
    return t


class _PixelSelection:
    def __init__(self, module: torch.nn.Module, channel: int, y: int, x: int) -> None:
        self.module = module
        self.channel = channel
        self.y = y
        self.x = x

    @property
    def pixel(self) -> Tuple[int, int, int]:
        return (self.channel, self.y, self.x)

    def __repr__(self) -> str:
        return f"({self.channel}, {self.y}, {self.x}), {self.module}"


class Model(QObject):
    channel_count_changed = pyqtSignal(int)

    def __init__(self, model: VideoNet):
        super().__init__()
        self.model = model
        self.module_selected: Optional[torch.nn.Module] = None

    @property
    def frames(self) -> int:
        return self.model.nb_frames

    def _to_input(self, id_frame: int) -> torch.Tensor:
        return torch.tensor([[id_frame]], dtype=self.model.dtype, device=self.model.device)

    def __data_from_selection(self, id_frame) -> torch.Tensor:
        data = None

        def _hook_fct(module, inputs, outputs):
            nonlocal data
            data = outputs

        handle = self.module_selected.register_forward_hook(_hook_fct)
        with torch.no_grad():
            self.model.forward(self._to_input(id_frame))
        handle.remove()

        return data

    def read_frame_channel(self, id_frame: int, channel: int) -> QImage:
        # If no layer has been selected, return the RGB final
        if self.module_selected is None:
            with torch.no_grad():
                frame = self.model.forward_rgb8(self._to_input(id_frame))[0]
            return torch_rgb_tensor_to_qimage(frame)
        # Get the data
        data = self.__data_from_selection(id_frame)
        image = data[0, channel : channel + 1]
        # Transform to QImage
        return torch_mono_tensor_to_qimage(_to_3d_tensor(image))

    def __add_pixel_hook(self, pixel: _PixelSelection) -> Tuple[List[float], RemovableHandle]:
        series = []

        def _hook_fct(module, inputs, outputs: torch.Tensor):
            nonlocal pixel, series
            outputs = _to_3d_tensor(outputs[0])
            for p in pixel.pixel:
                outputs = outputs[p]

            if outputs.is_quantized:
                outputs = outputs.dequantize()
            series.append(outputs.item())

        handle = pixel.module.register_forward_hook(_hook_fct)

        return series, handle

    def read_pixels(self, pixels: List[_PixelSelection]) -> List[List[float]]:
        data = []
        handles = []

        for selpix in pixels:

            series, handle = self.__add_pixel_hook(selpix)

            data.append(series)
            handles.append(handle)

        def _qt_response(**kwargs):
            QCoreApplication.processEvents()

        with torch.no_grad():
            self.model.run_once(callback=_qt_response)

        for handle in handles:
            handle.remove()

        return data

    def module_from_treeview(self, index: QModelIndex):
        ptr: LayerSelectItem = index.internalPointer()
        self.module_selected = ptr.module

        # Emit new channel count
        data = self.__data_from_selection(0)
        channel_count = data.shape[1]
        self.channel_count_changed.emit(channel_count)


def _create_icon_button(icon: QIcon, tip: str) -> QPushButton:
    button = QPushButton(icon, "")
    button.setFlat(True)
    button.setToolTip(tip)
    button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
    return button


class Controlbar(QWidget):
    updated = pyqtSignal()
    zoom_in = pyqtSignal()
    zoom_out = pyqtSignal()
    zoom_fit = pyqtSignal()
    zoom_original = pyqtSignal()

    def __init__(self, frames: int, parent=None) -> None:
        super().__init__(parent=parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.__create_btns())
        self.layout().addWidget(self.__create_sliders(frames))

    def __create_btns(self) -> QWidget:
        btn_zoom_in = self.__create_icon_button(get_icon_zoom_in(), "Zoom in", self.zoom_in)
        btn_zoom_out = self.__create_icon_button(get_icon_zoom_out(), "Zoom out", self.zoom_out)
        btn_fit = self.__create_icon_button(get_icon_fit(), "Fit", self.zoom_fit)
        btn_zoom_original = self.__create_icon_button(
            get_icon_zoom_original(), "Original size", self.zoom_original
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.setSpacing(0)
        layout.addWidget(btn_zoom_in)
        layout.addWidget(btn_zoom_out)
        layout.addWidget(btn_fit)
        layout.addWidget(btn_zoom_original)

        btns = QWidget()
        btns.setLayout(layout)
        return btns

    def __create_icon_button(self, icon: QIcon, tip: str, signal) -> QPushButton:
        button = _create_icon_button(icon, tip)
        button.clicked.connect(signal.emit)
        return button

    def __create_sliders(self, frames: int) -> QWidget:
        self.__current_frame = 0
        self.__current_channel = 0

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, frames - 1)
        self.frame_display = QLabel()
        self.frame_display.setText(str(self.__current_frame))

        self.frame_slider.valueChanged.connect(self.__frame_changed)

        self.channel_slider = QSlider(Qt.Orientation.Horizontal)
        self.channel_slider.setRange(0, 1)
        self.channel_display = QLabel()
        self.channel_slider.setDisabled(True)

        self.channel_slider.valueChanged.connect(self.__channel_changed)

        layout = QGridLayout()
        layout.addWidget(QLabel("Frame:"), 0, 0)
        layout.addWidget(self.frame_slider, 0, 1)
        layout.addWidget(self.frame_display, 0, 2)
        layout.addWidget(QLabel("Channel:"), 1, 0)
        layout.addWidget(self.channel_slider, 1, 1)
        layout.addWidget(self.channel_display, 1, 2)
        layout.setContentsMargins(0, 0, 0, 0)

        sliders = QWidget()
        sliders.setLayout(layout)
        return sliders

    @property
    def current_frame(self) -> int:
        return self.__current_frame

    @property
    def current_channel(self) -> int:
        return self.__current_channel

    def __frame_changed(self, value: int):
        self.__current_frame = value
        self.frame_display.setText(str(self.__current_frame))
        self.updated.emit()

    def __channel_changed(self, value: int):
        self.__current_channel = value
        self.channel_display.setText(str(self.__current_channel))
        self.updated.emit()

    def channel_count_changed(self, value: int):
        self.channel_slider.setRange(0, value - 1)
        self.channel_slider.setDisabled(False)
        if self.__current_channel >= value:
            self.__channel_changed(value - 1)
        else:
            self.__channel_changed(self.__current_channel)


class PixelSelectionItemWidget(QFrame):
    signal_remove = pyqtSignal()

    def __init__(self, pix: _PixelSelection) -> None:
        super().__init__()
        self.__pix = pix

        btn = _create_icon_button(get_icon_list_remove(), "Remove this pixel from list.")
        btn.clicked.connect(self.signal_remove)

        layout = QHBoxLayout()
        layout.addWidget(QLabel(str(pix)))
        layout.addWidget(btn)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.setFrameStyle(QFrame.Shadow.Plain | QFrame.Shape.Panel)

    @property
    def pix(self):
        return self.__pix


class PixelSelectionListWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.setLayout(layout)

    def addPixel(self, pix: _PixelSelection):
        item = PixelSelectionItemWidget(pix)

        def rm_item():
            nonlocal item
            self.layout().removeWidget(item)
            del item

        item.signal_remove.connect(rm_item)
        self.layout().addWidget(item)

    def count(self) -> int:
        return self.layout().count()

    def get_list_pixels(self) -> List[_PixelSelection]:
        pixels = []
        for i in range(self.layout().count()):
            pixels.append(self.layout().itemAt(i).widget().pix)
        return pixels


class Window(QWidget):
    def __init__(self, model_file: str, cpu: bool, parent=None) -> None:
        super().__init__(parent=parent)
        self.pixel_charts = []
        # Open model
        model: VideoNet = model_load(model_file)
        if cpu:
            model = model.cpu()
        else:
            model.cuda()
        # Widgets
        self.model = Model(model)
        self.layer_selection = LayerSelecter(model)
        central_panel = self.__create_central_panel(self.model)
        # Connect frame changed
        self.layer_selection.activated.connect(self.__update_selection)
        # Layout
        splitter = QSplitter()
        splitter.addWidget(self.layer_selection)
        splitter.addWidget(central_panel)

        hlayout = QHBoxLayout()
        hlayout.addWidget(splitter)
        hlayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(hlayout)
        # Initial
        self.__update()

    def __create_central_panel(self, model: Model) -> QWidget:
        layout = QVBoxLayout()
        layout.addWidget(self.__create_view_box(model))
        layout.addWidget(self.__create_pixel_box())
        layout.setContentsMargins(0, 0, 0, 0)
        panel = QWidget()
        panel.setLayout(layout)
        return panel

    def __create_pixel_box(self) -> QGroupBox:
        # Widgets
        self.list_pixel_selection = PixelSelectionListWidget()
        self.btn_run_pixels = QPushButton("Run")
        self.btn_run_pixels.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
        )
        self.btn_run_pixels.setEnabled(False)
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.list_pixel_selection)
        layout.addWidget(self.btn_run_pixels)
        box = QGroupBox("Pixel selection")
        box.setLayout(layout)
        # Size policy
        box.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.MinimumExpanding)
        # Connection
        self.btn_run_pixels.clicked.connect(self.__run_pixel)
        return box

    def __create_view_box(self, model: Model) -> QGroupBox:
        # Widgets
        self.im_viewer = QImageViewer()
        self.control_bar = Controlbar(model.frames)
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.im_viewer)
        layout.addWidget(self.control_bar)
        layout.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("View")
        box.setLayout(layout)
        # Connections
        self.im_viewer.mouse_left_doubleclick_pixel.connect(self.__pixel_clicked)
        self.model.channel_count_changed.connect(self.control_bar.channel_count_changed)
        self.control_bar.updated.connect(self.__update)
        self.control_bar.zoom_in.connect(self.im_viewer.zoom_in)
        self.control_bar.zoom_out.connect(self.im_viewer.zoom_out)
        self.control_bar.zoom_fit.connect(self.im_viewer.fit)
        self.control_bar.zoom_original.connect(self.im_viewer.zoom_original)
        # Size policy
        self.im_viewer.setMinimumSize(480, 320)
        box.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        return box

    def __pixel_clicked(self, point: QPoint):
        pix = _PixelSelection(
            self.model.module_selected, self.control_bar.current_channel, point.y(), point.x()
        )
        self.list_pixel_selection.addPixel(pix)
        self.btn_run_pixels.setEnabled(True)

    def __run_pixel(self):
        # Read values from model
        try:
            self.btn_run_pixels.setEnabled(False)
            pixlist = self.list_pixel_selection.get_list_pixels()
            values = self.model.read_pixels(self.list_pixel_selection.get_list_pixels())
        except Exception as e:
            QMessageBox.warning(self, "Error while processing pixel", str(e))
        finally:
            self.btn_run_pixels.setEnabled(True)
        # Print
        frames = list(range(len(values[0])))
        fig = go.Figure()
        fig.update_layout(hovermode="x")
        for i, series in enumerate(values):
            pix = pixlist[i]
            fig.add_trace(go.Scatter(x=frames, y=series, mode="lines", name=str(pix)))
        fig.show()

    def __update_selection(self, index):
        self.model.module_from_treeview(index)
        self.__update()
        self.im_viewer.fit()

    def __update(self):
        frame = self.control_bar.current_frame
        chan = self.control_bar.current_channel
        # Set image
        image = self.model.read_frame_channel(frame, chan)
        self.im_viewer.set_image(image)

    def keyReleaseEvent(self, event):
        if event.key() in [Qt.Key.Key_Q, Qt.Key.Key_Escape]:
            QCoreApplication.exit(0)
        event.accept()


def hook(model_file: str, cpu: bool):
    if not torch.cuda.is_available() and not cpu:
        print("WARNING: cuda not available, switching to cpu")
        cpu = True

    app = QKompilApp([])
    win = Window(model_file, cpu)
    win.show()
    sys.exit(app.exec_())
