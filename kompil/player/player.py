import os
import sys
import time
import torch
from typing import List, Dict, Optional
from datetime import datetime
from PyQt5.QtCore import QObject, Qt, pyqtSignal, QCoreApplication, QTimer, QPoint
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (
    QSizePolicy,
    QWidget,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QFileDialog,
    QAction,
    QToolBar,
    QSpinBox,
    QPushButton,
    QInputDialog,
)

from kompil.player.decoders.decoder import create_decoder, Decoder
from kompil.player.filters.filter import create_filter, Filter
from kompil.player.options import PlayerTargetOptions
from kompil.utils.ui.image import torch_rgb_tensor_to_qimage, torch_mono_tensor_to_qimage
from kompil.utils.ui.image_viewer import QImageViewer
from kompil.utils.ui.app import QKompilApp
from kompil.utils.ui.icons import *
from kompil.utils.paths import PATH_BUILD


class FrameBuilder:
    def __init__(
        self,
        files: List[str],
        decoders: Dict[int, str],
        filters: Dict[int, List[str]],
        opt: PlayerTargetOptions,
    ):
        self.__decoders: List[Decoder] = []
        self.__filters: List[Filter] = []
        self.__last_slider_position = 0.5
        for i, fpath in enumerate(files):
            # Create decoder
            dec_name = decoders.get(i, "auto")
            dec = create_decoder(dec_name, fpath, opt)
            self.__decoders.append(dec)
            # Create filter
            fil_data = filters.get(i, [])
            fil = create_filter(fil_data, opt)
            self.__filters.append(fil)

        self.__frames_count = min([dec.get_total_frames() for dec in self.__decoders])

        # Framerate
        framerates = [
            dec.get_framerate() for dec in self.__decoders if dec.get_framerate() is not None
        ]
        if not framerates:
            print("WARNING: no framerate defined, taking 30 as default")
            self.__framerate = 30.0
        else:
            self.__framerate = framerates[0]
            assert all([framerate == self.__framerate for framerate in framerates])

        # Slider
        self.__is_slider = any(fil.is_slider for fil in self.__filters)
        self.set_slider(0.5)

    @property
    def is_slider(self) -> bool:
        return self.__is_slider

    @property
    def slider_position(self) -> float:
        return self.__last_slider_position

    def set_slider(self, position: float):
        for fil in self.__filters:
            fil.set_slider(position)
        self.__last_slider_position = position

    def get_total_frames(self):
        return self.__frames_count

    def get_framerate(self):
        return self.__framerate

    def get_cur_frame(self) -> torch.Tensor:
        frames = []
        for i in range(len(self.__decoders)):
            dec = self.__decoders[i]
            fil = self.__filters[i]
            frames.append(fil.filter(dec.get_cur_frame(), dec.get_colorspace(), dec.get_position()))
        # Stack horizontally
        return torch.cat(frames, dim=-1)

    def set_position(self, pos):
        for decoder in self.__decoders:
            decoder.set_position(pos)


class ControlBar(QWidget):
    frame_changed = pyqtSignal(int)
    filter_changed = pyqtSignal(float)
    toggle_play = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.__max_frame = 0
        # Widgets
        w_play_pause = self.__setup_play_pause_button()
        self.sld_frame = QSlider(Qt.Orientation.Horizontal)
        self.sld_filter = QSlider(Qt.Orientation.Horizontal)
        self.sld_filter.setRange(0, 1000)
        self.sld_filter.setVisible(False)
        self.btn_jump = QPushButton(get_icon_go_jump(), "")
        self.btn_jump.setFlat(True)
        self.btn_jump.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sld_frame.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Layout
        layout = QGridLayout()
        layout.addWidget(w_play_pause, 0, 0)
        layout.addWidget(self.sld_frame, 0, 1)
        layout.addWidget(self.btn_jump, 0, 2)
        layout.addWidget(self.sld_filter, 1, 1)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        # Connections
        self.sld_frame.valueChanged.connect(self.frame_changed.emit)
        self.sld_filter.valueChanged.connect(self.__slider_filter_changed)
        self.btn_play.clicked.connect(lambda: self.toggle_play.emit(True))
        self.btn_pause.clicked.connect(lambda: self.toggle_play.emit(False))
        self.btn_jump.clicked.connect(self.__jump_to)

    def set_frames(self, frames: int):
        self.__max_frame = frames - 1
        self.sld_frame.setRange(0, self.__max_frame)

    def update_frame(self, value: int):
        if self.sld_frame.value() != value:
            self.sld_frame.setValue(value)

    def update_status(self, value: bool):
        if value:
            self.btn_play.hide()
            self.btn_pause.show()
        else:
            self.btn_play.show()
            self.btn_pause.hide()

    def __setup_play_pause_button(self) -> QWidget:
        # Widgets
        self.btn_play = QPushButton(get_icon_play(), "")
        self.btn_play.setFlat(True)
        self.btn_play.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_pause = QPushButton(get_icon_pause(), "")
        self.btn_pause.setFlat(True)
        self.btn_pause.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_pause)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def __slider_filter_changed(self, value: int):
        self.filter_changed.emit(value / 1000)

    def __jump_to(self):
        value, _ = QInputDialog.getInt(
            self,
            "Jump to frame",
            "frame",
            value=self.sld_frame.value(),
            min=0,
            max=self.__max_frame,
        )
        if value is not None:
            self.toggle_play.emit(False)
            self.sld_frame.setValue(value)
            # self.frame_changed.emit(value)


class Controller(QObject):
    playing_status = pyqtSignal(bool)
    image_changed = pyqtSignal(QImage)
    frame_changed = pyqtSignal(int)

    def __init__(self, decoder: FrameBuilder, framerate: Optional[int], parent=None):
        super().__init__(parent=parent)
        self.__current_frame: int = 0
        self.__last_image = None
        self.__playing: bool = False
        self.__framerate = framerate if framerate else decoder.get_framerate()
        # Open decoder
        self.__decoder = decoder
        # Objects
        self.__play_timer = QTimer(self)

    @property
    def current_image(self) -> QImage:
        return self.__last_image

    @property
    def current_frame(self) -> int:
        return self.__current_frame

    @property
    def framerate(self) -> int:
        return self.__framerate

    @property
    def frames(self) -> int:
        return self.__decoder.get_total_frames()

    @property
    def playing(self) -> bool:
        return self.__playing

    def set_framerate(self, value: int):
        self.__framerate = value

    def play(self):
        if self.playing:
            return
        if self.__current_frame == self.frames - 1:
            self.__current_frame = 0
        self.__playing = True
        self.__play_timer.singleShot(0, self.__play_timeout)
        self.playing_status.emit(True)

    def pause(self):
        if not self.playing:
            return
        self.__play_timer.stop()
        self.__playing = False
        self.playing_status.emit(False)

    def next_frame(self):
        self.pause()
        self.update_frame(min(self.__current_frame + 1, self.__decoder.get_total_frames() - 1))

    def previous_frame(self):
        self.pause()
        self.update_frame(max(self.__current_frame - 1, 0))

    def update_frame(self, frame: Optional[int] = None, force: bool = False):
        frame = self.__current_frame if frame is None else frame
        # Avoid uneccessary calls
        if not force and frame == self.__current_frame:
            return
        self.__current_frame = frame
        # Decode the frame and print it
        self.__decoder.set_position(frame)
        im = self.__decoder.get_cur_frame()
        if len(im.shape) != 3:
            raise NotImplementedError()
        if im.shape[-3] == 1:
            qim: QImage = torch_mono_tensor_to_qimage(im)
        if im.shape[-3] == 3:
            qim: QImage = torch_rgb_tensor_to_qimage(im)
        self.__last_image = qim
        self.frame_changed.emit(frame)
        self.image_changed.emit(qim)

    def __play_timeout(self):
        if not self.playing:
            return
        # Look for next frame
        next_frame = self.__current_frame + 1
        if next_frame >= self.__decoder.get_total_frames():
            self.pause()
            return
        # Update frame
        before_time = time.time()
        self.update_frame(next_frame)
        after_time = time.time()
        # Calculate delay
        spent_time = int(1000.0 * (after_time - before_time))
        next_delay = max(0, 1000 // self.__framerate - spent_time)
        self.__play_timer.singleShot(int(next_delay), self.__play_timeout)

    def play_pause(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def set_play_status(self, status: bool):
        if status:
            self.play()
        else:
            self.pause()

    def add_controlbar(self, bar: ControlBar):
        bar.set_frames(self.frames)
        bar.update_status(self.playing)
        bar.update_frame(self.__current_frame)
        bar.frame_changed.connect(self.__slider_frame_changed)
        self.frame_changed.connect(bar.update_frame)
        self.playing_status.connect(bar.update_status)
        bar.sld_filter.setVisible(self.__decoder.is_slider)
        bar.sld_filter.setValue(int(1000 * self.__decoder.slider_position))
        bar.filter_changed.connect(self.__filter_changed)
        bar.toggle_play.connect(self.set_play_status)

    def add_viewer(self, viewer: QImageViewer):
        self.image_changed.connect(viewer.set_image)

    def __filter_changed(self, value: float):
        self.__decoder.set_slider(value)
        self.update_frame(force=True)

    def __slider_frame_changed(self, value: int):
        self.update_frame(value)


class Window(QMainWindow):
    def __init__(self, name: str, decoder: FrameBuilder, framerate: Optional[int]) -> None:
        super().__init__()
        self.__fullscreen = False
        self.__controller = Controller(decoder, framerate)
        # Window setup
        self.setWindowTitle(name)
        self.setWindowIcon(get_icon_kompil())
        # Widget setup
        self.setCentralWidget(self.__setup_central_widget())
        # Actions
        self.action_save_image = QAction(get_icon_shot(), "Snapshot", self)
        self.action_zoom_original = QAction(get_icon_zoom_original(), "Original size", self)
        self.action_zoom_fit = QAction(get_icon_fit(), "Fit to screen", self)
        self.action_zoom_in = QAction(get_icon_zoom_in(), "Zoom in", self)
        self.action_zoom_out = QAction(get_icon_zoom_out(), "Zoom out", self)
        self.action_fullscreen = QAction(get_icon_fullscreen(), "Fullscreen", self)
        # Action actions
        self.action_save_image.triggered.connect(self.save_image)
        self.action_zoom_original.triggered.connect(self.widget_viewer.zoom_original)
        self.action_zoom_fit.triggered.connect(self.widget_viewer.fit)
        self.action_zoom_in.triggered.connect(self.widget_viewer.zoom_in)
        self.action_zoom_out.triggered.connect(self.widget_viewer.zoom_out)
        self.action_fullscreen.triggered.connect(self.toggle_fullscreen)
        # Toolbar widgets
        framerate_spinbox = QSpinBox()
        framerate_spinbox.setValue(int(self.__controller.framerate))
        framerate_spinbox.valueChanged.connect(self.__controller.set_framerate)
        # Statusbar widgets
        self.widget_display_frame = QLabel()
        self.widget_display_pixel = QLabel()
        self.__controller.frame_changed.connect(self.__frame_changed)
        self.widget_viewer.mouse_hover_pixel.connect(self.__pixel_changed)
        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.addAction(self.action_save_image)
        toolbar.addAction(self.action_zoom_original)
        toolbar.addAction(self.action_zoom_fit)
        toolbar.addAction(self.action_zoom_in)
        toolbar.addAction(self.action_zoom_out)
        toolbar.addAction(self.action_fullscreen)
        toolbar.addWidget(self.__get_spacing())
        toolbar.addWidget(QLabel("Framerate:"))
        toolbar.addWidget(framerate_spinbox)
        self.addToolBar(toolbar)
        # Statusbar
        self.statusBar().addWidget(self.widget_display_frame)
        self.statusBar().addWidget(self.__get_spacing(), stretch=1)
        self.statusBar().addWidget(self.widget_display_pixel)
        # Initialize
        self.__controller.add_controlbar(self.controlbar)
        self.__controller.add_viewer(self.widget_viewer)
        self.__controller.update_frame(0, force=True)
        self.widget_viewer.setBaseSize(self.__controller.current_image.size())

    @staticmethod
    def __get_spacing() -> QWidget:
        spacing = QWidget()
        spacing.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return spacing

    def __frame_changed(self, frame: int):
        self.widget_display_frame.setText(f"frame: {frame} / {self.__controller.frames - 1}")

    def __pixel_changed(self, pixel: QPoint):
        c = self.widget_viewer.image.pixelColor(pixel)
        self.widget_display_pixel.setText(
            f"[{pixel.x()},{pixel.y()}] -> (R: {c.red()}, G: {c.green()}, B: {c.blue()})"
        )

    def __setup_central_widget(self) -> QWidget:
        # Widgets
        self.controlbar = ControlBar()
        self.widget_viewer = QImageViewer()
        self.widget_viewer.setMouseTracking(True)
        self.setMouseTracking(True)
        # Connections
        self.widget_viewer.mouse_left_doubleclick_pixel.connect(lambda _: self.toggle_fullscreen())
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.widget_viewer)
        layout.addWidget(self.controlbar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def save_image(self):
        image = self.__controller.current_image
        if image is None:
            return
        base_name = "playshot_" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
        base_path = os.path.join(PATH_BUILD, base_name)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save screen shot", directory=base_path, filter="Images (*.png)"
        )
        if path is None:
            return
        image.save(path, format="png")

    def toggle_fullscreen(self):
        if self.__fullscreen:
            self.showNormal()
            self.__fullscreen = False
        else:
            self.showFullScreen()
            self.__fullscreen = True

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.__controller.play_pause()
        elif event.key() == Qt.Key.Key_Left:
            self.__controller.previous_frame()
        elif event.key() == Qt.Key.Key_Right:
            self.__controller.next_frame()
        elif event.key() == Qt.Key.Key_S:
            self.action_save_image.trigger()
        elif event.key() in [Qt.Key.Key_Q, Qt.Key.Key_Escape]:
            QCoreApplication.exit(0)
        event.accept()


class Player(object):
    def __init__(
        self,
        name: str,
        files: List[str],
        opt: PlayerTargetOptions,
        framerate: Optional[float],
        decoders: Dict[int, str],
        filters: Dict[int, List[str]],
    ):
        self.__name = name
        self.__framerate = framerate
        self.__decoder = FrameBuilder(files, decoders, filters, opt)

    def run(self):
        app = QKompilApp([])
        win = Window(self.__name, self.__decoder, self.__framerate)
        win.show()
        sys.exit(app.exec_())

    def pipe(self):
        raise NotImplementedError()

        # TODO: restore that
        # for idx in range(self._decoder.get_total_frames()):
        #     self._decoder.set_position(idx)
        #     frame = self._decoder.get_cur_frame()
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     try:
        #         sys.stdout.buffer.write(frame.tostring())
        #     except BrokenPipeError:
        #         break
