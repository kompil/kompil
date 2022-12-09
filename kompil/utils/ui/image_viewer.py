import math
from typing import Union

from PyQt5.QtCore import Qt, QPoint, QPointF, pyqtSignal, QSize
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QVBoxLayout, QFrame
from PyQt5.QtGui import QImage, QPixmap, QWheelEvent, QMouseEvent, QTransform


def _floor_point(point: QPointF) -> QPoint:
    return QPoint(int(math.floor(point.x())), int(math.floor(point.y())))


class QImageViewer(QFrame):
    mouse_hover_pixel = pyqtSignal(QPoint)
    mouse_left_release_pixel = pyqtSignal(QPoint)
    mouse_right_release_pixel = pyqtSignal(QPoint)
    mouse_left_press_pixel = pyqtSignal(QPoint)
    mouse_right_press_pixel = pyqtSignal(QPoint)
    mouse_left_doubleclick_pixel = pyqtSignal(QPoint)
    mouse_right_doubleclick_pixel = pyqtSignal(QPoint)

    __ZOOM_FACTOR = 1.1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__pixmap = None
        self.__image = None
        self.__draglast: QPoint = None
        self.__zoom = 1.0
        # Scene
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setScene(self.scene)
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)
        # Display
        layout.setContentsMargins(0, 0, 0, 0)  # Necessary to avoid scroll bar at full size
        self.view.setFrameStyle(QFrame.Shape.NoFrame)
        # Events
        self.view.wheelEvent = self.__wheel_event
        self.view.mouseMoveEvent = self.__mouse_move_event
        self.view.mousePressEvent = self.__mouse_press_event
        self.view.mouseReleaseEvent = self.__mouse_release_event
        self.view.mouseDoubleClickEvent = self.__mouse_double_click_event

    def sizeHint(self) -> QSize:
        if self.__pixmap == None:
            return super().sizeHint()
        rect = self.__pixmap.rect()
        return QSize(rect.width(), rect.height())

    def set_image(self, image: Union[QImage, QPixmap]):
        assert isinstance(image, (QPixmap, QImage)), "Argument must be a QImage or QPixmap."
        # Ensure it is a pixmap
        if isinstance(image, QImage):
            self.__image = image
            self.__pixmap = QPixmap.fromImage(image)
        else:
            self.__pixmap = image
            self.__image = QImage(image)
        # Draw
        rect = self.__pixmap.rect()
        self.scene.clear()
        self.scene.setSceneRect(rect.x(), rect.y(), rect.width(), rect.height())
        self.scene.addPixmap(self.__pixmap).setAcceptHoverEvents(True)

    def get_image(self) -> QImage:
        return self.__image

    def compute_fit_zoom(self) -> float:
        """Compute the value where the current image fit the window size"""
        view_w, view_h = self.view.width(), self.view.height()
        img_w, img_h = self.__pixmap.width(), self.__pixmap.height()
        return min(view_w / img_w, view_h / img_h)

    def zoom(self, value: float, limit: bool = True, mouse: bool = False):
        # Don't do anything if not required
        if self.__zoom == value:
            return
        # Check limits
        if limit:
            fit_zoom = self.compute_fit_zoom()
            zoom_in = value > self.__zoom
            underzoom = value < fit_zoom
            if underzoom and not zoom_in:
                value = self.__zoom
        # Apply the zoom
        if mouse:
            self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        else:
            self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.view.setTransform(QTransform.fromScale(value, value), combine=False)
        self.__zoom = value

    def zoom_in(self):
        self.zoom(self.__zoom * self.__ZOOM_FACTOR)

    def zoom_out(self):
        self.zoom(self.__zoom / self.__ZOOM_FACTOR)

    def fit(self):
        self.zoom(self.compute_fit_zoom())

    def zoom_original(self):
        self.zoom(1.0, limit=False)

    @property
    def pixmap(self) -> QPixmap:
        return self.__pixmap

    @property
    def image(self) -> QImage:
        return self.__image

    @property
    def current_zoom(self) -> float:
        return self.__zoom

    @property
    def overflow(self) -> bool:
        return self.__zoom > self.compute_fit_zoom()

    @property
    def fitting(self) -> bool:
        return self.__zoom == self.compute_fit_zoom()

    def __zoom_round_to_steps(self) -> int:
        if self.__zoom > 1.0:
            step_zoom = 1.0
            step = 0
            while step_zoom < self.__zoom:
                step_zoom = step_zoom * self.__ZOOM_FACTOR
                step += 1
            return step

        step_zoom = 1.0
        step = 0
        while step_zoom > self.__zoom:
            step_zoom = step_zoom / self.__ZOOM_FACTOR
            step -= 1
        return step

    def __wheel_event(self, event: QWheelEvent):
        steps = event.angleDelta().y() // 120
        scale_steps = self.__zoom_round_to_steps() - steps
        zoom = 1.0
        for _ in range(abs(scale_steps)):
            if scale_steps > 0:
                zoom = zoom * self.__ZOOM_FACTOR
            else:
                zoom = zoom / self.__ZOOM_FACTOR
        self.zoom(zoom, mouse=True)
        event.accept()

    def __mouse_move_event(self, event: QMouseEvent):
        QGraphicsView.mouseMoveEvent(self.view, event)
        # Forwarding signal
        pixel = _floor_point(self.view.mapToScene(event.pos()))
        rect = self.__pixmap.rect()
        if (
            pixel.x() >= rect.x()
            and pixel.x() < rect.width()
            and pixel.y() >= rect.y()
            and pixel.y() < rect.height()
        ):
            self.mouse_hover_pixel.emit(pixel)
        # Drag
        if self.__draglast is not None:
            vector = self.view.mapToScene(event.pos()) - self.view.mapToScene(self.__draglast)
            self.__draglast = event.pos()
            self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
            self.view.translate(vector.x(), vector.y())
        event.accept()

    def __mouse_press_event(self, event: QMouseEvent):
        QGraphicsView.mousePressEvent(self.view, event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_left_press_pixel.emit(_floor_point(self.view.mapToScene(event.pos())))
            if self.overflow:
                self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.__draglast = event.pos()
        if event.button() == Qt.MouseButton.RightButton:
            self.mouse_right_press_pixel.emit(_floor_point(self.view.mapToScene(event.pos())))
        event.accept()

    def __mouse_release_event(self, event: QMouseEvent):
        QGraphicsView.mouseReleaseEvent(self.view, event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_left_release_pixel.emit(_floor_point(self.view.mapToScene(event.pos())))
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.__draglast = None
        if event.button() == Qt.MouseButton.RightButton:
            self.mouse_right_release_pixel.emit(_floor_point(self.view.mapToScene(event.pos())))
        if event.button() == Qt.MouseButton.MiddleButton:
            self.zoom(1.0, limit=False)
        event.accept()

    def __mouse_double_click_event(self, event: QMouseEvent):
        QGraphicsView.mouseDoubleClickEvent(self.view, event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_left_doubleclick_pixel.emit(_floor_point(self.view.mapToScene(event.pos())))
        if event.button() == Qt.MouseButton.RightButton:
            self.mouse_right_doubleclick_pixel.emit(_floor_point(self.view.mapToScene(event.pos())))
        event.accept()
