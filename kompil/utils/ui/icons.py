import os
from typing import Callable
from PyQt5.QtGui import QIcon
from kompil.utils.paths import PATH_ROOT

__LOADED_THEME_ICONS = dict()
__LOADED_FILE_ICONS = dict()


def __icon_from_theme(theme: str) -> Callable[[], QIcon]:
    def build_icon() -> QIcon:
        global __LOADED_THEME_ICONS
        nonlocal theme
        if theme not in __LOADED_THEME_ICONS:
            __LOADED_THEME_ICONS[theme] = QIcon.fromTheme(theme)
        return __LOADED_THEME_ICONS[theme]

    return build_icon


def __icon_from_file(file: str) -> Callable[[], QIcon]:
    def build_icon() -> QIcon:
        global __LOADED_FILE_ICONS
        nonlocal file
        if file not in __LOADED_FILE_ICONS:
            __LOADED_FILE_ICONS[file] = QIcon(file)
        return __LOADED_FILE_ICONS[file]

    return build_icon


# Icons from file
get_icon_kompil = __icon_from_file(os.path.join(PATH_ROOT, "docs", "imgs", "logo.png"))

# Icons from theme
get_icon_play = __icon_from_theme("media-playback-start")
get_icon_pause = __icon_from_theme("media-playback-pause")
get_icon_shot = __icon_from_theme("camera-photo")
get_icon_fit = __icon_from_theme("zoom-fit-best")
get_icon_fullscreen = __icon_from_theme("view-fullscreen")
get_icon_zoom_in = __icon_from_theme("zoom-in")
get_icon_zoom_out = __icon_from_theme("zoom-out")
get_icon_zoom_original = __icon_from_theme("zoom-original")
get_icon_go_jump = __icon_from_theme("go-jump")
get_icon_list_add = __icon_from_theme("list-add")
get_icon_list_remove = __icon_from_theme("list-remove")


if __name__ == "__main__":
    from kompil.utils.ui.app import QKompilApp
    from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSizePolicy

    def new_btn(icon):
        button = QPushButton(icon, "")
        button.setFlat(True)
        button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        return button

    app = QKompilApp([])

    layout = QHBoxLayout()
    layout.addWidget(new_btn(get_icon_play()))
    layout.addWidget(new_btn(get_icon_pause()))
    layout.addWidget(new_btn(get_icon_shot()))
    layout.addWidget(new_btn(get_icon_fit()))
    layout.addWidget(new_btn(get_icon_fullscreen()))
    layout.addWidget(new_btn(get_icon_zoom_in()))
    layout.addWidget(new_btn(get_icon_zoom_out()))
    layout.addWidget(new_btn(get_icon_zoom_original()))
    layout.addWidget(new_btn(get_icon_go_jump()))
    layout.addWidget(new_btn(get_icon_list_add()))
    layout.addWidget(new_btn(get_icon_list_remove()))

    win = QWidget()
    win.show()
    win.setLayout(layout)
    exit(app.exec_())
