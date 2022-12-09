import torch
from typing import Optional

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex
from PyQt5.QtWidgets import QTreeView


class LayerSelectItem:
    def __init__(self, name: str, module: torch.nn.Module, parent: Optional["LayerSelectItem"]):
        self.name = name
        self.module = module
        self.parent = parent
        self.children = []


def _list_all_modules(module: torch.nn.Module) -> LayerSelectItem:
    def _feed_children(item: Optional[LayerSelectItem]) -> None:
        for child_name, child in item.module.named_children():
            child_full_name = child_name if item.name == "root" else item.name + "." + child_name
            child = LayerSelectItem(child_full_name, child, item)
            _feed_children(child)
            item.children.append(child)

    root = LayerSelectItem("root", module, None)
    _feed_children(root)
    return root


class ModuleSelectionModel(QAbstractItemModel):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.__data = _list_all_modules(module.sequence)

    def index(self, row: int, column: int, parent: Optional[QModelIndex] = None) -> QModelIndex:
        if parent is None or not parent.isValid():
            parent_ptr: LayerSelectItem = self.__data
        else:
            parent_ptr: LayerSelectItem = parent.internalPointer()

        if len(parent_ptr.children) == 0:
            return QModelIndex()

        return self.createIndex(row, column, parent_ptr.children[row])

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()
        parent = index.internalPointer().parent

        if parent is None:
            return QModelIndex()

        return self.createIndex(0, 0, parent)

    def rowCount(self, index: QModelIndex):
        if not index.isValid():
            return len(self.__data.children)
        return len(index.internalPointer().children)

    def columnCount(self, index: QModelIndex) -> int:
        return 1

    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        ptr: LayerSelectItem = index.internalPointer()
        if role == Qt.ItemDataRole.DisplayRole:
            return ptr.name + ", " + str(ptr.module).splitlines()[0]
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(ptr.module)


class LayerSelecter(QTreeView):
    def __init__(self, module: torch.nn.Module, parent=None) -> None:
        super().__init__(parent=parent)
        self.setModel(ModuleSelectionModel(module))
        self.setHeaderHidden(True)
