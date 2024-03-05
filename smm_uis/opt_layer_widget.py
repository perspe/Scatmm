from typing import List, Tuple, Union
import uuid

from PyQt5 import QtGui
from PyQt5.QtCore import (
        QEvent, QLocale, QMimeData, QObject, Qt, pyqtSignal, QPoint, QByteArray
)
from PyQt5.QtWidgets import (
        QAction, QApplication, QMenu, QVBoxLayout, QWidget
)
import logging
from .smm_oplayer_widget import Ui_OpLayer

log_config = {
    "format": "%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:"
    "%(lineno)d:%(message)s",
    "level": logging.DEBUG,
}
logging.basicConfig(**log_config)


class OptLayerWidget(QWidget):
    """
    Main widget for the Optimization Layer
    This widget has the mouseMoveEvent for the drag and drop
    Signals:
        deleted: deleted button clicked
    Args:
        materials: List with the materials for the combobox
    Methods:
        thickness, material, uuid: Internal variables stored
        update_materials: Update all the Combobox materials
    """

    deleted = pyqtSignal(uuid.UUID)

    def __init__(self, materials: List[str]):
        super().__init__()
        self.ui = Ui_OpLayer()
        self.ui.setupUi(self)
        # Item identifier
        self._uuid: uuid.UUID = uuid.uuid4()
        # Setup all the elements
        self.ui.mat_cb.addItems(materials)
        _double_validator = QtGui.QDoubleValidator()
        _double_validator.setLocale(QLocale("en_US"))
        self.ui.tlow_edit.setValidator(_double_validator)
        self.ui.tup_edit.setValidator(_double_validator)
        # Add connectors
        self.ui.del_button.clicked.connect(lambda: self.deleted.emit(self._uuid))
        self.ui.mat_cb.installEventFilter(self)
        self.show()

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        if a0.buttons() == Qt.LeftButton and self.ui.move_label.underMouse():
            logging.debug("Detected Mouse Movement")
            drag = QtGui.QDrag(self)
            mime = QMimeData()
            mime.setData("widget/layer_widget", QByteArray())
            drag.setMimeData(mime)
            # Add a image to indicate the drag effect
            pixmap = QtGui.QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)
            drag.setHotSpot(QPoint(10, 20))
            # Move the item
            drag.exec_(Qt.MoveAction)
        return super().mousePressEvent(a0)


    """ Obtain the internal properties of the subwidgets """

    def thickness(self) -> Tuple[float, float]:
        return float(self.ui.tlow_edit.text()), float(self.ui.tup_edit.text())

    def material(self) -> str:
        return self.ui.mat_cb.currentText()

    @property
    def uuid(self) -> uuid.UUID:
        return self._uuid

    def update_materials(self, materials: List[str]) -> None:
        """Update all the materials in the Combobox"""
        curr_item: str = self.ui.mat_cb.currentText()
        self.ui.mat_cb.clear()
        self.ui.mat_cb.addItems(materials)
        if curr_item in materials:
            self.ui.mat_cb.setCurrentText(curr_item)
        logging.debug(f"Updated CB: {curr_item} also set")

    def eventFilter(self, a0: 'QObject', a1: 'QEvent') -> bool:
        if a1.type() == QEvent.Wheel and a0 is self.ui.mat_cb:
            return True
        return super().eventFilter(a0, a1)


class OptLayerLayout(QWidget):
    """
    Main Widget to hold all the OptLayerWidgets
    Args:
        materials: List with the materials for the Combobox
        layers: Number of layers to start with
    Methods:
        add_layer: add a new layer
        rmv_layer: rmv a specific layer
        rmv_layer_id: Remove layer from uuid
        update_cb_items: Update all the items in the Comboboxes
        layer_info: return all the info from all the layers
    """

    def __init__(self, parent, materials: List[str], layers: int = 2):
        super().__init__(parent)
        self._parent = parent
        self.setAcceptDrops(True)
        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(10, 0, 10, 0)
        # Add the default number of layers
        for _ in range(layers):
            layer_widget = OptLayerWidget(materials)
            layer_widget.deleted.connect(lambda x: self.rmv_layer_id(x))
            self.vlayout.addWidget(layer_widget)
        self.setLayout(self.vlayout)
        self.create_actions()
        self.show()

    def create_actions(self):
        """ Create all the actions """
        add_above_action = QAction("Add Layer Above", self)
        self._actions = [add_above_action]

    """ Functions to manage the layers """

    def add_layer(self, materials: List[str]) -> None:
        """Add a new layer"""
        widget = OptLayerWidget(materials)
        widget.deleted.connect(lambda x: self.rmv_layer_id(x))
        self.vlayout.addWidget(widget)

    def rmv_layer(self, index: int) -> None:
        """Remove layer from particular index"""
        if self.vlayout.count() <= 1:
            logging.info("There should be at least one singular layer")
            return
        self.rmv = self.vlayout.itemAt(index).widget()
        self.rmv.deleteLater()

    def rmv_layer_id(self, id: uuid.UUID) -> None:
        """Delete particular item from uuid"""
        for n in range(self.vlayout.count()):
            widget = self.vlayout.itemAt(n).widget()
            if widget.uuid == id:
                self.rmv_layer(n)

    def layer_info(self) -> List[Tuple[str, Tuple[float, float]]]:
        """Return all the information about the current layers"""
        info: List[Tuple[str, Tuple[float, float]]] = []
        for n in range(self.vlayout.count()):
            widget = self.vlayout.itemAt(n).widget()
            material: str = widget.material()
            thickness: Tuple[float, float] = widget.thickness()
            info.append((material, thickness))
        logging.debug(f"Layer Info:{info}")
        return info

    def update_cb_items(self, materials: List[str]) -> None:
        """Add a new item to the combobox"""
        for n in range(self.vlayout.count()):
            widget = self.vlayout.itemAt(n).widget()
            widget.update_materials(materials)

    def setEnabledAll(self, enabled: bool=True) -> None:
        """ Re enable all layer widgets """
        for n in range(self.vlayout.count()):
            self.vlayout.itemAt(n).widget().setEnabled(enabled)

    """ Drag Event functions """

    def dragEnterEvent(self, a0: QtGui.QDragEnterEvent):
        if a0.mimeData().hasUrls():
            logging.debug(f"Drag entered with URL")
        elif a0.mimeData().hasFormat("widget/layer_widget"):
            logging.debug(f"Drag entered with OptWidget")
            source = a0.source()
            self.setEnabledAll(True)
            source.setDisabled(True)
            a0.accept()
        else:
            raise Exception("Not expected drag type")
        return super().dragEnterEvent(a0)

    def dropEvent(self, a0: QtGui.QDropEvent) -> None:
        """Handle the widget drop"""
        pos = a0.pos()
        widget = a0.source()
        index = self.vlayout.indexOf(widget)
        # Properly place widget
        for n in range(self.vlayout.count()):
            w = self.vlayout.itemAt(n).widget()
            if pos.y() < w.y() + w.size().height() // 2:
                place_location = n - 1 if n > index else n
                logging.debug(f"Droping to position {place_location}")
                self.vlayout.insertWidget(place_location, widget)
                break
        # Update Tab Layout
        for n in range(self.vlayout.count() - 1):
            curr_item = self.vlayout.itemAt(n).widget()
            next_item = self.vlayout.itemAt(n + 1).widget()
            self._parent.setTabOrder(curr_item.ui.mat_cb, curr_item.ui.tlow_edit)
            self._parent.setTabOrder(curr_item.ui.tlow_edit, curr_item.ui.tup_edit)
            self._parent.setTabOrder(curr_item.ui.tup_edit, next_item.ui.mat_cb)
            self._parent.setTabOrder(next_item.ui.mat_cb, next_item.ui.tlow_edit)
            self._parent.setTabOrder(next_item.ui.tlow_edit, next_item.ui.tup_edit)
            self.vlayout.itemAt(n).widget().setDisabled(
                False
            )  # Reenable all widgets and set the right tab order
        self.vlayout.itemAt(self.vlayout.count() - 1).widget().setDisabled(False)
        a0.accept()
        return super().dropEvent(a0)

    def dragLeaveEvent(self, a0: QtGui.QDragLeaveEvent) -> None:
        """Perform cleanup in case the object is dragged outside the region"""
        logging.debug(f"Drag Left acceptable region...")
        self.setEnabledAll(True)
        return super().dragLeaveEvent(a0)

    """ Add Context Menu """

    def contextMenuEvent(self, a0: QtGui.QContextMenuEvent) -> None:
        logging.debug("Entering Context Menu")
        context = QMenu(self)
        context.addActions(self._actions)
        context.exec_(self.mapToGlobal(a0.pos()))
        return super().contextMenuEvent(a0)


if __name__ == "__main__":
    app = QApplication([])
    sim_layer = OptLayerWidget(["asddas", "asdfasdf"])
    app.exec_()
