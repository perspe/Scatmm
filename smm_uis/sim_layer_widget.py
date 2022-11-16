from typing import List, Tuple
import uuid

from PyQt5 import QtGui
from PyQt5.QtCore import QLocale, QMimeData, QPoint, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QMenu,
    QVBoxLayout,
    QWidget,
)
from matplotlib import logging

from .smm_simlayer_widget import Ui_SimLayer

log_config = {
    "format": "%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:"
    "%(lineno)d:%(message)s",
    "level": logging.DEBUG,
}
logging.basicConfig(**log_config)


class SimLayerWidget(QWidget):
    """
    Main widget for the Simulation Layer
    This widget has the mouseMoveEvent for the drag and drop
    Signals:
        checked: checkbox checked
        deleted: deleted button clicked
    Args:
        materials: List with the materials for the combobox
        check (False): state for the checkbox
        disable (False): wether to disable checkbox or not
    Methods:
        clear_abs_cb: Uncheck the absorption checkbox
        thickness, material, uuid: Internal variables stored
        update_materials: Update all the Combobox materials
    """

    checked = pyqtSignal(bool, uuid.UUID)
    deleted = pyqtSignal(uuid.UUID)

    def __init__(
        self, materials: List[str], check: bool = False, disable: bool = False
    ):
        QWidget.__init__(self)
        self.ui = Ui_SimLayer()
        self.ui.setupUi(self)
        # Item identifier
        self._uuid = uuid.uuid4()
        # Setup all the elements
        self.ui.mat_cb.addItems(materials)
        self.ui.abs_cb.setChecked(check)
        self.ui.abs_cb.setDisabled(disable)
        # Add validator
        _double_validator = QtGui.QDoubleValidator()
        _double_validator.setLocale(QLocale("en_US"))
        self.ui.thickness_edit.setValidator(_double_validator)
        # Add connectors
        self.ui.abs_cb.clicked.connect(lambda x: self.checked.emit(x, self._uuid))
        self.ui.del_button.clicked.connect(lambda: self.deleted.emit(self._uuid))
        self.setStyleSheet("background: #FFFFFFFF")
        self.show()

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        if a0.buttons() == Qt.LeftButton and self.ui.move_label.underMouse():
            logging.debug("Detected Mouse Movement")
            drag = QtGui.QDrag(self)
            mime = QMimeData()
            drag.setMimeData(mime)
            # Add a image to indicate the drag effect
            pixmap = QtGui.QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)
            drag.setHotSpot(QPoint(10, 20))
            # Move the item
            drag.exec_(Qt.MoveAction)
        return super().mouseMoveEvent(a0)

    """ Interaction with interanal widgets """

    def clear_abs_cb(self, disable: bool = False):
        """Clear the Absorption CheckBox"""
        self.ui.abs_cb.blockSignals(True)
        self.ui.abs_cb.setChecked(False)
        self.ui.abs_cb.setDisabled(disable)
        self.ui.abs_cb.blockSignals(False)

    """ Obtain the internal properties of the subwidgets """

    def thickness(self) -> float:
        return float(self.ui.thickness_edit.text())

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


class SimLayerLayout(QWidget):
    """
    Main Widget to hold all the SimLayerWidgets
    Signals:
        abs_clicked: Any absorption checkbox was clicked
    Args:
        materials: List with the materials for the Combobox
        layers: Number of layers to start with
    Methods:
        add_layer: add a new layer
        rmv_layer: rmv a specific layer
        rmv_layer_id: Remove layer from uuid
        update_cb_items: Update all the items in the Comboboxes
        reinit_abs_checkbox: Reinitialize all the abs checkboxes
        layer_info: return all the info from all the layers
    """

    abs_clicked = pyqtSignal(int, bool, uuid.UUID)

    def __init__(self, parent, materials: List[str], layers: int = 2):
        super().__init__(parent)
        self._parent = parent
        self.setAcceptDrops(True)
        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(10, 0, 10, 0)
        # Add Actions for the menu
        add_above_action = QAction("Add Layer Above", self)
        self._actions = [add_above_action]
        # Add the default number of layers
        for _ in range(layers):
            layer_widget = SimLayerWidget(materials)
            layer_widget.checked.connect(lambda x, y: self.clicked_abs(x, y))
            layer_widget.deleted.connect(lambda x: self.rmv_layer_id(x))
            self.vlayout.addWidget(layer_widget)
        self.reinit_abs_checkbox(disable=True)
        self.setLayout(self.vlayout)
        self.show()

    """ Functions to manage the layers """

    def add_layer(
        self, materials: List[str], check: bool = False, disable: bool = False
    ) -> None:
        """Add a new layer"""
        widget = SimLayerWidget(materials, check, disable)
        widget.deleted.connect(lambda x: self.rmv_layer_id(x))
        widget.checked.connect(lambda x, y: self.clicked_abs(x, y))
        self.vlayout.addWidget(widget)
        self.reinit_abs_checkbox(disable=True)

    def rmv_layer(self, index: int) -> None:
        """Remove layer from particular index"""
        if self.vlayout.count() <= 1:
            logging.info("There should be at least one singular layer")
            return
        self.rmv = self.vlayout.itemAt(index).widget()
        self.rmv.deleteLater()
        self.reinit_abs_checkbox(disable=True)

    def rmv_layer_id(self, id: uuid.UUID) -> None:
        """Delete particular item from uuid"""
        for n in range(self.vlayout.count()):
            widget = self.vlayout.itemAt(n).widget()
            if widget.uuid == id:
                self.rmv_layer(n)

    def layer_info(self) -> List[Tuple[str, float]]:
        """Return all the information about the current layers"""
        info: List[Tuple[str, float]] = []
        for n in range(self.vlayout.count()):
            widget = self.vlayout.itemAt(n).widget()
            material: str = widget.material()
            thickness: float = widget.thickness()
            info.append((material, thickness))
        logging.debug(f"Layer Info:{info}")
        return info

    def update_cb_items(self, materials: List[str]) -> None:
        """Add a new item to the combobox"""
        for n in range(self.vlayout.count()):
            widget = self.vlayout.itemAt(n).widget()
            widget.update_materials(materials)

    def reinit_abs_checkbox(self, disable: bool = False) -> None:
        """Clear all the absorption checkboxes"""
        for n in range(self.vlayout.count()):
            logging.debug(f"Clear CB {n}")
            widget = self.vlayout.itemAt(n).widget()
            widget.clear_abs_cb(disable)

    """ Signals """

    def clicked_abs(self, state: bool, id: uuid.UUID) -> None:
        """
        Emit signal indicating:
            Index of the checked widget in the stack
            State of the widget
            ID of the checked widget
        """
        clicked_index: int = 0
        for n in range(self.vlayout.count()):
            widget = self.vlayout.itemAt(n).widget()
            if widget.uuid == id:
                clicked_index = n
                break
        logging.debug(f"CB Index: {clicked_index}::State {state}")
        self.abs_clicked.emit(clicked_index, state, id)

    """ Drag Event functions """

    def dragEnterEvent(self, a0: QtGui.QDragEnterEvent):
        if a0.mimeData().hasUrls():
            logging.debug(f"Drag with URL")
            return super().dragEnterEvent(a0)
        else:
            logging.debug(f"Drag with SimWidget")
            source = a0.source()
            source.setDisabled(True)
            a0.accept()
            return super().dragEnterEvent(a0)

    def dragMoveEvent(self, a0: QtGui.QDragMoveEvent) -> None:
        return super().dragMoveEvent(a0)

    def dropEvent(self, a0: QtGui.QDropEvent) -> None:
        """Handle the widget drop"""
        pos = a0.pos()
        widget = a0.source()
        index = self.vlayout.indexOf(widget)
        for n in range(self.vlayout.count()):
            w = self.vlayout.itemAt(n).widget()
            if pos.y() < w.y() + w.size().height() // 2:
                place_location = n - 1 if n > index else n
                logging.debug(f"Droping to position {place_location}")
                self.vlayout.insertWidget(place_location, widget)
                break
        # Reenable all widgets and set the right tab order
        for n in range(self.vlayout.count() - 1):
            curr_item = self.vlayout.itemAt(n).widget()
            next_item = self.vlayout.itemAt(n + 1).widget()
            self._parent.setTabOrder(curr_item.ui.abs_cb, curr_item.ui.mat_cb)
            self._parent.setTabOrder(curr_item.ui.mat_cb, curr_item.ui.thickness_edit)
            self._parent.setTabOrder(curr_item.ui.thickness_edit, next_item.ui.abs_cb)
            self._parent.setTabOrder(next_item.ui.abs_cb, next_item.ui.mat_cb)
            self._parent.setTabOrder(next_item.ui.mat_cb, next_item.ui.thickness_edit)
            self.vlayout.itemAt(n).widget().setDisabled(False)
        self.vlayout.itemAt(self.vlayout.count() - 1).widget().setDisabled(False)
        a0.accept()
        logging.debug("Droped object")
        self.reinit_abs_checkbox(disable=True)
        return super().dropEvent(a0)

    def dragLeaveEvent(self, a0: QtGui.QDragLeaveEvent) -> None:
        """
        Perform cleanup in case the the object is draged outside the region
        """
        logging.debug(f"Drag Left acceptable region...")
        a0.setAccepted(False)
        for n in range(self.vlayout.count()):
            self.vlayout.itemAt(n).widget().setDisabled(False)
        return super().dragLeaveEvent(a0)

    """ Add Context Menu """

    # TODO
    def contextMenuEvent(self, a0: QtGui.QContextMenuEvent) -> None:
        logging.debug("Entering Context Menu")
        context = QMenu(self)
        context.addActions(self._actions)
        context.exec_(self.mapToGlobal(a0.pos()))
        return super().contextMenuEvent(a0)


if __name__ == "__main__":
    app = QApplication([])
    sim_layer = SimLayerLayout(["asddas", "asdfasdf"])
    app.exec_()
