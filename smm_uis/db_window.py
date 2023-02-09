""" Class for the DB Window """
import logging
from typing import List, Tuple, Union
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QHeaderView,
    QMenu,
    QMessageBox,
    QTableView,
    QWidget,
    QAbstractItemView,
)
from modules.fig_class import PltFigure
import numpy as np
import uuid

from .smm_view_database import Ui_Database
from .formula_window import FormulaWindow
from .imp_window import ImpPrevWindow


class dbTableView(QTableView):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setAcceptDrops(False)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


class DBWindow(QWidget):
    """Main Class for the DB Window"""

    db_updated = pyqtSignal()

    def __init__(self, parent, database):
        """Initialize the elements of the main window"""
        self.database = database
        self.parent = parent
        super(DBWindow, self).__init__()
        self.ui = Ui_Database()
        self.ui.setupUi(self)
        self.data = QStandardItemModel()
        self.db_table = dbTableView(self)
        self.ui.table_layout.addWidget(self.db_table)
        self.db_table.setModel(self.data)
        self.data.setColumnCount(3)
        self.import_window: Union[None, QWidget] = None
        self.formula_window: Union[None, QWidget] = None
        self.update_db_preview()
        # Setup plot area
        self.plot_canvas = PltFigure(self.ui.figure_widget, "Wavelength (nm)", "n")
        self.ui.widget.customContextMenuRequested.connect(self.plotContext)
        self.db_table.customContextMenuRequested.connect(self.tableContext)
        self.n_plot = self.plot_canvas.axes
        self.n_plot.spines["left"].set_color("b")
        self.n_plot.spines["right"].set_color("r")
        self.n_plot.tick_params(axis="y", colors="b")
        self.n_plot.set_ylabel("n", color="b")
        self.k_plot = self.n_plot.twinx()
        self.k_plot.set_ylabel("k", color="r")
        self.k_plot.spines["left"].set_color("b")
        self.k_plot.spines["right"].set_color("r")
        self.k_plot.tick_params(axis="y", colors="r")
        self._ploted_info: List[Tuple[str, uuid.UUID]] = []
        self._action_list: List[QAction] = []
        self.initializeUI()

    def initializeUI(self):
        """Connect elements to specific functions"""
        self.ui.add_material.clicked.connect(self.add_db_material)
        self.ui.add_formula.clicked.connect(self.add_formula)
        self.ui.rmv_material.clicked.connect(self.rmv_db_material)

    def update_db_preview(self):
        """
        Update visualizer of database materials
        """
        logging.debug("Updating DB list")
        self.data.clear()
        self.data.setHorizontalHeaderLabels(
            ["Material", "Min Wav (µm)", "Max Wav (µm)"]
        )
        for index, material in enumerate(self.database.content):
            data_array = self.database[material]
            lmb_min = np.min(data_array[:, 0])
            lmb_max = np.max(data_array[:, 0])
            data = [
                QStandardItem(material),
                QStandardItem(str(np.around(lmb_min / 1000, 2))),
                QStandardItem(str(np.around(lmb_max / 1000, 2))),
            ]
            self.data.insertRow(index, data)

    """ Functions for the different buttons in the DB Interface """

    def add_db_material(self):
        """
        Open a new UI to manage importing new data
        """
        if self.import_window is not None:
            self.import_window.raise_()
            return
        self.import_window = ImpPrevWindow(self)
        self.import_window.imp_clicked.connect(self._import)
        self.import_window.show()

    @pyqtSlot(object, str)
    def _import(self, import_data, name):
        logging.debug(f"Imported data detected:\n {import_data=}\n{name=}")
        if self.import_window is None:
            logging.critical("Unknown Error")
            return
        override = QMessageBox.NoButton
        if name in self.database.content:
            override = QMessageBox.question(
                self,
                "Material name in Database",
                "The Database already has a material with this name.."
                + "\nOverride the material?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
        logging.debug(f"{override=}")
        if override == QMessageBox.No:
            return
        elif override == QMessageBox.Yes:
            self.database.rmv_content(name)
            self.database.add_content(name, import_data.values[:, [0, 1, 2]])
        else:
            self.database.add_content(name, import_data.values[:, [0, 1, 2]])
        self.update_db_preview()
        self.db_updated.emit()
        self.import_window.close()

    def add_formula(self):
        """Open a new UI with the formula manager"""
        if self.formula_window is not None:
            self.formula_window.raise_()
            return
        self.formula_window = FormulaWindow(self)
        self.formula_window.show()

    def rmv_db_material(self):
        """
        Remove the currently selected material from the database
        """
        choice = self.db_table.currentIndex().row()
        if choice < 0:
            QMessageBox.information(
                self,
                "Choose a material",
                "No material chosen to be removed!!",
                QMessageBox.Ok,
                QMessageBox.Ok,
            )
            # Put focus on the database window
            self.setFocus(True)
            self.activateWindow()
            self.raise_()
            return
        # Ask the user for confirmation to delete the material
        answer = QMessageBox.question(
            self,
            "Remove Material",
            f"Do you really want to delete {self.database.content[choice]}?",
            QMessageBox.No | QMessageBox.Yes,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            mat_choice = self.database.content[choice]
            ret = self.database.rmv_content(mat_choice)
            # Rebuild the material comboboxes in the main gui
            self.db_updated.emit()
            if ret == 0:
                QMessageBox.information(
                    self,
                    "Removed Successfully",
                    "Material Removed Successfully!!",
                    QMessageBox.Ok,
                    QMessageBox.Ok,
                )
            self.update_db_preview()
        # Put focus on the Database window
        self.setFocus(True)
        self.activateWindow()
        self.raise_()

    """ Events to add """

    def dragEnterEvent(self, a0: QtGui.QDragEnterEvent) -> None:
        widget = a0.source()
        if a0.mimeData().hasFormat("application/x-qstandarditemmodeldatalist"):
            a0.accept()
        else:
            a0.ignore()
        return super().dragEnterEvent(a0)

    def dropEvent(self, a0: QtGui.QDropEvent) -> None:
        widget = a0.source()
        target = QApplication.widgetAt(self.mapToParent(a0.pos()))
        if isinstance(target, PltFigure):
            logging.debug("Accepting Drop...")
            mat_index = widget.currentIndex().row()
            data = self.database[mat_index]
            lmb, n, k = data[:, 0], data[:, 1], data[:, 2]
            # Preview chosen material
            mat_name: str = self.database.content[mat_index]
            mat_id: uuid.UUID = uuid.uuid4()
            self.n_plot.plot(lmb, n, "b.", label=mat_name, gid=mat_id)
            self.k_plot.plot(lmb, k, "r.", gid=mat_id)
            self.n_plot.legend()
            self._update_plot_limits()
            self.plot_canvas.draw()
            self._ploted_info.append((mat_name, mat_id))
            a0.accept()
        return super().dropEvent(a0)

    """ Context Menus and helper functions """

    def plotContext(self, position):
        plot_menu = QMenu(self)
        self._action_list: List[QAction] = [
            QAction(mat_name, self, checkable=True) for mat_name, _ in self._ploted_info
        ]
        for action in self._action_list:
            action.setChecked(True)
            action.triggered.connect(self._update_preview)
        plot_menu.addActions(self._action_list)
        plot_menu.exec_(self.ui.widget.mapToGlobal(position))

    def _update_preview(self, action):
        logging.debug("Updating Preview Plot")
        index = 0
        for index, action in enumerate(self._action_list):
            if not action.isChecked():
                _, mat_gid = self._ploted_info[index]
                self._delete_plot(mat_gid)
                self.n_plot.legend()
                self.plot_canvas.draw()
                break
        self._action_list.pop(index)
        self._ploted_info.pop(index)
        logging.debug(f"Updated action list:\n{self._action_list}")

    def _update_plot_limits(self):
        n_lines = self.n_plot.get_lines()
        k_lines = self.k_plot.get_lines()
        n_min = np.min([np.min(y_data.get_data()[1]) for y_data in n_lines])
        n_max = np.max([np.max(y_data.get_data()[1]) for y_data in n_lines])
        k_min = np.min([np.min(y_data.get_data()[1]) for y_data in k_lines])
        k_max = np.max([np.max(y_data.get_data()[1]) for y_data in k_lines])
        wvl_min = np.min([np.min(y_data.get_data()[0]) for y_data in k_lines])
        wvl_max = np.max([np.max(y_data.get_data()[0]) for y_data in k_lines])
        logging.debug(f"Updated Limits:{n_min}::{n_max}::{k_min}::{k_max}::{wvl_min}::{wvl_max}")
        self.n_plot.set_ylim(n_min, n_max)
        self.k_plot.set_ylim(k_min, k_max)
        self.n_plot.set_xlim(wvl_min, wvl_max)
        self.k_plot.set_xlim(wvl_min, wvl_max)

    def _delete_plot(self, id: uuid.UUID):
        """Generic function to delete plots"""
        logging.debug(f"Deleting ID plot: {id}")
        for n_plot, k_plot in zip(self.n_plot.get_lines(), self.k_plot.get_lines()):
            if n_plot.get_gid() == id:
                n_plot.remove()
            if k_plot.get_gid() == id:
                k_plot.remove()
        if len(self.n_plot.get_lines()) == 0:
            return
        self._update_plot_limits()

    def _table_actions(self):
        pass

    def tableContext(self, position):
        print("Table Context")
        pass

    """ Close Event """

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.import_window is not None:
            self.import_window.close()
        if self.formula_window is not None:
            self.formula_window.close()
        self.parent.db_ui = None
        return super().closeEvent(a0)
