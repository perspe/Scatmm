""" Class for the DB Window """
import logging
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QMessageBox, QWidget
from modules.fig_class import FigWidget, PltFigure
import numpy as np
from scipy.interpolate import interp1d

from .smm_view_database import Ui_Database
from .formula_window import FormulaWindow
from .imp_window import ImpPrevWindow


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
        self.db_table = self.ui.database_table
        self.db_table.setModel(self.data)
        self.data.setColumnCount(3)
        self.db_import_window = None
        self.formula_window = None
        self.update_db_preview()
        self.initializeUI()

    def initializeUI(self):
        """Connect elements to specific functions"""
        self.ui.add_material.clicked.connect(self.add_db_material)
        self.ui.add_formula.clicked.connect(self.add_formula)
        self.ui.rmv_material.clicked.connect(self.rmv_db_material)
        self.ui.view_material.clicked.connect(self.db_preview)

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
        self.db_import_window = ImpPrevWindow(self)
        self.db_import_window.imp_clicked.connect(self._import)
        self.db_import_window.show()

    @pyqtSlot(object, str)
    def _import(self, import_data, name):
        logging.debug(f"Imported data detected:\n {import_data=}\n{name=}")
        if self.db_import_window is None:
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
        self.db_import_window.close()

    def add_formula(self):
        """Open a new UI with the formula manager"""
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

    def db_preview(self):
        """
        Get selected material from the QTableView Widget and plot the
        stored values and their interpolations
        """
        choice = self.db_table.currentIndex().row()
        if choice < 0:
            QMessageBox.information(
                self,
                "Choose a material",
                "No material chosen for preview!!",
                QMessageBox.Ok,
                QMessageBox.Ok,
            )
            # Put focus on the database window
            self.setFocus(True)
            self.activateWindow()
            self.raise_()
            return
        # Get the database values and the interpolations
        data = self.database[choice]
        lmb, n, k = data[:, 0], data[:, 1], data[:, 2]
        interp_n = interp1d(lmb, n)
        interp_k = interp1d(lmb, k)
        # Preview chosen material
        self.preview_choice = FigWidget(f"Preview {self.database.content[choice]}")
        self.preview_choice_fig = PltFigure(
            self.preview_choice.layout, "Wavelength (nm)", "n/k", width=7
        )
        self.n_plot = self.preview_choice_fig.axes
        self.n_plot.set_ylabel("n")
        self.n_plot.plot(lmb, n, "b.", label="n")
        self.n_plot.plot(lmb, interp_n(lmb), "b", label="interp n")
        self.k_plot = self.n_plot.twinx()
        self.k_plot.set_ylabel("k")
        self.k_plot.plot(lmb, k, "r.", label="k")
        self.k_plot.plot(lmb, interp_k(lmb), "r", label="interp k")
        self.n_plot.legend(loc="upper right")
        self.k_plot.legend(loc="center right")
        self.preview_choice.show()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.db_import_window is not None:
            self.db_import_window.close()
        if self.formula_window is not None:
            self.formula_window.close()
        return super().closeEvent(a0)
