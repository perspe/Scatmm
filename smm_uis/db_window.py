""" Class for the DB Window """
import logging
import os

from PyQt5 import QtGui
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidget, QWidget
from modules.fig_class import FigWidget, PltFigure
import numpy as np
from scipy.interpolate import interp1d

from .smm_database_window import Ui_Database
from .smm_import_db_mat import Ui_ImportDB

Units = {"nm": 1, "um": 1e3, "mm": 1e6}


class DBWindow(QWidget):
    """ Main Class for the DB Window """
    def __init__(self, parent, database):
        """ Initialize the elements of the main window """
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
        self.update_db_preview()
        self.initializeUI()

    def initializeUI(self):
        """ Connect elements to specific functions """
        self.ui.add_material.clicked.connect(self.add_db_material)
        self.ui.rmv_material.clicked.connect(self.rmv_db_material)
        self.ui.view_material.clicked.connect(self.db_preview)

    def update_db_preview(self):
        """
        Update visualizer of database materials
        """
        self.data.clear()
        self.data.setHorizontalHeaderLabels(
            ["Material", "Min Wav (µm)", "Max Wav (µm)"])
        for index, material in enumerate(self.database.content):
            data_array = self.database[material]
            lmb_min = np.min(data_array[:, 0])
            lmb_max = np.max(data_array[:, 0])
            data = [
                QStandardItem(material),
                QStandardItem(str(np.around(lmb_min / 1000, 2))),
                QStandardItem(str(np.around(lmb_max / 1000, 2)))
            ]
            self.data.insertRow(index, data)

    def update_mat_comboboxes(self):
        """
        Update checkboxes in simulation and optimization tabs with db materials
        """
        for smat, omat in zip(self.parent.sim_mat, self.parent.opt_mat):
            smat.clear()
            omat.clear()
            smat.addItems(self.database.content)
            omat.addItems(self.database.content)

    """ Functions for the different buttons in the DB Interface """

    def add_db_material(self):
        """
        Open a new UI to manage importing new data
        """
        self.new_mat = np.array([])
        self.db_import_window = QTableWidget()
        self.db_import_ui = Ui_ImportDB()
        self.db_import_ui.setupUi(self.db_import_window)
        self.db_import_window.show()
        self.db_import_ui.unit_combobox.addItems(list(Units.keys()))
        self.db_import_ui.choose_file_button.clicked.connect(
            self.choose_db_mat)
        self.db_import_ui.import_button.clicked.connect(self.import_db_mat)
        self.db_import_ui.preview_button.clicked.connect(self.preview_db_mat)

    def rmv_db_material(self):
        """
        Remove the currently selected material from the database
        """
        choice = self.db_table.currentIndex().row()
        if choice < 0:
            QMessageBox.information(self, "Choose a material",
                                    "No material chosen to be removed!!",
                                    QMessageBox.Ok, QMessageBox.Ok)
            # Put focus on the database window
            self.setFocus(True)
            self.activateWindow()
            self.raise_()
            return
        # Ask the user for confirmation to delete the material
        answer = QMessageBox.question(
            self, "Remove Material",
            f"Do you really want to delete {self.database.content[choice]}?",
            QMessageBox.No | QMessageBox.Yes, QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            mat_choice = self.database.content[choice]
            ret = self.database.rmv_content(mat_choice)
            # Rebuild the material comboboxes in the main gui
            self.update_mat_comboboxes()
            if ret == 0:
                QMessageBox.information(self, "Removed Successfully",
                                        "Material Removed Successfully!!",
                                        QMessageBox.Ok, QMessageBox.Ok)
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
            QMessageBox.information(self, "Choose a material",
                                    "No material chosen for preview!!",
                                    QMessageBox.Ok, QMessageBox.Ok)
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
        self.preview_choice = FigWidget(
            f"Preview {self.database.content[choice]}")
        self.preview_choice_fig = PltFigure(self.preview_choice.layout,
                                            "Wavelength (nm)",
                                            "n/k",
                                            width=7)
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

    """ Functions for the Add button """

    def choose_db_mat(self):
        """
        Open a AskFileDialog to choose a file with the material data and
        update the qtextlabel with the name of the file
        """
        filepath = QFileDialog.getOpenFileName(self, 'Open File')
        if filepath[0] == '':
            return
        if self.db_import_window is None:
            raise Exception("Unknown Error...")
        logging.debug("Chosen Unit: {chosen_unit}")
        filename = os.path.basename(filepath[0])
        logging.debug("Getting data for new material")
        self.new_mat = self.parent.get_data_from_file(filepath[0])
        if self.new_mat.shape[0] == 0:
            logging.debug("No data in file... Ignoring")
            return
        if self.new_mat.shape[1] < 3:
            title = "Incomplete import"
            msg = "The data must have 3 columns (wavelength/n/k)"
            QMessageBox.warning(self, title, msg, QMessageBox.Close,
                                QMessageBox.Close)
            self.new_mat = np.array([])
        else:
            self.db_import_ui.chosen_file_label.setText(filename)
        self.db_import_window.raise_()
        self.db_import_window.setFocus(True)
        self.db_import_window.activateWindow()

    def preview_db_mat(self):
        """
        Show a preview of the imported data with the interpolation
        """
        if self.db_import_window is None:
            raise Exception("Unknown Error...")
        if self.new_mat.shape[0] == 0:
            QMessageBox.information(self, "Choose File",
                                    "Must select a file before previewing",
                                    QMessageBox.Ok, QMessageBox.Ok)
            return
        lmb, n, k = self.new_mat[:, 0], self.new_mat[:, 1], self.new_mat[:, 2]
        interp_n = interp1d(lmb, n)
        interp_k = interp1d(lmb, k)
        # Plot the results
        # Preview chosen material
        self.preview_import = FigWidget("Preview New Material")
        self.preview_import_fig = PltFigure(self.preview_import.layout,
                                            "Wavelength (File Units)",
                                            "n/k",
                                            width=7)
        self.n_plot = self.preview_import_fig.axes
        self.n_plot.set_ylabel("n")
        self.n_plot.plot(lmb, n, "b.", label="n")
        self.n_plot.plot(lmb, interp_n(lmb), "b", label="interp n")
        self.k_plot = self.n_plot.twinx()
        self.k_plot.set_ylabel("k")
        self.k_plot.plot(lmb, k, "r.", label="k")
        self.k_plot.plot(lmb, interp_k(lmb), "r", label="interp k")
        self.n_plot.legend(loc="upper right")
        self.k_plot.legend(loc="center right")
        self.preview_import.show()

    def import_db_mat(self):
        """
        Add the chosen material to the database
        """
        if self.db_import_window is None:
            raise Exception("Unknown Error...")
        chosen_unit = self.db_import_ui.unit_combobox.currentText()
        mat_name = self.db_import_ui.mat_name_edit.text()
        if self.new_mat.shape[0] == 0:
            QMessageBox.information(self, "Choose File",
                                    "Must select a file before importing",
                                    QMessageBox.Ok, QMessageBox.Ok)
            return
        elif mat_name == '':
            QMessageBox.information(
                self, "No material name",
                "Please select a material name before importing",
                QMessageBox.Ok, QMessageBox.Ok)
            return
        self.new_mat[:, 0] *= Units[chosen_unit]
        print(self.new_mat[:, 0])
        self.database.add_content(mat_name, self.new_mat)
        self.new_mat = np.array([])
        self.db_import_window.close()
        QMessageBox.information(self, "Import Successful",
                                "Material imported successfully",
                                QMessageBox.Ok, QMessageBox.Ok)
        # Update the database previews
        self.update_db_preview()
        # Update the comboboxes with the materials
        self.update_mat_comboboxes()
        self.raise_()
        self.setFocus(True)
        self.activateWindow()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.db_import_window is not None:
            self.db_import_window.close()
        return super().closeEvent(a0)
