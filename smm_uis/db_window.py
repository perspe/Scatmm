""" Class for the DB Window """
import logging
import os
from typing import Union, Any

from PyQt5 import QtGui
from PyQt5.QtCore import QAbstractTableModel, QRegExp, Qt
from PyQt5.QtGui import QRegExpValidator, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget
from modules.fig_class import FigWidget, PltFigure
import numpy as np
import pandas as pd
from pandas.errors import ParserError
from scipy.interpolate import interp1d

from .smm_database_window import Ui_Database
from .smm_import_db_mat import Ui_ImportDB

Units = {"nm": 1, "um": 1e3, "mm": 1e6}
Decimal = {"Dot (.)": ".", "Comma (,)": ","}
Separator = {"comma (,)": ",", "space ( )": " ", ";": ";", "other": " "}
""" Model to show a preview of the imported data """


class TableModel(QAbstractTableModel):
    """ Abstract model to implement for the imported data preview
    This is built to work with pandas DFs
    """
    def __init__(self, data) -> None:
        super().__init__()
        self._data = data

    def data(self, index, role) -> Union[str, None]:
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, _) -> int:
        return self._data.shape[0]

    def columnCount(self, _) -> int:
        return self._data.shape[1]

    def headerData(self, section, orientation, role) -> Union[str, None]:
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return f"Column {section}"
            if orientation == Qt.Vertical:
                return str(self._data.index[section])


class ImportWindow(QWidget):
    def __init__(self, parent):
        self.parent = parent
        super(ImportWindow, self).__init__()
        self.ui = Ui_ImportDB()
        self.ui.setupUi(self)
        # Startup variables
        self.filepath = None
        self.data = None
        self.preview_import = None
        self.import_args = {
            "sep": " ",
            "decimal": ".",
            "skiprows": 0,
            "header": None
        }
        # Things for preview table
        self.import_table = self.ui.tableView
        self.import_model = TableModel(pd.DataFrame())
        self.import_table.setModel(self.import_model)
        self.ui.unit_combobox.addItems(list(Units.keys()))
        self.initializeUI()

    def initializeUI(self):
        """ Initialize UI elements """
        # General buttons
        self.ui.choose_file_button.clicked.connect(self.choose_db_mat)
        self.ui.import_button.clicked.connect(self.import_db_mat)
        self.ui.preview_button.clicked.connect(self.preview_db_mat)
        # Add validators to line edits
        self.ui.ignore_lines_edit.setValidator(
            QtGui.QIntValidator(0, 1000, self))
        # Dont accept numbers in the other line edit
        other_regex = QRegExp("[^0-9]+")
        self.ui.other_edit.setValidator(QRegExpValidator(other_regex))
        # Buttons that should update the table view preview
        self.ui.decimal_group.buttonClicked.connect(
            lambda btn: self.update_decimal(btn))
        self.ui.delimiter_group.buttonClicked.connect(
            lambda btn: self.update_delimiter(btn))
        self.ui.other_edit.textChanged.connect(self.update_other_label)
        self.ui.ignore_lines_edit.textChanged.connect(self.update_skiprows)
        self.ui.unit_combobox.currentTextChanged.connect(self.update_preview)

    """ Update each button group value in self.import_args """

    def update_skiprows(self):
        logging.debug(f"Update skiprows: {self.import_args['skiprows']=}")
        number = self.ui.ignore_lines_edit.text()
        if number == '':
            return
        self.import_args["skiprows"] = int(number)
        self.update_preview()

    def update_decimal(self, btn):
        logging.debug(f"Updated decimal:{btn.text()} '{Decimal[btn.text()]}'")
        self.import_args["decimal"] = Decimal[btn.text()]
        self.update_preview()

    def update_other_label(self):
        other_text = self.ui.other_edit.text()
        if other_text == '':
            logging.debug("other text empty ignoring update")
            return
        self.import_args["sep"] = other_text
        self.update_preview()

    def update_delimiter(self, btn):
        logging.debug(f"Updated Sep: {btn.text()} '{Separator[btn.text()]}'")
        if btn.text() != "other":
            self.ui.other_edit.setEnabled(False)
            self.import_args["sep"] = Separator[btn.text()]
        else:
            # Action to update label from other_cb status
            logging.debug("Activating QLine Edit for other option")
            self.ui.other_edit.setEnabled(True)
            other_text = self.ui.other_edit.text()
            logging.debug(f"other text.... '{other_text}'")
            if other_text == '':
                logging.debug(f"No text in other... Aborting")
                return
            self.update_other_label()
        self.update_preview()

    def update_preview(self):
        """ Update self.data and the table view representation """
        if self.filepath is None:
            logging.debug("No filepath chosen.. Not Updating..")
            return
        try:
            self.data: pd.DataFrame = pd.read_csv(self.filepath,
                                                  **self.import_args)
            self.data.iloc[:, 0] *= Units[self.ui.unit_combobox.currentText()]
        except (ParserError, TypeError) as error:
            logging.error(f"Parse Error when importing data:\n{error=}")
            return
        self.import_model._data = self.data
        self.import_model.layoutChanged.emit()

    def choose_db_mat(self):
        """
        Open a AskFileDialog to choose a file with the material data and
        update the qtextlabel with the name of the file
        """
        filepath = QFileDialog.getOpenFileName(self, 'Open File')
        logging.debug(f"Chosen filepath: {filepath}")
        if filepath[0] == '':
            return
        self.filepath = filepath[0]
        filename = os.path.basename(filepath[0])
        self.update_preview()
        self.ui.chosen_file_label.setText(filename)

    def preview_db_mat(self):
        """
        Show a preview of the imported data with the interpolation
        """
        if self.data is None:
            QMessageBox.information(self, "Choose File",
                                    "Must select a file before previewing",
                                    QMessageBox.Ok, QMessageBox.Ok)
            return
        data = self.data.values
        try:
            lmb, n, k = data[:, 0], data[:, 1], data[:, 2]
        except IndexError as error:
            logging.error(f"{error=}")
            return
        interp_n = interp1d(lmb, n)
        interp_k = interp1d(lmb, k)
        # Plot the results
        # Preview chosen material
        self.preview_import = FigWidget("Preview New Material")
        self.preview_import_fig = PltFigure(self.preview_import.layout,
                                            "Wavelength (nm)",
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
        mat_name = self.ui.mat_name_edit.text()
        if self.data is None:
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
        data = self.data.values
        self.parent.database.add_content(mat_name, data)
        QMessageBox.information(self, "Import Successful",
                                "Material imported successfully",
                                QMessageBox.Ok, QMessageBox.Ok)
        # Update the database previews
        self.close()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.preview_import is not None:
            self.preview_import.close()
        self.parent.update_db_preview()
        self.parent.update_mat_comboboxes()
        return super().closeEvent(a0)


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
        self.db_import_window = ImportWindow(self)
        self.db_import_window.show()

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

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.db_import_window is not None:
            self.db_import_window.close()
        return super().closeEvent(a0)
