import logging
import os
from typing import Union, Any

from PyQt5 import QtGui
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, QRegExp, Qt, pyqtSignal
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget
from modules.fig_class import FigWidget, PltFigure
from pandas import DataFrame
import pandas as pd
from pandas.errors import ParserError
from scipy.interpolate import interp1d

from .smm_import_db_mat import Ui_ImportDB

Units = {"nm": 1, "um": 1e3, "mm": 1e6}
Decimal = {"Dot (.)": ".", "Comma (,)": ","}
Separator = {"comma (,)": ",", "space ( )": r"\s+", ";": ";", "other": " "}
""" Model to show a preview of the imported data """


class TableModel(QAbstractTableModel):
    """ Abstract model to implement for the imported data preview
    This is built to work with pandas DFs
    """
    def __init__(self, data, good_cols: int = 0) -> None:
        """ Init for Table Model
        Args:
            data: data to start the table with
            good_cols: column not to highlight red (if 0 just ignore)
        """
        super().__init__()
        self._data: DataFrame = data
        self._good_cols: int = good_cols

    def data(self, index: QModelIndex, role) -> Any:
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        if role == Qt.BackgroundRole and self._good_cols != 0:
            if index.column() >= self._good_cols:
                return QtGui.QColor("#FF6666")

    def rowCount(self, _) -> int:
        return self._data.shape[0]

    def columnCount(self, _) -> int:
        return self._data.shape[1]

    def headerData(self, section, orientation, role) -> Union[str, None]:
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if self._good_cols == 0:
                    return f"Column {section}"
                else:
                    return f"Column {section}" if section <= self._good_cols - 1 else "Ignore"
            if orientation == Qt.Vertical:
                return str(self._data.index[section])


class ImpPrevWindow(QWidget):

    # Emit signal to parent when import button is clicked
    imp_clicked = pyqtSignal(object, str)

    def __init__(self,
                 parent,
                 mode: str = "db",
                 filepath: Union[str, None] = None):
        """ General import window
        mode - mode for the window (db or imp)
        """
        self.parent = parent
        self.mode: str = mode
        super().__init__()
        self.ui = Ui_ImportDB()
        self.ui.setupUi(self)
        # Startup variables
        self.filepath: Union[str, None] = filepath or None
        self.data: DataFrame = DataFrame()
        self.preview_import = None
        self.import_args = {
            "sep": " ",
            "decimal": ".",
            "skiprows": 0,
            "header": None,
            "comment": "#"
        }
        # Things for preview table
        self.import_table = self.ui.tableView
        if self.mode == "db":
            self.import_model = TableModel(DataFrame(), 3)
        elif self.mode == "imp":
            self.import_model = TableModel(DataFrame(), 2)
        else:
            logging.critical(f"Unknown {self.mode}")
            raise Exception("Unknown self.mode")
        self.import_table.setModel(self.import_model)
        self.ui.unit_combobox.addItems(list(Units.keys()))
        self.initializeUI()
        self.update_preview()

    def initializeUI(self):
        """
        Initialize UI elements
        In this case depending on the type of window asked it connect
        different buttons
        """
        # General buttons
        if self.mode == "db":
            self.ui.choose_file_button.clicked.connect(self.db_choose)
            self.ui.preview_button.clicked.connect(self.db_prev)
        elif self.mode == "imp":
            self.ui.chosen_file_label.setText(os.path.basename(self.filepath))
            self.ui.choose_file_button.hide()
            self.ui.mat_name_edit.hide()
            self.ui.mat_name_label.hide()
            self.ui.preview_button.clicked.connect(self.imp_prev)
        else:
            logging.critical("Unknown {self.mode=}")
            raise Exception("Unknown error")
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
        self.ui.import_button.clicked.connect(self._imp_clicked)
        self.ui.other_edit.textChanged.connect(self.update_other_label)
        self.ui.ignore_lines_edit.textChanged.connect(self.update_skiprows)
        self.ui.unit_combobox.currentTextChanged.connect(self.update_preview)

    def _imp_clicked(self):
        """ Filter information before passing the signal to the parent window """
        if self.data is None:
            return
        if self.mode == "db" and self.data.shape[1] < 3:
            logging.info(f"Invalid Shape for Imported Data {self.data.shape=}")
            return
        elif self.mode == "imp" and self.data.shape[1] < 2:
            logging.info(f"Invalid Shape for Imported Data {self.data.shape=}")
            return
        mat_name = self.ui.mat_name_edit.text()
        if self.mode == "db" and mat_name == '':
            QMessageBox.information(
                self, "No material name",
                "Please select a material name before importing",
                QMessageBox.Ok, QMessageBox.Ok)
            return
        self.imp_clicked.emit(self.data, mat_name)

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
            self.data = pd.read_csv(self.filepath, **self.import_args)
            self.data.iloc[:, 0] *= Units[self.ui.unit_combobox.currentText()]
        except (ParserError, TypeError) as error:
            logging.error(f"Parse Error when importing data:\n{error=}")
            return
        self.import_model._data = self.data
        self.import_model.layoutChanged.emit()

    """ Functions for Comparison data import """

    def imp_prev(self):
        """
        Show a preview of the data
        """
        data = self.data.values
        # Plot the results
        # Preview chosen material
        self.preview_import = FigWidget("Preview Data")
        self.preview_import_fig = PltFigure(self.preview_import.layout,
                                            "Wavelength (nm)",
                                            "R/T/Abs",
                                            width=7)
        prev_plot = self.preview_import_fig.axes
        prev_plot.plot(data[:, 0], data[:, 1])
        self.preview_import.show()

    """ Functions for the Mat Import Behavior """

    def db_choose(self):
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

    def db_prev(self):
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

    """ Other Events """

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.preview_import is not None:
            self.preview_import.close()
        if self.mode == "db":
            self.parent.update_db_preview()
            self.parent.update_mat_comboboxes()
        return super().closeEvent(a0)
