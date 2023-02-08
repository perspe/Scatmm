from enum import Flag, auto
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


class ImpFlag(Flag):
    DB = auto()  # Flag to indicate 2-y plot
    DATA = auto()  # Flag to indicate 1-y plot
    BUTTON = auto()  # Flag to indicate import button was clicked
    NONAME = auto()  # Flag to indicate that name is necessary
    _DRAG = auto()
    DRAG = _DRAG | NONAME  # Flag to indicate data from drag and drop


""" Model to show a preview of the imported data """


class TableModel(QAbstractTableModel):
    """Abstract model to implement for the imported data preview
    This is built to work with pandas DFs
    """

    def __init__(self, data, good_cols: int = 0) -> None:
        """Init for Table Model
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
                    return (
                        f"Column {section}"
                        if section <= self._good_cols - 1
                        else "Ignore"
                    )
            if orientation == Qt.Vertical:
                return str(self._data.index[section])


class ImpPrevWindow(QWidget):

    # Emit signal to parent when import button is clicked
    imp_clicked = pyqtSignal(object, str)

    def __init__(
        self,
        parent,
        imp_flag: ImpFlag = ImpFlag.DB | ImpFlag.BUTTON,
        filepath: Union[str, None] = None,
    ):
        """General import window
        imp_flags: Flags that indicate how to create the window
        """
        self.parent = parent
        self.imp_flag: ImpFlag = imp_flag
        super().__init__()
        self.ui = Ui_ImportDB()
        self.ui.setupUi(self)
        # Startup variables
        self.filepath: Union[str, None] = filepath or None
        if ImpFlag.DRAG in self.imp_flag and self.filepath is None:
            logging.critical("Import Window initialized improperly")
            self.close()
        self.data: DataFrame = DataFrame()
        self.preview_import = None
        self.import_args = {
            "sep": "\s+",
            "decimal": ".",
            "skiprows": 0,
            "header": None,
            "comment": "#",
            "usecols": None,
            "skipfooter": 0,
        }
        # Things for preview table
        self.import_table = self.ui.tableView
        if ImpFlag.DB in self.imp_flag:
            self.import_model = TableModel(DataFrame(), 3)
        elif ImpFlag.DATA in self.imp_flag:
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
        # Create the interface based on the provided flags
        if ImpFlag.BUTTON in self.imp_flag:
            self.ui.choose_file_button.clicked.connect(self.choose_file)
        # Connect to preview function
        if ImpFlag.DB in self.imp_flag:
            self.ui.preview_button.clicked.connect(self.db_prev)
            filename_regex = QRegExp("[^/:{}\[\]'\"]*")
            self.ui.mat_name_edit.setValidator(QRegExpValidator(filename_regex))
        elif ImpFlag.DATA in self.imp_flag:
            self.ui.preview_button.clicked.connect(self.data_prev)
        if ImpFlag.DRAG in self.imp_flag:
            self.ui.chosen_file_label.setText(os.path.basename(self.filepath))
            self.ui.choose_file_button.hide()
        if ImpFlag.NONAME in self.imp_flag:
            self.ui.mat_name_edit.hide()
            self.ui.mat_name_label.hide()
        # Add validators to line edits
        self.ui.ignore_lines_top_edit.setValidator(QtGui.QIntValidator(0, 1000, self))
        self.ui.ignore_lines_bottom_edit.setValidator(
            QtGui.QIntValidator(0, 1000, self)
        )
        ignore_cols_regex = QRegExp("^[0-9, ]*")
        self.ui.choose_columns_edit.setValidator(QRegExpValidator(ignore_cols_regex))
        # Don't accept numbers in the other line edit
        other_regex = QRegExp("^[^?][^0-9]*")
        self.ui.other_edit.setValidator(QRegExpValidator(other_regex))
        # Buttons that should update the table view preview
        self.ui.decimal_group.buttonClicked.connect(
            lambda btn: self.update_decimal(btn)
        )
        self.ui.delimiter_group.buttonClicked.connect(
            lambda btn: self.update_delimiter(btn)
        )
        self.ui.import_button.clicked.connect(self._imp_clicked)
        self.ui.other_edit.textChanged.connect(self.update_other_label)
        self.ui.ignore_lines_top_edit.editingFinished.connect(self.update_skiprows)
        self.ui.ignore_lines_bottom_edit.editingFinished.connect(self.update_skipfooter)
        self.ui.choose_columns_edit.editingFinished.connect(self.update_ignore_cols)
        self.ui.unit_combobox.currentTextChanged.connect(self.update_preview)

    def _imp_clicked(self):
        """Filter information before passing the signal to the parent window"""
        if self.data is None:
            return
        if ImpFlag.DB in self.imp_flag and self.data.shape[1] < 3:
            logging.info(f"Invalid Shape for Imported Data {self.data.shape=}")
            return
        elif ImpFlag.DATA in self.imp_flag and self.data.shape[1] < 2:
            logging.info(f"Invalid Shape for Imported Data {self.data.shape=}")
            return
        mat_name = self.ui.mat_name_edit.text()
        if ~ImpFlag.NONAME in self.imp_flag and mat_name == "":
            QMessageBox.information(
                self,
                "No material name",
                "Please select a material name before importing",
                QMessageBox.Ok,
                QMessageBox.Ok,
            )
            return
        self.imp_clicked.emit(self.data, mat_name)

    """ Update each button group value in self.import_args """

    def update_skiprows(self):
        logging.debug(f"Update skiprows: {self.import_args['skiprows']=}")
        number = self.ui.ignore_lines_top_edit.text().strip()
        if number == "":
            self.ui.ignore_lines_top_edit.setText(str(self.import_args["skiprows"]))
            self.update_preview()
            return
        self.import_args["skiprows"] = int(number)
        self.update_preview()

    def update_skipfooter(self):
        logging.debug(f"Update skipfooter: {self.import_args['skipfooter']=}")
        number = self.ui.ignore_lines_bottom_edit.text().strip()
        if number == "":
            self.ui.ignore_lines_bottom_edit.setText(
                str(self.import_args["skipfooter"])
            )
            self.update_preview()
            return
        self.import_args["skipfooter"] = int(number)
        self.update_preview()

    def update_ignore_cols(self):
        logging.debug(f"Updating Ignore Cols: {self.import_args['usecols']=}")
        pattern: str = self.ui.choose_columns_edit.text().strip(" ,")
        if pattern == "":
            self.import_args["usecols"] = None
            self.update_preview()
            return
        ignore_list = list([int(split_i) - 1 for split_i in pattern.split(",")])
        logging.debug(f"{ignore_list=}")
        self.import_args["usecols"] = ignore_list
        self.update_preview()

    def update_decimal(self, btn):
        logging.debug(f"Updated decimal:{btn.text()} '{Decimal[btn.text()]}'")
        self.import_args["decimal"] = Decimal[btn.text()]
        self.update_preview()

    def update_other_label(self):
        other_text = self.ui.other_edit.text()
        if other_text == "":
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
            if other_text == "":
                logging.debug(f"No text in other... Aborting")
                return
            self.update_other_label()
        self.update_preview()

    def update_preview(self):
        """Update self.data and the table view representation"""
        if self.filepath is None:
            logging.debug("No filepath chosen.. Not Updating..")
            return
        try:
            self.data = pd.read_csv(self.filepath, **self.import_args)
            self.data.iloc[:, 0] *= Units[self.ui.unit_combobox.currentText()]
        except (ParserError, TypeError, ValueError) as error:
            logging.error(f"Parse Error when importing data:\n{error=}")
            return
        self.import_model._data = self.data
        self.import_model.layoutChanged.emit()

    """ Choose File Button """

    def choose_file(self):
        """
        Open a AskFileDialog to choose a file with the material data and
        update the qtextlabel with the name of the file
        """
        filepath = QFileDialog.getOpenFileName(self, "Open File")
        logging.debug(f"Chosen filepath: {filepath}")
        if filepath[0] == "":
            return
        self.filepath = filepath[0]
        filename = os.path.basename(filepath[0])
        self.update_preview()
        self.ui.chosen_file_label.setText(filename)

    """ Functions to preview data """

    def data_prev(self) -> None:
        """
        Show a preview of the data
        """
        data = self.data.values
        if self.data is None:
            return
        if ImpFlag.DB in self.imp_flag and self.data.shape[1] < 3:
            logging.info(f"Invalid Shape to preview data {self.data.shape=}")
            return
        elif ImpFlag.DATA in self.imp_flag and self.data.shape[1] < 2:
            logging.info(f"Invalid Shape to preview data {self.data.shape=}")
            return
        # Plot the results
        # Preview chosen material
        self.preview_import = FigWidget("Preview Data")
        self.preview_import_fig = PltFigure(
            self.preview_import.layout, "Wavelength (nm)", "R/T/Abs", width=7
        )
        prev_plot = self.preview_import_fig.axes
        prev_plot.plot(data[:, 0], data[:, 1])
        self.preview_import.show()

    def db_prev(self):
        """
        Show a preview of the imported data with the interpolation
        """
        if self.data is None:
            QMessageBox.information(
                self,
                "Choose File",
                "Must select a file before previewing",
                QMessageBox.Ok,
                QMessageBox.Ok,
            )
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
        self.preview_import_fig = PltFigure(
            self.preview_import.layout, "Wavelength (nm)", "n/k", width=7
        )
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
        self.parent.import_window=None
        return super().closeEvent(a0)
