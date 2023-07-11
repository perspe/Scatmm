from enum import Enum, auto
import logging
from typing import List

from PyQt5 import QtCore
from PyQt5.QtGui import QRegExpValidator, QCloseEvent
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QSizePolicy, QSpacerItem
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from modules.dispersion import (
    cauchy,
    cauchy_abs,
    const,
    new_amorphous_n,
    sellmeier_abs,
    tauc_lorentz_n,
)
from modules.fig_class import PltFigure
import numpy as np
import scipy.constants as scc

from .custom_widgets import CustomSlider
from .smm_formula_mat import Ui_Formula
from .imp_window import ImpPrevWindow, ImpFlag

# Alias to convert nm to ev and ev to nm
_nm_to_ev = (scc.h * scc.c) / (scc.e * 1e-9)
Units = {
    "Energy (eV)": ("Emin (eV)", "Emax (eV)"),
    "Wavelength (nm)": ("λmin (nm)", "λmax (nm)"),
}

# Dictionary with all the dispersion equations
methods: dict = {
    "Constant": const,
    "Tauc Lorentz": tauc_lorentz_n,
    "New Amorphous": new_amorphous_n,
    "Cauchy": cauchy,
    "Cauchy Absorbent": cauchy_abs,
    "Sellmeier Absorbent": sellmeier_abs,
}

# Dictionaries with default values (to keep track of internal changes)
val_tauc_lorentz = {"N peak": 1, "ε∞": 1.5, "Eg": 1.5}
[
    val_tauc_lorentz.update({f"E0{i+1}": 3, f"A{i+1}": 50, f"C{i+1}": 1})
    for i in range(11)
]
val_new_amorphous = {"N peak": 1, "n∞": 1.5, "ωg": 1.5}
[
    val_new_amorphous.update({f"f{i}": 0.2, f"Γ{i}": 1.3, f"ω{i}": 2.5})
    for i in range(11)
]
val_const = {"n": 1.5, "k": 0.5}
val_cauchy = {"A": 1, "B": 2, "C": 3}
val_cauchy_abs = {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1}
val_sellmeier_abs = {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1}

func_vals = {
    "Constant": val_const,
    "Cauchy": val_cauchy,
    "Cauchy Absorbent": val_cauchy_abs,
    "Sellmeier Absorbent": val_sellmeier_abs,
    "Tauc Lorentz": val_tauc_lorentz,
    "New Amorphous": val_new_amorphous,
}


class Observable(Enum):
    """Enum for the Observable variables"""

    Ei = auto()
    Er = auto()
    N = auto()
    K = auto()
    ALPHA = auto()


def convert_observable(lmb, n, k, wanted: Observable):
    """Convert n/k to wanted observable"""
    if wanted == Observable.Ei:
        return np.imag((n + 1j * k) ** 2)
    elif wanted == Observable.Er:
        return np.real((n + 1j * k) ** 2)
    elif wanted == Observable.N:
        return n
    elif wanted == Observable.K:
        return k
    elif wanted == Observable.ALPHA:
        return 2 * np.pi * k / (lmb * 1e-7)
    return n, k


class FormulaWindow(QMainWindow):
    def __init__(self, parent: QtCore.QObject) -> None:
        self.parent = parent
        super(FormulaWindow, self).__init__()
        logging.debug("Opened Formula Window")
        self.ui = Ui_Formula()
        self.ui.setupUi(self)
        # Internal variables of interest
        # Connect Combobox names to functions
        self.import_window = None
        self.observables = {
            "n": lambda x, y, z: convert_observable(x, y, z, wanted=Observable.N),
            "k": lambda x, y, z: convert_observable(x, y, z, wanted=Observable.K),
            "εr": lambda x, y, z: convert_observable(x, y, z, wanted=Observable.Er),
            "εi": lambda x, y, z: convert_observable(x, y, z, wanted=Observable.Ei),
            "α (1/cm)": lambda x, y, z: convert_observable(
                x, y, z, wanted=Observable.ALPHA
            ),
        }
        self.methods: dict = {
            "Constant": self.MConst,
            "Tauc Lorentz": self.MTaucLorentz,
            "New Amorphous": self.MNewAmorphous,
            "Cauchy": self.MCauchy,
            "Cauchy Absorbent": self.MCauchyAbs,
            "Sellmeier Absorbent": self.MSellmeierAbs,
        }
        self._curr_method: str = "Constant"
        self.slider_list: List[CustomSlider] = []
        # Fill comboboxes
        self.ui.units_cb.addItems([str(unit) for unit in Units.keys()])
        self.ui.method_cb.addItems([key for key in self.methods.keys()])
        self.ui.left_axis_cb.addItems([key for key in self.observables.keys()])
        self.ui.right_axis_cb.addItems([key for key in self.observables.keys()])
        self.ui.left_axis_cb.setCurrentText("n")
        self.ui.right_axis_cb.setCurrentText("k")
        # Define the xdata, xlims and the xvariable
        self._xmin: float = 0.5
        self._xmax: float = 6.5
        self.ui.xmin_value.setText(str(self._xmin))
        self.ui.xmax_value.setText(str(self._xmax))
        self._xres: int = 500
        self._e = np.linspace(self._xmin, self._xmax, self._xres, dtype=np.float64)
        self._lmb = _nm_to_ev * (1 / self._e)
        # Create the figure to plot the data (with 2 y axis for n and k)
        self.plot_canvas = PltFigure(
            self.ui.plot_layout, self.ui.units_cb.currentText(), "n"
        )
        self.addToolBar(
            QtCore.Qt.TopToolBarArea, NavigationToolbar2QT(self.plot_canvas, self)
        )
        self.left_plot = self.plot_canvas.axes
        self.left_plot.spines["left"].set_color("b")
        self.left_plot.spines["right"].set_color("r")
        self.left_plot.tick_params(axis="y", colors="b")
        self.left_plot.set_ylabel("n", color="b")
        self.right_plot = self.left_plot.twinx()
        self.right_plot.set_ylabel("k", color="r")
        self.right_plot.spines["left"].set_color("b")
        self.right_plot.spines["right"].set_color("r")
        self.right_plot.tick_params(axis="y", colors="r")
        # InitializeUI and update plot info
        self.initializeUI()
        # Make the first plot iteration
        self.methods[self.ui.method_cb.currentText()]()
        self._update_nk()
        nplot = self.left_plot.plot(self._e, self._left, c="b")
        kplot = self.right_plot.plot(self._e, self._right, c="r")
        self._left_plot = nplot[0]
        self._right_plot = kplot[0]
        self._rebuild_plot()

    def initializeUI(self) -> None:
        """Connect functions to buttons"""
        # Add validators to QLineEdits
        double_regex = QtCore.QRegExp("[0-9]+\\.?[0-9]*j?")
        self.ui.xmin_value.setValidator(QRegExpValidator(double_regex))
        self.ui.xmax_value.setValidator(QRegExpValidator(double_regex))
        filename_regex = QtCore.QRegExp("[^/:{}\[\]'\"]*")
        self.ui.mat_name.setValidator(QRegExpValidator(filename_regex))
        # Connect buttons and signals to functions
        self.ui.method_cb.currentIndexChanged.connect(self.update_method)
        self.ui.add_db_button.clicked.connect(self.add_database)
        self.ui.import_button.clicked.connect(self.import_data)
        self.ui.units_cb.currentIndexChanged.connect(self.update_xvar)
        self.ui.xmin_value.editingFinished.connect(self.update_xlim)
        self.ui.xmax_value.editingFinished.connect(self.update_xlim)
        self.ui.left_axis_cb.currentIndexChanged.connect(self.update_left_axis)
        self.ui.right_axis_cb.currentIndexChanged.connect(self.update_right_axis)

    """ Update the n/k info and the plot """

    def update_method(self) -> None:
        """Update the formula"""
        new_method: str = self.ui.method_cb.currentText()
        logging.debug(f"Method Updated to: {new_method}")
        self._save_changed_values()
        self.methods[new_method]()
        self._update_nk()
        self._rebuild_plot()
        self._curr_method = new_method

    def _save_changed_values(self) -> None:
        """Save the current chosen values before updating the method"""
        val_dict = func_vals[self._curr_method]
        logging.debug(self.slider_list)
        for slider in self.slider_list:
            val_dict[slider.name] = slider.curr_value()
        logging.debug(val_dict)

    def _update_nk(self) -> None:
        """
        Update the calculated n/k for the specified method
        Update the actual parameter to plot from the comboboxes atop the plot
        self._n/self._k: Global variables that store the calculated n and k
        self._left/self._right: Variables with the data to plot, after using the cb variable
        """
        curr_method: str = self.ui.method_cb.currentText()
        curr_values = tuple([slider_i.curr_value() for slider_i in self.slider_list])
        # Pass the arguments to the calculating function
        self._n, self._k = methods[curr_method](self._e, *curr_values)
        left_var: str = self.ui.left_axis_cb.currentText()
        right_var: str = self.ui.right_axis_cb.currentText()
        # Pass the arguments to the convertion function
        self._left = self.observables[left_var](self._lmb, self._n, self._k)
        self._right = self.observables[right_var](self._lmb, self._n, self._k)

    def _update_plot(self) -> None:
        """Calculate n/k from variables in widget and update plot data"""
        self._update_nk()
        self._left_plot.set_ydata(self._left)
        self._right_plot.set_ydata(self._right)
        self.plot_canvas.draw()

    def _rebuild_plot(self) -> None:
        """Rebuild plot after xvar or xlims is changed"""
        logging.debug(self.ui.units_cb.currentText())
        self.left_plot.set_xlabel(self.ui.units_cb.currentText())
        if self.ui.units_cb.currentText() == list(Units.keys())[0]:
            self._left_plot.set_xdata(self._e)
            self._left_plot.set_ydata(self._left)
            self._right_plot.set_xdata(self._e)
            self._right_plot.set_ydata(self._right)
        else:
            self._left_plot.set_xdata(self._lmb)
            self._left_plot.set_ydata(self._left)
            self._right_plot.set_xdata(self._lmb)
            self._right_plot.set_ydata(self._left)
        # Update ylimits to accompain data change
        self.left_plot.set_ylim(
            np.min(self._left) - np.min(self._left) * 0.2,
            np.max(self._left) + np.max(self._left) * 0.2,
        )
        self.right_plot.set_ylim(
            np.min(self._right) - np.min(self._right) * 0.2,
            np.max(self._right) + np.max(self._right) * 0.2,
        )
        self.left_plot.set_xlim(self._xmin, self._xmax)
        self.plot_canvas.draw()

    def update_left_axis(self):
        """Update the left axis variable and recreate the plot"""
        new_var = self.ui.left_axis_cb.currentText()
        self._left = self.observables[new_var](self._lmb, self._n, self._k)
        self.left_plot.set_ylabel(new_var)
        self._rebuild_plot()

    def update_right_axis(self):
        """Update the right axis variable and recreate the plot"""
        new_var = self.ui.right_axis_cb.currentText()
        self._right = self.observables[new_var](self._lmb, self._n, self._k)
        self.right_plot.set_ylabel(new_var)
        self._rebuild_plot()

    """ Window buttons functions """

    def add_database(self) -> None:
        if self.parent is None:
            logging.critical("No parent to add data to database " "")
            raise Exception("Critical Error")
        mat_name = self.ui.mat_name.text()
        logging.debug(f"{mat_name=}")
        if mat_name == "":
            QMessageBox.information(
                self,
                "No material name",
                "Please select a material name before importing",
                QMessageBox.Ok,
                QMessageBox.Ok,
            )
            return
        data = np.c_[self._lmb, self._n, self._k]
        override = QMessageBox.NoButton
        if mat_name in self.parent.database.content:
            override = QMessageBox.question(
                self,
                "Material name in Database",
                "The Database already has a material with this name.."
                + "\nOverride the material?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
        if override == QMessageBox.No:
            return
        elif override == QMessageBox.Yes:
            self.parent.database.rmv_content(mat_name)
            self.parent.database.add_content(mat_name, data)
        else:
            self.parent.database.add_content(mat_name, data)

        QMessageBox.information(
            self,
            "Success",
            "Material added Successfully to Database",
            QMessageBox.Ok,
            QMessageBox.Ok,
        )
        self.parent.update_db_preview()
        self.parent.db_updated.emit()

    def update_xvar(self) -> None:
        """Update the overall representation of xvar, between E and Wvl"""
        new_var: str = self.ui.units_cb.currentText()
        min_label, max_label = Units[new_var]
        logging.debug(f"{new_var=}::{min_label=}::{max_label=}")
        # Update label text
        self.ui.xmin_label.setText(min_label)
        self.ui.xmax_label.setText(max_label)
        limit_1, limit_2 = _nm_to_ev * (1 / self._xmin), _nm_to_ev * (1 / self._xmax)
        # Update QLineEdit Limits
        self._xmin = min([limit_1, limit_2])
        self._xmax = max([limit_1, limit_2])
        logging.debug(f"{self._xmin=}::{self._xmax=}")
        self.ui.xmin_value.setText(f"{self._xmin:.1f}")
        self.ui.xmax_value.setText(f"{self._xmax:.1f}")
        # Rebuild plot
        self._rebuild_plot()

    def update_xlim(self) -> None:
        """Update the xlims for calculation"""
        curr_var: str = self.ui.units_cb.currentText()
        xmin: float = float(self.ui.xmin_value.text())
        xmax: float = float(self.ui.xmax_value.text())
        if curr_var == list(Units.keys())[0]:
            # For energy
            logging.debug(f"Update Energy: {curr_var=}::{xmin=}::{xmax=}")
            self._e = np.linspace(xmin, xmax, self._xres)
            self._lmb = _nm_to_ev * (1 / self._e)
            self._xmin, self._xmax = np.min(self._e), np.max(self._e)
        else:
            logging.debug(f"Update Wavelength: {curr_var=}::{xmin=}::{xmax=}")
            self._lmb = np.linspace(xmin, xmax, self._xres)
            self._e = _nm_to_ev * (1 / self._lmb)
            self._xmin, self._xmax = np.min(self._lmb), np.max(self._lmb)
        self._update_nk()
        self._rebuild_plot()

    """ Import Materials to preview/compare """

    def import_data(self):
        self.import_window = ImpPrevWindow(
            self, ImpFlag.NONAME | ImpFlag.DB | ImpFlag.BUTTON
        )
        self.import_window.imp_clicked.connect(self._import)
        self.import_window.show()

    QtCore.pyqtSlot(object, str)

    def _import(self, imported_data, name):
        """Plot the imported data"""
        if self.import_window is None:
            logging.critical("Unknown Error...")
            return
        data = imported_data.values
        self.left_plot.plot(data[:, 0], data[:, 1], "--b")
        self.right_plot.plot(data[:, 0], data[:, 2], "--r")
        self.plot_canvas.draw()
        self.import_window.close()

    """ Functions to Build the Variables for Different Methods """

    def _clear_variable_layout(self) -> None:
        """Function to clear all the widgets in the layout"""
        while self.ui.variable_layout.count():
            child = self.ui.variable_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _update_layout(self, layout) -> None:
        """Update the slider layout"""
        for slider in self.slider_list:
            layout.addWidget(slider)
            slider.changed.connect(self._update_plot)
        verticalSpacer = QSpacerItem(20, 80, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(verticalSpacer)

    def MConst(self):
        """Create the variables for the const method
        Vars: n, k
        """
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        n = CustomSlider("n", val_const["n"], 1, 3)
        k = CustomSlider("k", val_const["k"], 0, 1)
        self.slider_list = [n, k]
        self._update_layout(layout)
        self._update_plot()

    def _update_tl_peaks(self) -> None:
        """Update the number of custom Sliders for the peaks"""
        # Save previous values before updating
        self._save_changed_values()
        n_peaks: int = int(self.slider_list[0].curr_value())
        val_tauc_lorentz["N peak"] = n_peaks
        self._clear_variable_layout()
        # Rebuild all widgets
        n_peak = CustomSlider(
            "N peak", n_peaks, 1, 11, resolution=100, fixed_lim=True, int_change=True
        )
        self.ui.variable_layout.addWidget(n_peak)
        n_peak.changed.connect(self._update_tl_peaks)
        einf = CustomSlider("ε∞", val_tauc_lorentz["ε∞"], 1, 5)
        eg = CustomSlider("Eg", val_tauc_lorentz["Eg"], 0.5, 10)
        self.slider_list = [einf, eg]
        for i in range(n_peaks):
            e0 = CustomSlider(f"E0{i+1}", val_tauc_lorentz[f"E0{i+1}"], 0.5, 10)
            a = CustomSlider(f"A{i+1}", val_tauc_lorentz[f"A{i+1}"], 0.1, 200)
            c = CustomSlider(f"C{i+1}", val_tauc_lorentz[f"C{i+1}"], 0.1, 10)
            slider_peak = [e0, a, c]
            self.slider_list.extend(slider_peak)
        self._update_layout(self.ui.variable_layout)
        self.slider_list.insert(0, n_peak)
        self._rebuild_plot()

    def MTaucLorentz(self) -> None:
        """Update the number of custom Sliders for the peaks"""
        self._clear_variable_layout()
        n_peaks = int(val_tauc_lorentz["N peak"])
        # Rebuild all widgets
        n_peak = CustomSlider(
            "N peak", n_peaks, 1, 11, resolution=100, fixed_lim=True, int_change=True
        )
        self.ui.variable_layout.addWidget(n_peak)
        n_peak.changed.connect(self._update_tl_peaks)
        einf = CustomSlider("ε∞", val_tauc_lorentz["ε∞"], 1, 5)
        eg = CustomSlider("Eg", val_tauc_lorentz["Eg"], 0.5, 10)
        self.slider_list = [einf, eg]
        for i in range(n_peaks):
            e0 = CustomSlider(f"E0{i+1}", val_tauc_lorentz[f"E0{i+1}"], 0.5, 10)
            a = CustomSlider(f"A{i+1}", val_tauc_lorentz[f"A{i+1}"], 0.1, 200)
            c = CustomSlider(f"C{i+1}", val_tauc_lorentz[f"C{i+1}"], 0.1, 10)
            slider_peak = [e0, a, c]
            self.slider_list.extend(slider_peak)
        self._update_layout(self.ui.variable_layout)
        self.slider_list.insert(0, n_peak)
        self._update_plot()

    def _update_na_peaks(self) -> None:
        """Update the number of custom Sliders for the peaks"""
        self._save_changed_values()
        n_peaks: int = int(self.slider_list[0].curr_value())
        logging.debug(f"Update number of peaks to: {n_peaks}")
        val_new_amorphous["N peak"] = n_peaks
        self._clear_variable_layout()
        # Rebuild all widgets
        n_peak = CustomSlider(
            "N peak", n_peaks, 1, 11, resolution=100, fixed_lim=True, int_change=True
        )
        self.ui.variable_layout.addWidget(n_peak)
        n_peak.changed.connect(self._update_na_peaks)
        ninf = CustomSlider("n∞", val_new_amorphous["n∞"], 1, 5)
        wg = CustomSlider("ωg", val_new_amorphous["ωg"], 0.5, 10)
        self.slider_list = [ninf, wg]
        for i in range(n_peaks):
            fj = CustomSlider(f"f{i}", val_new_amorphous[f"f{i}"], 0, 1)
            gammaj = CustomSlider(f"Γ{i}", val_new_amorphous[f"Γ{i}"], 0.2, 8)
            wj = CustomSlider(f"ω{i}", val_new_amorphous[f"ω{i}"], 1.5, 10)
            slider_peak = [fj, gammaj, wj]
            self.slider_list.extend(slider_peak)
        self._update_layout(self.ui.variable_layout)
        self.slider_list.insert(0, n_peak)
        self._rebuild_plot()

    def MNewAmorphous(self) -> None:
        self._clear_variable_layout()
        n_peaks = int(val_new_amorphous["N peak"])
        n_peak = CustomSlider(
            "N peak", n_peaks, 1, 11, resolution=100, fixed_lim=True, int_change=True
        )
        n_peak.changed.connect(self._update_na_peaks)
        self.ui.variable_layout.addWidget(n_peak)
        ninf = CustomSlider("n∞", val_new_amorphous["n∞"], 1, 5)
        wg = CustomSlider("ωg", val_new_amorphous["ωg"], 0.5, 10)
        self.slider_list = [ninf, wg]
        for i in range(n_peaks):
            fj = CustomSlider(f"f{i}", val_new_amorphous[f"f{i}"], 0, 1)
            gammaj = CustomSlider(f"Γ{i}", val_new_amorphous[f"Γ{i}"], 0.2, 8)
            wj = CustomSlider(f"ω{i}", val_new_amorphous[f"ω{i}"], 1.5, 10)
            slider_peak = [fj, gammaj, wj]
            self.slider_list.extend(slider_peak)
        self._update_layout(self.ui.variable_layout)
        self.slider_list.insert(0, n_peak)
        self._update_plot()

    def MCauchy(self) -> None:
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        a = CustomSlider("A", val_cauchy["A"], 1, 5)
        b = CustomSlider("B", val_cauchy["B"], 0.5, 10)
        c = CustomSlider("C", val_cauchy["C"], 0.5, 10)
        self.slider_list = [a, b, c]
        self._update_layout(layout)

    def MCauchyAbs(self) -> None:
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        a = CustomSlider("A", val_cauchy_abs["A"], 1, 5)
        b = CustomSlider("B", val_cauchy_abs["B"], 0.5, 10)
        c = CustomSlider("C", val_cauchy_abs["C"], 0.5, 10)
        d = CustomSlider("D", val_cauchy_abs["D"], 0.5, 10)
        e = CustomSlider("E", val_cauchy_abs["E"], 0.5, 10)
        f = CustomSlider("F", val_cauchy_abs["F"], 0.5, 10)
        self.slider_list = [a, b, c, d, e, f]
        self._update_layout(layout)

    def MSellmeierAbs(self) -> None:
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        a = CustomSlider("A", val_sellmeier_abs["A"], 1, 5)
        b = CustomSlider("B", val_sellmeier_abs["B"], 0.5, 10)
        c = CustomSlider("C", val_sellmeier_abs["C"], 0.5, 10)
        d = CustomSlider("D", val_sellmeier_abs["D"], 0.5, 10)
        e = CustomSlider("E", val_sellmeier_abs["E"], 0.5, 10)
        self.slider_list = [a, b, c, d, e]
        self._update_layout(layout)

    """ QWidgets Methods """
    def closeEvent(self, a0: QCloseEvent) -> None:
        self.parent.formula_window = None
        return super().closeEvent(a0)
