import logging
from typing import List, Union

from PyQt5 import QtCore
from PyQt5.QtGui import QRegExpValidator
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

# Dictionary with all the dispersion equations
methods: dict = {
    "Constant": const,
    "Tauc Lorentz": tauc_lorentz_n,
    "New Amorphous": new_amorphous_n,
    "Cauchy": cauchy,
    "Cauchy Absorbent": cauchy_abs,
    "Sellmeier Absorbent": sellmeier_abs
}

# Alias to convert nm to ev and ev to nm
_nm_to_ev = (scc.h * scc.c) / (scc.e * 1e-9)
Units = {
    "Energy (eV)": ("Emin (eV)", "Emax (eV)"),
    "Wavelength (nm)": ("λmin (nm)", "λmax (nm)")
}


class FormulaWindow(QMainWindow):
    def __init__(self, parent: Union[QtCore.QObject, None] = None) -> None:
        self.parent: Union[QtCore.QObject, None] = parent
        super(FormulaWindow, self).__init__()
        logging.debug("Opened Formula Window")
        self.ui = Ui_Formula()
        self.ui.setupUi(self)
        # Internal variables of interest
        # Connect Combobox names to functions
        self.methods: dict = {
            "Constant": self.MConst,
            "Tauc Lorentz": self.MTaucLorentz,
            "New Amorphous": self.MNewAmorphous,
            "Cauchy": self.MCauchy,
            "Cauchy Absorbent": self.MCauchyAbs,
            "Sellmeier Absorbent": self.MSellmeierAbs
        }
        self.slider_list: List[CustomSlider] = []
        # Fill comboboxes
        self.ui.units_cb.addItems([str(unit) for unit in Units.keys()])
        self.ui.method_cb.addItems([key for key in self.methods.keys()])
        # Define the xdata, xlims and the xvariable
        self._xmin: float = 0.5
        self._xmax: float = 6.5
        self.ui.xmin_value.setText(str(self._xmin))
        self.ui.xmax_value.setText(str(self._xmax))
        self._xres: int = 500
        self._e = np.linspace(self._xmin,
                              self._xmax,
                              self._xres,
                              dtype=np.float64)
        self._lmb = _nm_to_ev * (1 / self._e)
        # Create the figure to plot the data (with 2 y axis for n and k)
        self.plot_canvas = PltFigure(self.ui.plot_layout,
                                     self.ui.units_cb.currentText(), "n")
        self.addToolBar(QtCore.Qt.TopToolBarArea,
                        NavigationToolbar2QT(self.plot_canvas, self))
        self.n_plot = self.plot_canvas.axes
        self.n_plot.spines["left"].set_color("b")
        self.n_plot.spines["right"].set_color("r")
        self.n_plot.tick_params(axis='y', colors='b')
        self.n_plot.set_ylabel("n", color='b')
        self.n_plot.set_ylim((0, 5))
        self.k_plot = self.n_plot.twinx()
        self.k_plot.set_ylabel("k", color="r")
        self.k_plot.set_ylim((0, 5))
        self.k_plot.spines["left"].set_color("b")
        self.k_plot.spines["right"].set_color("r")
        self.k_plot.tick_params(axis='y', colors='r')
        # InitializeUI and update plot info
        self.initializeUI()
        # Make the first plot iteration
        self.methods[self.ui.method_cb.currentText()]()
        self._update_nk()
        nplot = self.n_plot.plot(self._e, self._n, c='b')
        kplot = self.k_plot.plot(self._e, self._k, c='r')
        self._n_plot = nplot[0]
        self._k_plot = kplot[0]

    def initializeUI(self) -> None:
        """ Connect functions to buttons """
        # Add validators to QLineEdits
        double_regex = QtCore.QRegExp("[0-9]+\\.?[0-9]*j?")
        self.ui.xmin_value.setValidator(QRegExpValidator(double_regex))
        self.ui.xmax_value.setValidator(QRegExpValidator(double_regex))
        # Connect buttons and signals to functions
        self.ui.method_cb.currentIndexChanged.connect(self.update_method)
        self.ui.add_db_button.clicked.connect(self.add_database)
        self.ui.units_cb.currentIndexChanged.connect(self.update_xvar)
        self.ui.xmin_value.editingFinished.connect(self.update_xlim)
        self.ui.xmax_value.editingFinished.connect(self.update_xlim)

    """ Update the n/k info and the plot """

    def update_method(self) -> None:
        """ Update the formula """
        new_method: str = self.ui.method_cb.currentText()
        logging.debug(f"Method Updated to: {new_method}")
        self.methods[new_method]()
        self._update_nk()
        self._rebuild_plot()

    def _update_nk(self) -> None:
        """ Get updated nk values for current given parameters """
        curr_method: str = self.ui.method_cb.currentText()
        curr_values = tuple(
            [slider_i.curr_value() for slider_i in self.slider_list])
        self._n, self._k = methods[curr_method](self._e, *curr_values)

    def _update_plot(self) -> None:
        """ Calculate n/k from variables in widget and plot """
        self._update_nk()
        self._n_plot.set_ydata(self._n)
        self._k_plot.set_ydata(self._k)
        self.plot_canvas.draw()

    def _rebuild_plot(self) -> None:
        """ Rebuild plot after xvar or xlims is changed """
        logging.debug(self.ui.units_cb.currentText())
        self.n_plot.set_xlabel(self.ui.units_cb.currentText())
        if self.ui.units_cb.currentText() == list(Units.keys())[0]:
            self._n_plot.set_xdata(self._e)
            self._n_plot.set_ydata(self._n)
            self._k_plot.set_xdata(self._e)
            self._k_plot.set_ydata(self._k)
        else:
            self._n_plot.set_xdata(self._lmb)
            self._n_plot.set_ydata(self._n)
            self._k_plot.set_xdata(self._lmb)
            self._k_plot.set_ydata(self._k)
        self.n_plot.set_xlim(self._xmin, self._xmax)
        self.plot_canvas.draw()

    """ Window buttons functions """

    def add_database(self) -> None:
        if self.parent is None:
            logging.critical("No parent to add data to database " "")
            raise Exception("Critical Error")
        mat_name = self.ui.mat_name.text()
        logging.debug(f"{mat_name=}")
        if mat_name == '':
            QMessageBox.information(
                self, "No material name",
                "Please select a material name before importing",
                QMessageBox.Ok, QMessageBox.Ok)
            return
        data = np.c_[self._lmb, self._n, self._k]
        override = QMessageBox.NoButton
        if mat_name in self.parent.database.content:
            override = QMessageBox.question(
                self, "Material name in Database",
                "The Database already has a material with this name.." +
                "\nOverride the material?", QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes)
        if override == QMessageBox.No:
            return
        elif override == QMessageBox.Yes:
            self.parent.database.rmv_content(mat_name)
            self.parent.database.add_content(mat_name, data)
        else:
            self.parent.database.add_content(mat_name, data)

        QMessageBox.information(self, "Success",
                                "Material added Successfully to Database",
                                QMessageBox.Ok, QMessageBox.Ok)
        self.parent.update_db_preview()
        self.parent.update_mat_comboboxes()

    def update_xvar(self) -> None:
        """ Update the overall representation of xvar, between E and Wvl """
        new_var: str = self.ui.units_cb.currentText()
        min_label, max_label = Units[new_var]
        logging.debug(f"{new_var=}::{min_label=}::{max_label=}")
        # Update label text
        self.ui.xmin_label.setText(min_label)
        self.ui.xmax_label.setText(max_label)
        limit_1, limit_2 = _nm_to_ev * (1 / self._xmin), _nm_to_ev * (
            1 / self._xmax)
        # Update QLineEdit Limits
        self._xmin = min([limit_1, limit_2])
        self._xmax = max([limit_1, limit_2])
        logging.debug(f"{self._xmin=}::{self._xmax=}")
        self.ui.xmin_value.setText(f"{self._xmin:.1f}")
        self.ui.xmax_value.setText(f"{self._xmax:.1f}")
        # Rebuild plot
        self._rebuild_plot()

    def update_xlim(self) -> None:
        """ Update the xlims for calculation """
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

    """ Functions to Build the Variables for Different Methods """

    def _clear_variable_layout(self) -> None:
        """ Function to clear all the widgets in the layout """
        while self.ui.variable_layout.count():
            child = self.ui.variable_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _update_layout(self, layout) -> None:
        """ Update the slider layout """
        for slider in self.slider_list:
            layout.addWidget(slider)
            slider.changed.connect(self._update_plot)
        verticalSpacer = QSpacerItem(20, 80, QSizePolicy.Minimum,
                                     QSizePolicy.Expanding)
        layout.addItem(verticalSpacer)

    def MConst(self):
        """ Create the variables for the const method
        Vars: n, k
        """
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        n = CustomSlider("n", 1.5, 1, 5)
        k = CustomSlider("k", 0.5, 0, 5)
        layout.addWidget(n)
        layout.addWidget(k)
        self.slider_list = [n, k]
        self._update_layout(layout)

    def _update_tl_peaks(self) -> None:
        """ Update the number of custom Sliders for the peaks """
        self._clear_variable_layout()
        n_peaks: int = int(self.slider_list[0].curr_value())
        logging.debug(f"Updating TL peaks... {n_peaks=}")
        # Rebuild all widgets
        n_peak = CustomSlider("N peak",
                              n_peaks,
                              1,
                              11,
                              resolution=100,
                              fixed_lim=True)
        self.ui.variable_layout.addWidget(n_peak)
        n_peak.changed.connect(self._update_tl_peaks)
        einf = CustomSlider("ε∞", 1.5, 1, 5)
        eg = CustomSlider("Eg", 1.5, 0.5, 10)
        self.slider_list = [einf, eg]
        for i in range(n_peaks):
            e0 = CustomSlider(f"E0{i+1}", 3, 0.5, 10)
            a = CustomSlider(f"A{i+1}", 50, 0.1, 200)
            c = CustomSlider(f"C{i+1}", 1, 0.1, 10)
            slider_peak = [e0, a, c]
            self.slider_list.extend(slider_peak)
        self._update_layout(self.ui.variable_layout)
        self.slider_list.insert(0, n_peak)
        logging.debug(f"Slider List... {self.slider_list}")
        self._update_plot()

    def MTaucLorentz(self) -> None:
        """ Create the variables for the const method
        Vars: einf, eg, e0, a, c
        """
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        n_peak = CustomSlider("N peak",
                              1,
                              1,
                              11,
                              resolution=100,
                              fixed_lim=True)
        n_peak.changed.connect(self._update_tl_peaks)
        layout.addWidget(n_peak)
        einf = CustomSlider("ε∞", 1.5, 1, 5)
        eg = CustomSlider("Eg", 1.5, 0.5, 10)
        e0 = CustomSlider("E01", 3, 0.5, 10)
        a = CustomSlider("A1", 50, 0.1, 200)
        c = CustomSlider("C1", 1, 0.1, 10)
        self.slider_list = [einf, eg, e0, a, c]
        self._update_layout(self.ui.variable_layout)
        self.slider_list.insert(0, n_peak)
        logging.debug(f"Slider List... {self.slider_list}")

    def _update_na_peaks(self) -> None:
        """ Update the number of custom Sliders for the peaks """
        self._clear_variable_layout()
        n_peaks: int = int(self.slider_list[0].curr_value())
        logging.debug(f"Updating NA peaks... {n_peaks=}")
        # Rebuild all widgets
        n_peak = CustomSlider("N peak",
                              n_peaks,
                              1,
                              11,
                              resolution=100,
                              fixed_lim=True)
        self.ui.variable_layout.addWidget(n_peak)
        n_peak.changed.connect(self._update_na_peaks)
        ninf = CustomSlider("n∞", 1.5, 1, 5)
        wg = CustomSlider("ωg", 1.5, 0.5, 10)
        self.slider_list = [ninf, wg]
        for i in range(n_peaks):
            fj = CustomSlider(f"f{i}", 0.3, 0, 1)
            gammaj = CustomSlider(f"Γ{i}", 1, 0.2, 8)
            wj = CustomSlider(f"ω{i}", 2, 1.5, 10)
            slider_peak = [fj, gammaj, wj]
            self.slider_list.extend(slider_peak)
        self._update_layout(self.ui.variable_layout)
        self.slider_list.insert(0, n_peak)
        logging.debug(f"Slider List... {self.slider_list}")
        self._update_plot()

    def MNewAmorphous(self) -> None:
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        n_peak = CustomSlider("N peak",
                              1,
                              1,
                              11,
                              resolution=100,
                              fixed_lim=True)
        n_peak.changed.connect(self._update_na_peaks)
        layout.addWidget(n_peak)
        ninf = CustomSlider("n∞", 1.5, 1, 5, 10)
        wg = CustomSlider("ωg", 1.5, 0.5, 10)
        fj = CustomSlider("fj", 0.3, 0, 1)
        gammaj = CustomSlider("Γj", 1, 0.2, 8)
        wj = CustomSlider("ωj", 2, 1.5, 10)
        self.slider_list = [ninf, wg, fj, gammaj, wj]
        self._update_layout(self.ui.variable_layout)
        self.slider_list.insert(0, n_peak)
        logging.debug(f"Slider List... {self.slider_list}")

    def MCauchy(self) -> None:
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        a = CustomSlider("A", 1, 1, 5)
        b = CustomSlider("B", 2, 0.5, 10)
        c = CustomSlider("C", 3, 0.5, 10)
        self.slider_list = [a, b, c]
        self._update_layout(layout)

    def MCauchyAbs(self) -> None:
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        a = CustomSlider("A", 1, 1, 5)
        b = CustomSlider("B", 2, 0.5, 10)
        c = CustomSlider("C", 3, 0.5, 10)
        d = CustomSlider("D", 3, 0.5, 10)
        e = CustomSlider("E", 3, 0.5, 10)
        f = CustomSlider("F", 3, 0.5, 10)
        self.slider_list = [a, b, c, d, e, f]
        self._update_layout(layout)

    def MSellmeierAbs(self) -> None:
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        a = CustomSlider("A", 1, 1, 5)
        b = CustomSlider("B", 2, 0.5, 10)
        c = CustomSlider("C", 3, 0.5, 10)
        d = CustomSlider("D", 3, 0.5, 10)
        e = CustomSlider("E", 3, 0.5, 10)
        self.slider_list = [a, b, c, d, e]
        self._update_layout(layout)
