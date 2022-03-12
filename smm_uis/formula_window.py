import logging

from PyQt5 import Qt, QtCore
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QSizePolicy, QSpacerItem, QWidget
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from modules.fig_class import PltFigure
import numpy as np
import scipy.constants as scc

from .custom_widgets import CustomSlider
from .smm_formula_mat import Ui_Formula


def const(e, n, k):
    """ Formula for a constant refractive index """
    arr = np.ones_like(e, np.float64)
    return arr * n, arr * k


def tauc_lorentz_peak(e, eg, e0, a, c):
    """ Formula to calcuate one peak for the Tauc Lorentz formula """
    logging.debug(f"{eg=}::{e0=}::{a=}::{c=}")
    ei = np.zeros_like(e, dtype=np.float64)
    er = np.zeros_like(e, dtype=np.float64)
    # Interim variables
    e2, eg2, e02, c2 = np.power(e, 2), eg**2, e0**2, c**2
    gamma = np.sqrt(e02 - c2 / 2)
    alpha = np.sqrt(4 * e02 - c2)
    zeta4 = np.power(e2 - gamma**2, 2) + alpha**2 * c2 / 4
    ain = (eg2 - e02) * e2 + eg2 * c2 - e02 * (e02 + 3 * eg2)
    atan = (e2 - e02) * (e02 + eg2) + eg2 * c2
    # Proceed to calculation
    eg_mask = e > eg
    # Imaginary part
    ei[eg_mask] = (1 / e[eg_mask]) * (a * e0 * c * (e[eg_mask] - eg)**2) / (
        (e2[eg_mask] - e02)**2 + c2 * e2[eg_mask])
    # Real part
    er += (a * c * ain / (2 * np.pi * zeta4 * alpha * e0)) * np.log(
        (e02 + eg2 + alpha * eg) / (e02 + eg2 - alpha * eg))
    er -= (a * atan) / (np.pi * zeta4 * e0) * (np.pi - np.arctan(
        (2 * eg + alpha) / c) + np.arctan((alpha - 2 * eg) / c))
    er += (4 * a * e0 * eg * (e2 - gamma**2) /
           (np.pi * zeta4 * alpha)) * (np.arctan(
               (alpha + 2 * eg) / c) + np.arctan((alpha - 2 * eg) / c))
    er -= (a * e0 * c * (e2 + eg2) /
           (np.pi * zeta4 * e)) * np.log(np.abs(e - eg) / (e + eg))
    er += (2 * a * e0 * c * eg / (np.pi * zeta4)) * np.log(
        np.abs(e - eg) * (e + eg) / (np.sqrt((e02 - eg2)**2 + eg2 * c2)))
    return er, ei


def tauc_lorentz_1(e, einf, eg, e0, a, c):
    """ Tauc Lorentz Equation with one peak """
    er, ei = tauc_lorentz_peak(e, eg, e0, a, c)
    er += einf
    e_complex = er + 1j * ei
    n = np.sqrt(e_complex)
    return np.real(n), np.imag(n)


def tauc_lorentz_2(e, einf, eg, e01, a1, c1, e02, a2, c2):
    """ Tauc Lorentz Equation with one peak """
    er1, ei1 = tauc_lorentz_peak(e, eg, e01, a1, c1)
    er2, ei2 = tauc_lorentz_peak(e, eg, e02, a2, c2)
    er = er1 + er2 + einf
    ei = ei1 + ei2
    e_complex = er + 1j * ei
    n = np.sqrt(e_complex)
    return np.real(n), np.imag(n)


def tauc_lorentz_n(e, n_peak, einf, eg, *args):
    """ Tauc Lorentx Equation for multiple peaks
    Args:
        e (array): energy array
        n_peak (int): Number of peaks
        einf (float): einf variable
        eg (float): eg variable
        *args (e0, a, c)*n_peak: remaining peak variables
    """
    if len(args) < 3 * n_peak:
        raise Exception("Number of arguments not compatible with n_peak")
    er = np.zeros_like(e, dtype=np.float64)
    ei = np.zeros_like(e, dtype=np.float64)
    for i in range(n_peak):
        e0i, ai, ci = args[i * 3 + 0], args[i * 3 + 1], args[i * 3 + 2]
        er_i, ei_i = tauc_lorentz_peak(e, eg, e0i, ai, ci)
        er += er_i
        ei += ei_i
    er += einf
    e_complex = er + 1j * ei
    n = np.sqrt(e_complex)
    return np.real(n), np.imag(n)


def cauchy(e, a, b, c):
    """ Standard non-absorber cauchy formula """
    logging.debug(f"{a=}::{b=}::{c=}")
    lmb = (scc.h * scc.c) / (scc.e * e) * 1e9
    logging.debug(lmb)
    n = a + 1e4 * b / lmb**2 + 1e9 * c / lmb**4
    k = np.zeros_like(n)
    return n, k


methods: dict = {
    "Constant": const,
    "Tauc Lorentz 1": tauc_lorentz_1,
    "Tauc Lorentz 2": tauc_lorentz_2,
    "Cauchy": cauchy
}

_nm_to_ev = (scc.h * scc.c) / (scc.e * 1e-9)
# _ev_to_nm = (scc.h * scc.c) / (scc.e * 1e-9)
Units = {
    "Energy (eV)": ("Emin (eV)", "Emax (eV)"),
    "Wavelength (nm)": ("λmin (nm)", "λmax (nm)")
}


class FormulaWindow(QMainWindow):
    def __init__(self, parent=None):
        self.parent = parent
        super(FormulaWindow, self).__init__()
        logging.debug("Opened Formula Window")
        self.ui = Ui_Formula()
        self.ui.setupUi(self)
        # Internal variables of interest
        # Connect Combobox names to functions
        self.methods: dict = {
            "Constant": self.method_const,
            "Tauc Lorentz 1": self.method_TL_1,
            "Tauc Lorentz 2": self.method_TL_2,
            "Cauchy": self.cauchy
        }
        self.slider_list = []
        # Fill comboboxes
        self.ui.units_cb.addItems([str(unit) for unit in Units.keys()])
        self.ui.method_cb.addItems([key for key in self.methods.keys()])
        # Define the xdata, xlims and the xvariable
        self._xmin = 0.5
        self._xmax = 6.5
        self._xres = 500
        self._e = np.linspace(self._xmin,
                              self._xmax,
                              self._xres,
                              dtype=np.float64)
        self._lmb = _nm_to_ev * (1 / self._e)
        self.xlim = [self.ui.xmin_value, self.ui.xmax_value]
        self.xlim[0].setText(str(self._xmin))
        self.xlim[1].setText(str(self._xmax))
        # Create the figure to plot the data (with 2 y axis for n and k)
        self.plot_canvas = PltFigure(self.ui.plot_layout,
                                     self.ui.units_cb.currentText(), "n")
        self.addToolBar(QtCore.Qt.TopToolBarArea,
                        NavigationToolbar2QT(self.plot_canvas, self))
        self.n_plot = self.plot_canvas.axes
        self.n_plot.set_ylabel("n", color='b')
        self.n_plot.set_ylim((0, 5))
        self.k_plot = self.n_plot.twinx()
        self.k_plot.set_ylabel("k", color="r")
        self.k_plot.set_ylim((0, 5))
        # InitializeUI and update plot info
        self.initializeUI()
        # Make the first plot iteration
        self.methods[self.ui.method_cb.currentText()]()
        self._update_nk()
        nplot = self.n_plot.plot(self._e, self._n, c='b')
        kplot = self.k_plot.plot(self._e, self._k, c='r')
        self._n_plot = nplot[0]
        self._k_plot = kplot[0]

    def initializeUI(self):
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

    def update_method(self):
        """ Update the formula """
        new_method: str = self.ui.method_cb.currentText()
        logging.debug(f"Method Updated to: {new_method}")
        self.methods[new_method]()
        self._rebuild_plot()

    def _update_nk(self):
        """ Get updated nk values for current given parameters """
        curr_method: str = self.ui.method_cb.currentText()
        curr_values = tuple(
            [slider_i.curr_value() for slider_i in self.slider_list])
        self._n, self._k = methods[curr_method](self._e, *curr_values)

    def _update_plot(self):
        """ Calculate n/k from variables in widget and plot """
        self._update_nk()
        self._n_plot.set_ydata(self._n)
        self._k_plot.set_ydata(self._k)
        self.plot_canvas.draw()

    def _rebuild_plot(self):
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

    def add_database(self):
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

    def update_xvar(self):
        """ Update the overall representation of xvar, between E and Wvl """
        new_var = self.ui.units_cb.currentText()
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

    def update_xlim(self):
        """ Update the xlims for calculation """
        curr_var = self.ui.units_cb.currentText()
        xmin = float(self.ui.xmin_value.text())
        xmax = float(self.ui.xmax_value.text())
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

    def _clear_variable_layout(self):
        """ Function to clear all the widgets in the layout """
        while self.ui.variable_layout.count():
            child = self.ui.variable_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _update_layout(self, layout):
        """ Update the slider layout """
        for slider in self.slider_list:
            layout.addWidget(slider)
            slider.changed.connect(self._update_plot)
        verticalSpacer = QSpacerItem(20, 80, QSizePolicy.Minimum,
                                     QSizePolicy.Expanding)
        layout.addItem(verticalSpacer)

    def method_const(self):
        """ Create the variables for the const method
        Vars: n, k
        """
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        n = CustomSlider("n", 1, 5)
        k = CustomSlider("k", 0, 5)
        layout.addWidget(n)
        layout.addWidget(k)
        self.slider_list = [n, k]
        self._update_layout(layout)

    def method_TL_1(self):
        """ Create the variables for the const method
        Vars: einf, eg, e0, a, c
        """
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        einf = CustomSlider("ε∞", 1, 5)
        eg = CustomSlider("Eg", 0.5, 10)
        e0 = CustomSlider("E0", 0.5, 10)
        a = CustomSlider("A", 0.1, 500, 500)
        c = CustomSlider("C", 0.1, 10)
        self.slider_list = [einf, eg, e0, a, c]
        self._update_layout(layout)

    def method_TL_2(self):
        """ Create the variables for the const method
        Vars: einf, eg, e01, a1, c1, e02, a2, c2
        """
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        einf = CustomSlider("ε∞", 1, 5)
        eg = CustomSlider("Eg", 0.5, 10)
        e01 = CustomSlider("E0", 0.5, 10)
        a1 = CustomSlider("A", 0.1, 500, 500)
        c1 = CustomSlider("C", 0.1, 10)
        e02 = CustomSlider("E0", 0.5, 10)
        a2 = CustomSlider("A", 0.1, 500, 500)
        c2 = CustomSlider("C", 0.1, 10)
        self.slider_list = [einf, eg, e01, a1, c1, e02, a2, c2]
        self._update_layout(layout)

    def cauchy(self):
        """ Create the variables for the const method
        Vars: A, B, C
        """
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        a = CustomSlider("A", 1, 5)
        b = CustomSlider("B", 0.5, 10)
        c = CustomSlider("C", 0.5, 10)
        self.slider_list = [a, b, c]
        self._update_layout(layout)
