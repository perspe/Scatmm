import logging
import sys

from PyQt5 import Qt, QtCore
from PyQt5.QtWidgets import QMainWindow, QSizePolicy, QSpacerItem, QWidget
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


class FormulaWindow(QMainWindow):
    def __init__(self, parent=None):
        self.parent = parent
        super(FormulaWindow, self).__init__()
        logging.debug("Opened Formula Window")
        self.ui = Ui_Formula()
        self.ui.setupUi(self)
        # Internal variables of interest
        self.methods: dict = {
            "Constant": self.method_const,
            "Tauc Lorentz 1": self.method_TL_1,
            "Tauc Lorentz 2": self.method_TL_2,
            "Cauchy": self.cauchy
        }
        self._e = np.linspace(0.5, 6.5, 500, dtype=np.float64)
        self._lmb = (scc.c * scc.h) / (scc.e *
                                       self._e) * 1e9  # Wavelength in nm
        self.slider_list = []
        # Create the figure to plot the data
        self.plot_canvas = PltFigure(self.ui.plot_layout,
                                     "Energy (eV)",
                                     "n",
                                     width=7)
        self.n_plot = self.plot_canvas.axes
        self.n_plot.set_ylabel("n", color='b')
        self.n_plot.set_ylim((0, 5))
        self.k_plot = self.n_plot.twinx()
        self.k_plot.set_ylabel("k", color="r")
        self.k_plot.set_ylim((0, 5))
        # Update Combobox with name
        self.ui.method_cb.addItems([key for key in self.methods.keys()])
        self.initializeUI()
        self.update_method(True)
        # Make the initialization plot
        self._n, self._k = self.update_nk()
        nplot = self.n_plot.plot(self._e, self._n, c='b')
        kplot = self.k_plot.plot(self._e, self._k, c='r')
        self._n_plot = nplot[0]
        self._k_plot = kplot[0]
        self.addToolBar(QtCore.Qt.TopToolBarArea,
                        NavigationToolbar2QT(self.plot_canvas, self))

    def initializeUI(self):
        """ Connect functions to buttons """
        self.ui.method_cb.currentIndexChanged.connect(
            lambda a: self.update_method(False))
        self.ui.add_db_button.clicked.connect(self.add_database)

    """ Update the n/k info and the plot """

    def update_method(self, first_run=False):
        """ Update the formula """
        new_method: str = self.ui.method_cb.currentText()
        logging.debug(f"Method Updated to: {new_method}")
        # Update the variables in methods
        self.methods[new_method]()
        if first_run is False:
            logging.debug("Updating plot for new method")
            self.update_plot()

    def update_nk(self):
        """ Get updated nk values for current given parameters """
        curr_method: str = self.ui.method_cb.currentText()
        curr_values = tuple(
            [slider_i.curr_value() for slider_i in self.slider_list])
        n, k = methods[curr_method](self._e, *curr_values)
        return n, k

    def update_plot(self):
        """ Calculate n/k from variables in widget and plot """
        self._n, self._k = self.update_nk()
        self._n_plot.set_ydata(self._n)
        self._k_plot.set_ydata(self._k)
        self.plot_canvas.draw()

    """ Window buttons functions """

    def add_database(self):
        return

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
            slider.changed.connect(self.update_plot)
        verticalSpacer = QSpacerItem(20, 80, QSizePolicy.Minimum,
                                     QSizePolicy.Expanding)
        layout.addItem(verticalSpacer)

    def method_const(self):
        """ Create the variables for the const method
        Vars: n, k
        """
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        n = CustomSlider("n", 1, 10)
        k = CustomSlider("k")
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
