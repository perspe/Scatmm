import sys
import logging
import numpy as np
from PyQt5 import Qt, QtCore
from PyQt5.QtWidgets import QMainWindow, QWidget, QSpacerItem, QSizePolicy
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from .smm_formula_mat import Ui_Formula
from .custom_widgets import CustomSlider
from modules.fig_class import PltFigure


def const(e, n, k):
    """ Formula for a constant refractive index """
    arr = np.ones_like(e, np.float64)
    return arr * n, arr * k


def tauc_lorentz(e, einf, eg, e0, a, c):
    """ Tauc Lorentz Formula for a single oscilator """
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
    er += (4 * a * e0 * eg * (e2 -
                             gamma**2) / (np.pi * zeta4 * alpha)) * (np.arctan(
                                 (alpha + 2 * eg) / c) + np.arctan(
                                     (alpha - 2 * eg) / c))
    er -= (a * e0 * c * (e2 + eg2) / (np.pi * zeta4 * e)) * np.log(
        np.abs(e - eg) / (e + eg))
    er += (2 * a * e0 * c * eg / (np.pi * zeta4)) * np.log(
        np.abs(e - eg) * (e + eg) / (np.sqrt((e02 - eg2)**2 + eg2 * c2)))

    er += einf
    # Convert to n/k
    e = er + 1j * ei
    n_comp = np.sqrt(e)
    n, k = np.real(n_comp), np.imag(n_comp)
    return n, k


methods: dict = {"Constant": const, "Tauc Lorentz": tauc_lorentz}


class FormulaWindow(QMainWindow):
    def __init__(self):
        super(FormulaWindow, self).__init__()
        logging.debug("Opened Formula Window")
        self.ui = Ui_Formula()
        self.ui.setupUi(self)
        # Internal variables of interest
        self.methods: dict = {
            "Constant": self.method_const,
            "Tauc Lorentz": self.method_TL
        }
        self._e = np.linspace(0.5, 6.5, 500, dtype=np.float64)
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

    """ Functions to Build the Variables for Different Methods """

    def _clear_variable_layout(self):
        while self.ui.variable_layout.count():
            child = self.ui.variable_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

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
        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum,
                                     QSizePolicy.Expanding)
        layout.addItem(verticalSpacer)
        self.slider_list = [n, k]
        for slider in self.slider_list:
            slider.changed.connect(self.update_plot)

    def method_TL(self):
        """ Create the variables for the const method
        Vars: n, k
        """
        self._clear_variable_layout()
        layout = self.ui.variable_layout
        einf = CustomSlider("ε∞", 1, 5)
        eg = CustomSlider("Eg", 0.5, 10)
        e0 = CustomSlider("E0", 0.5, 10)
        a = CustomSlider("A", 0.1, 500, 500)
        c = CustomSlider("C", 0.1, 10)
        self.slider_list = [einf, eg, e0, a, c]
        for slider in self.slider_list:
            layout.addWidget(slider)
            slider.changed.connect(self.update_plot)
        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum,
                                     QSizePolicy.Expanding)
        layout.addItem(verticalSpacer)
