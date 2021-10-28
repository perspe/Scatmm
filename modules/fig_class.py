""" Module with the Main classes to plot the Figures """
import logging
from typing import Any

from PyQt5.QtWidgets import QLayout, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class FigWidget(QWidget):
    """ Class to create a new window with Plots """

    def __init__(self, title: str):
        logging.info("FigWidget window added to VBOXLayout")
        super().__init__()
        self.setWindowTitle(title)
        self.layout: QLayout = QVBoxLayout()
        self.setLayout(self.layout)


class PltFigure(FigureCanvasQTAgg):
    """ Class to draw canvas for a particular figure"""

    def __init__(self, parent, xlabel, ylabel,  width=6, height=5, dpi=100):
        """ Initialize all the figure main elements """
        logging.info("Initialize Figure Canvas")
        self.xlabel = xlabel
        self.ylabel = ylabel
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes: Any = fig.add_subplot(111)
        self.draw_axes()
        super(PltFigure, self).__init__(fig)
        parent.addWidget(self)

    def draw_axes(self, xlabel=None, ylabel=None):
        """Draw x/y labels"""
        logging.debug("Draw/Labelling Axis")
        if xlabel:
            self.xlabel: str = xlabel
        if ylabel:
            self.ylabel: str = ylabel
        self.axes.yaxis.grid(True, linestyle="--")
        self.axes.xaxis.grid(True, linestyle="--")
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_xlabel(self.xlabel)


    def reinit(self):
        """ Clean and then reinit the figure elements """
        logging.debug("Reinitialize the plot")
        self.axes.clear()
        self.draw_axes()
        self.draw()
