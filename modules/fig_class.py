""" Module with the Main classes to plot the Figures """
import logging
from typing import Any

from PyQt5.QtWidgets import QLayout, QVBoxLayout, QWidget
from matplotlib.backend_bases import MouseButton
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
    def __init__(self, parent, xlabel, ylabel, width=6, height=5, dpi=100):
        """ Initialize all the figure main elements """
        logging.info("Initialize Figure Canvas")
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._vline = None
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes: Any = fig.add_subplot(111)
        self.draw_axes()
        super(PltFigure, self).__init__(fig)
        parent.addWidget(self)
        self.mpl_connect('button_press_event', self.mouse_event)
        self.mpl_connect('motion_notify_event', self.mouse_moved)

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

    def mouse_moved(self, event):
        """ Detect if mouse was moved.. If vline is in range... move it """
        if event.button == MouseButton.RIGHT:
            if self._vline is None:
                return
            elif self._vline is not None:
                logging.debug(f"{event.xdata}::{self._vline.get_xdata()}")
                x_min, x_max = self.axes.get_xlim()
                if abs((event.xdata - self._vline.get_xdata()[0]) /
                       (x_max - x_min)) < 0.1:
                    self._vline.set_xdata([event.xdata, event.xdata])
                    self.draw()

    def mouse_event(self, event):
        """ Detect mouse event and if it is double click,
        add or remove a vertical line (to help with fitting) """
        logging.debug(f"{event.xdata}::{event.ydata}")
        if event.dblclick:
            x_min, x_max = self.axes.get_xlim()
            if self._vline is None:
                self._vline = self.axes.axvline(event.xdata,
                                                linestyle='--',
                                                color='k',
                                                picker=True)
            elif abs(event.xdata -
                     self._vline.get_xdata()[0]) / (x_max - x_min) < 0.01:
                logging.debug(event.xdata - self._vline.get_xdata()[0])
                self._vline.remove()
                self._vline = None
        self.draw()

    def reinit(self):
        """ Clean and then reinit the figure elements """
        logging.debug("Reinitialize the plot")
        self.axes.clear()
        self.draw_axes()
        self.draw()
