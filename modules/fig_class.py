""" Module with the Main classes to plot the Figures """
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout


class FigWidget(QWidget):
    """ Class to create a new window with Plots """

    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self.layout = QVBoxLayout()


class PltFigure(FigureCanvasQTAgg):
    """ Class to draw canvas for a particular figure"""

    def __init__(self, parent, xlabel, ylabel, width=6, height=5, dpi=100):
        """ Initialize all the figure main elements """
        self.xlabel = xlabel
        self.ylabel = ylabel
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.draw_axes()
        super(PltFigure, self).__init__(fig)
        parent.addWidget(self)

    def draw_axes(self):
        """Draw x/y labels"""
        self.axes.yaxis.grid(True, linestyle="--")
        self.axes.xaxis.grid(True, linestyle="--")
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_xlabel(self.xlabel)

    def reinit(self):
        """Clean and then reinit the figure elements"""
        self.axes.clear()
        self.draw_axes()
        self.draw()
