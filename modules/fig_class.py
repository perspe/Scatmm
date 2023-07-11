""" Module with the Main classes to plot the Figures """
import logging
from typing import Any

from PyQt5.QtWidgets import QLayout, QVBoxLayout, QWidget
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class FigWidget(QWidget):
    """Class to create a new window with Plots"""

    def __init__(self, title: str):
        logging.info("FigWidget window added to VBOXLayout")
        super().__init__()
        self.setWindowTitle(title)
        self.layout: QLayout = QVBoxLayout()
        self.setLayout(self.layout)


class PltFigure(FigureCanvasQTAgg):
    """
    Class to draw canvas for a particular figure
    Args:
        parent: parent widget where the figure will be drawn
        xlabel, ylabel: Labels for x and yaxis
        width, height, dpi: Image size/resolution
        interactive: wheter to add interactive vertical guideline
    Properties:
        figBuffer: Holder for the figure buffer (for interactive plots)
    Methods:
        draw_axes: Redraw axes elements (given a xlabel and ylabel)
        reinit: Clean an reinitialize figure elements
        reset_figBuffer: Update the figure buffer (for when new lines are added)
        fast_draw: Fast draw line elements onto the plot
        bufferRedraw: wrapper around draw to also save the figure to the buffer
        deleteLinesGID: delete lines from the plot from their gid
    """
    def __init__(
        self, parent, xlabel, ylabel, width=6, height=5, dpi=100, interactive=True
    ):
        """Initialize all the figure main elements"""
        logging.info("Initialize Figure Canvas")
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._vline = None
        self._vlineLock = True
        self._fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes: Any = self._fig.add_subplot(111)
        self.draw_axes()
        super(PltFigure, self).__init__(self._fig)
        parent.addWidget(self)
        # Add connections to mouse events
        if interactive:
            self._press = self.mpl_connect("button_press_event", self.mouse_event)
            self._motion = self.mpl_connect("motion_notify_event", self.mouse_moved)
            self._release = self.mpl_connect(
                "button_release_event", self.mouse_released
            )
        # Variable to store the figure buffer for fast redraw
        self._fig_buffer = self.copy_from_bbox(self._fig.bbox)

    """ Initialization functions """

    def draw_axes(self, xlabel=None, ylabel=None):
        """ Update x and ylabels for the plot """
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
        """Clean and then reinit the figure elements"""
        logging.debug("Reinitialize the plot")
        self.axes.clear()
        self.draw_axes()
        self.bufferRedraw()

    """ Figure Properties """

    @property
    def figBuffer(self):
        return self._fig_buffer

    @figBuffer.setter
    def figBuffer(self, value):
        self._fig_buffer = value

    """ Figure Buffer properties """

    def reset_figBuffer(self):
        """
        Overwrite the figBuffer variable to accommodate new plot changes
        Also considers and ignores the vertical guideline
        """
        # Check for the vertical guideline to avoid storing it in the buffer
        if self._vline is not None:
            xdata = self._vline.get_xdata()[0]
            logging.debug(f"Vline: {xdata}")
            self.deleteLinesGID(["vline"])
            self._vline = self.axes.axvline(
                xdata, linestyle="--", color="k", picker=True, gid="vline", pickradius=15
            )
            self._fig_buffer = self.copy_from_bbox(self._fig.bbox)
            self.fast_draw([self._vline])
        else:
            self._fig_buffer = self.copy_from_bbox(self._fig.bbox)

    def fast_draw(self, lines):
        """ Fast draw line elements onto the canvas """
        self._fig.canvas.restore_region(self._fig_buffer)
        for line in lines:
            self.axes.draw_artist(line)
        self._fig.canvas.blit(self._fig.bbox)
        self._fig.canvas.flush_events()

    def bufferRedraw(self):
        """Wrapper around the draw function to also run reset_figBuffer"""
        self.draw()
        self.reset_figBuffer()

    """ Interaction functions """

    def deleteLinesGID(self, lines):
        """ Delete lines based on gid """
        for line in self.axes.get_lines():
            if line.get_gid() in lines:
                logging.debug(f"Removing line: {line.get_gid()}")
                line.remove()
        self.draw()

    """ Mouse events """

    def mouse_moved(self, event):
        """ Update vline position based on mouse movement """
        if event.inaxes != self.axes:
            return
        if event.button == MouseButton.LEFT and not self._vlineLock:
            if self._vline is None:
                return
            elif self._vline is not None:
                x_min, x_max = self.axes.get_xlim()
                if (
                    abs((event.xdata - self._vline.get_xdata()[0]) / (x_max - x_min))
                    < 0.1
                ):
                    self._vline.set_xdata([event.xdata, event.xdata])
                    self.fast_draw([self._vline])

    def mouse_event(self, event):
        """
        This event has 2 main aspects:
            1. Detect double click on canvas → create a vertical guideline
            2. Detect "start drag event" to unlock vline movement
                (avoids event colisions)
        """
        logging.debug(f"{event.xdata}::{event.ydata}")
        if len(self.axes.get_lines()) == 0:
            logging.debug("No plot lines... Not adding vline")
            return
        x_min, x_max = self.axes.get_xlim()
        if event.dblclick:
            # Add line
            if self._vline is None:
                self.draw()
                self._vline = self.axes.axvline(
                    event.xdata,
                    linestyle="--",
                    color="k",
                    picker=True,
                    gid="vline",
                    pickradius=5,
                )
                # Store figure in buffer
                self.reset_figBuffer()
                self.fast_draw([self._vline])
            # Delete line
            elif abs(event.xdata - self._vline.get_xdata()[0]) / (x_max - x_min) < 0.01:
                logging.debug("Remove vline")
                self.deleteLinesGID(["vline"])
                self._vline = None
            else:
                logging.debug("Line too far")
        # Start moving line
        elif event.button == MouseButton.LEFT and self._vline is not None:
            if abs(event.xdata - self._vline.get_xdata()[0]) / (x_max - x_min) < 0.01:
                logging.debug("Picking vline... Releasing lock")
                self._vlineLock = False
        else:
            logging.critical("Non-considered mouse event")

    def mouse_released(self, _):
        if not self._vlineLock and self._vline is not None:
            logging.debug("Relocking vertical line")
            self._vlineLock = True
