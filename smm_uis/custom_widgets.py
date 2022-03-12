import sys
import logging
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QRegExp
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QRegExpValidator


class CustomSlider(QtWidgets.QWidget):
    """
    Custom QtWidget with a QSlider and 2 QLineEdits by the side
    that allow the user to change the slider maximum and minimum
    """
    changed = pyqtSignal(bool)

    def __init__(self,
                 var_name: str,
                 slider_min: float = 1,
                 slider_max: float = 10,
                 resolution: int = 100) -> None:
        super().__init__()
        # Base variables
        self._slider_min: float = slider_min
        self._slider_max: float = slider_max
        self._resolution: int = resolution
        layout = QtWidgets.QGridLayout()
        layout.setHorizontalSpacing(5)
        layout.setVerticalSpacing(2)
        # QLabel to input the variable name
        self._var_label = QtWidgets.QLabel()
        self._var_label.setText(var_name)
        layout.addWidget(self._var_label, 0, 0)
        # Add left QLineEdit
        self._min_edit = QtWidgets.QLineEdit()
        self._min_edit.setValidator(QIntValidator())
        self._min_edit.setText(str(self._slider_min))
        layout.addWidget(self._min_edit, 1, 0)
        # Add right QLineEdit
        self._max_edit = QtWidgets.QLineEdit()
        self._min_edit.setValidator(QIntValidator())
        self._max_edit.setText(str(self._slider_max))
        layout.addWidget(self._max_edit, 1, 2)
        # Add QSlider
        # The slider goes from 0-100 in 2 spaces
        # This then needs to be adapted to the given variable limits
        self._qslider = QtWidgets.QSlider(Qt.Horizontal)
        self._qslider.setMinimum(0)
        self._qslider.setMaximum(self._resolution)
        self._qslider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self._qslider.setTickInterval(int(self._resolution / 10))
        layout.addWidget(self._qslider, 1, 1)
        # Add Indicator for current Slider value
        self._curr_value = QtWidgets.QLineEdit()
        self._curr_value.setMaximumWidth(80)
        num_regex = QRegExp("[0-9]+\\.?[0-9]*j?")
        self._curr_value.setValidator(QRegExpValidator(num_regex))
        self._update_label()
        layout.addWidget(self._curr_value, 0, 1)
        self.setLayout(layout)
        # Ratio 1 - 2 - 1 for horizontal space for widgets
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 1)
        self.initializeUI()

    def initializeUI(self) -> None:
        """ Connect elements to functions """
        self._max_edit.editingFinished.connect(self._change_max_edit)
        self._min_edit.editingFinished.connect(self._change_min_edit)
        self._qslider.sliderMoved.connect(self._update_label)
        self._qslider.valueChanged.connect(self.changed.emit)
        self._curr_value.editingFinished.connect(self._update_slider)

    def curr_value(self) -> float:
        """ Get the current value in the slider, normalized for the 
        actual limits """
        curr_value: int = self._qslider.value()
        updated_curr_value: float = self._slider_min + (
            self._slider_max - self._slider_min) * (curr_value /
                                                    self._resolution)
        return updated_curr_value

    def _update_slider(self):
        """ Update slider after the value label is changed """
        value: float = float(self._curr_value.text())
        if value < self._slider_min:
            value = self._slider_min
            self._curr_value.setText(f"{value:.2f}")
        if value > self._slider_max:
            value = self._slider_max
            self._curr_value.setText(f"{value:.2f}")
        logging.debug(f"{value=}")
        diff = (value - self._slider_min)/(self._slider_max - self._slider_min)
        self._qslider.setValue(int(diff*self._resolution))

    def _update_label(self) -> None:
        """ Update Label from Slider Value """
        updated_curr_value: float = self.curr_value()
        self._curr_value.setText(f"{updated_curr_value:.2f}")

    def _change_max_edit(self) -> None:
        """ Update max bound for the slider """
        self._slider_max: float = int(self._max_edit.text())
        self._update_label()

    def _change_min_edit(self) -> None:
        """ Update min bound for the slider """
        self._slider_min: float = int(self._min_edit.text())
        self._update_label()


if __name__ == "__main__":
    """ Test Custom widgets """
    app = QtWidgets.QApplication(sys.argv)
    widget = CustomSlider("Eg")
    widget.show()
    app.exec_()
