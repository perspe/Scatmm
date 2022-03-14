import sys
import logging
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QRegExp
from PyQt5.QtGui import QIntValidator, QRegExpValidator


class CustomSlider(QtWidgets.QWidget):
    """
    Custom QtWidget with a QSlider and 2 QLineEdits by the side
    that allow the user to change the slider maximum and minimum
    Signal:
        changed: Signal for when the slider is moved
    Args:
        var_name (str): Variable name
        default_value (float): default value to show
        slider_min (float): minimum value for the slider
        slider_max (float): maximum value for the slider
        resolution (int): Resolution of the slider (bigger == more resolution)
        fixed_lim (bool): Fix the min and max values of the slider
    """
    changed = pyqtSignal(bool)

    def __init__(self,
                 var_name: str,
                 default_value: float,
                 slider_min: float = 1,
                 slider_max: float = 10,
                 resolution: int = 1000,
                 fixed_lim: bool = False) -> None:
        super().__init__()
        # Base variables
        self._var_name = var_name
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
        self._min_edit.setDisabled(fixed_lim)
        layout.addWidget(self._min_edit, 1, 0)
        # Add right QLineEdit
        self._max_edit = QtWidgets.QLineEdit()
        self._min_edit.setValidator(QIntValidator())
        self._max_edit.setText(str(self._slider_max))
        self._max_edit.setDisabled(fixed_lim)
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
        self._update_slider(default_value)
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
        self._curr_value.editingFinished.connect(self._update_slider)
        self._qslider.valueChanged.connect(self._update_label)
        self._qslider.valueChanged.connect(self.changed.emit)
        self._qslider.sliderMoved.connect(self._update_label)

    def curr_value(self) -> float:
        """ Get the current value in the slider, normalized for the 
        actual limits """
        curr_value: int = self._qslider.value()
        updated_curr_value: float = self._slider_min + (
            self._slider_max - self._slider_min) * (curr_value /
                                                    self._resolution)
        return updated_curr_value

    def _update_slider(self, value=None):
        """ Update slider after the value label is changed """
        str_val = self._curr_value.text()
        if value is None:
            value: float = float(str_val)
        if str_val == '':
            self._update_label()
            return
        logging.debug(f"{value=}")
        if value < self._slider_min:
            value = self._slider_min
        if value > self._slider_max:
            value = self._slider_max
        diff = (value - self._slider_min) / (self._slider_max -
                                             self._slider_min)
        self._qslider.setValue(int(diff * self._resolution))
        # Reupdate label to account for resolution mismatch
        self._update_label()

    def _update_label(self) -> None:
        """ Update Label from Slider Value """
        updated_curr_value: float = self.curr_value()
        self._curr_value.setText(f"{updated_curr_value:.3f}")

    def _change_max_edit(self) -> None:
        """ Update max bound for the slider """
        value = self._max_edit.text()
        if value == '':
            self._max_edit.setText(str(self._slider_max))
        else:
            self._slider_max: float = int(self._max_edit.text())
        self._update_slider()

    def _change_min_edit(self) -> None:
        """ Update min bound for the slider """
        value = self._min_edit.text()
        if value == '':
            self._min_edit.setText(str(self._slider_min))
        else:
            self._slider_min: float = int(self._min_edit.text())
        self._slider_min: float = int(self._min_edit.text())
        self._update_slider()

    @property
    def name(self):
        return self._var_name

    def __repr__(self) -> str:
        return f"{self._var_name} ({self.curr_value()})"


if __name__ == "__main__":
    """ Test Custom widgets """
    app = QtWidgets.QApplication(sys.argv)
    widget = CustomSlider("Eg")
    widget.show()
    app.exec_()
