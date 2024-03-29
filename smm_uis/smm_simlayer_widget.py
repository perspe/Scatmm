# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'simlayer_widget.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SimLayer(object):
    def setupUi(self, SimLayer):
        SimLayer.setObjectName("SimLayer")
        SimLayer.resize(505, 52)
        SimLayer.setMinimumSize(QtCore.QSize(0, 50))
        SimLayer.setMaximumSize(QtCore.QSize(16777215, 52))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setItalic(False)
        SimLayer.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/main_icon/Designer_UIs/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        SimLayer.setWindowIcon(icon)
        SimLayer.setStyleSheet("SimLayer{border: 5px solib black}")
        self.horizontalLayout = QtWidgets.QHBoxLayout(SimLayer)
        self.horizontalLayout.setContentsMargins(0, 2, 0, 0)
        self.horizontalLayout.setSpacing(15)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget_frame = QtWidgets.QFrame(SimLayer)
        self.widget_frame.setMinimumSize(QtCore.QSize(0, 50))
        self.widget_frame.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setItalic(False)
        self.widget_frame.setFont(font)
        self.widget_frame.setStyleSheet("QFrame#widget_frame{border: 1px solid black; margin: 2px; border-radius: 13px; padding: 0px}")
        self.widget_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.widget_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.widget_frame.setObjectName("widget_frame")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_frame)
        self.horizontalLayout_2.setContentsMargins(0, 0, 5, 0)
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.move_label = QtWidgets.QLabel(self.widget_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.move_label.sizePolicy().hasHeightForWidth())
        self.move_label.setSizePolicy(sizePolicy)
        self.move_label.setMinimumSize(QtCore.QSize(20, 45))
        self.move_label.setMaximumSize(QtCore.QSize(20, 45))
        self.move_label.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.move_label.setMouseTracking(True)
        self.move_label.setAutoFillBackground(False)
        self.move_label.setStyleSheet("QLabel#move_label { background-color : rgb(0, 170, 255); margin: 0px; border-top-left-radius: 10px; border-bottom-left-radius: 10px; border: 2px solid black;}")
        self.move_label.setLineWidth(0)
        self.move_label.setText("")
        self.move_label.setObjectName("move_label")
        self.horizontalLayout_2.addWidget(self.move_label)
        self.abs_label = QtWidgets.QLabel(self.widget_frame)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setItalic(False)
        self.abs_label.setFont(font)
        self.abs_label.setObjectName("abs_label")
        self.horizontalLayout_2.addWidget(self.abs_label)
        self.abs_cb = QtWidgets.QCheckBox(self.widget_frame)
        self.abs_cb.setText("")
        self.abs_cb.setObjectName("abs_cb")
        self.horizontalLayout_2.addWidget(self.abs_cb)
        spacerItem = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.mat_cb = QtWidgets.QComboBox(self.widget_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(7)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mat_cb.sizePolicy().hasHeightForWidth())
        self.mat_cb.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setItalic(False)
        self.mat_cb.setFont(font)
        self.mat_cb.setObjectName("mat_cb")
        self.horizontalLayout_2.addWidget(self.mat_cb)
        self.thickness_edit = QtWidgets.QLineEdit(self.widget_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thickness_edit.sizePolicy().hasHeightForWidth())
        self.thickness_edit.setSizePolicy(sizePolicy)
        self.thickness_edit.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setItalic(False)
        self.thickness_edit.setFont(font)
        self.thickness_edit.setAlignment(QtCore.Qt.AlignCenter)
        self.thickness_edit.setObjectName("thickness_edit")
        self.horizontalLayout_2.addWidget(self.thickness_edit)
        self.unit_label = QtWidgets.QLabel(self.widget_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.unit_label.sizePolicy().hasHeightForWidth())
        self.unit_label.setSizePolicy(sizePolicy)
        self.unit_label.setMinimumSize(QtCore.QSize(0, 0))
        self.unit_label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setItalic(False)
        self.unit_label.setFont(font)
        self.unit_label.setObjectName("unit_label")
        self.horizontalLayout_2.addWidget(self.unit_label)
        self.del_button = QtWidgets.QPushButton(self.widget_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.del_button.sizePolicy().hasHeightForWidth())
        self.del_button.setSizePolicy(sizePolicy)
        self.del_button.setMaximumSize(QtCore.QSize(15, 16777215))
        self.del_button.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/cross_button/Designer_UIs/cross-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.del_button.setIcon(icon1)
        self.del_button.setFlat(True)
        self.del_button.setObjectName("del_button")
        self.horizontalLayout_2.addWidget(self.del_button)
        self.horizontalLayout.addWidget(self.widget_frame)

        self.retranslateUi(SimLayer)
        QtCore.QMetaObject.connectSlotsByName(SimLayer)

    def retranslateUi(self, SimLayer):
        _translate = QtCore.QCoreApplication.translate
        SimLayer.setWindowTitle(_translate("SimLayer", "Form"))
        self.abs_label.setText(_translate("SimLayer", "Abs"))
        self.abs_cb.setToolTip(_translate("SimLayer", "Plot the absorption profile of this particular layer in the stack"))
        self.thickness_edit.setToolTip(_translate("SimLayer", "Layer thickness"))
        self.thickness_edit.setPlaceholderText(_translate("SimLayer", "Thickness"))
        self.unit_label.setText(_translate("SimLayer", "nm"))
from . import icons_rc
