# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'smm_uis/Designer_UIs/export_gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ExportWindow(object):
    def setupUi(self, ExportWindow):
        ExportWindow.setObjectName("ExportWindow")
        ExportWindow.resize(579, 322)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        ExportWindow.setWindowIcon(icon)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(ExportWindow)
        self.horizontalLayout_2.setContentsMargins(15, 15, 15, 15)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.checkbox_frame = QtWidgets.QFrame(ExportWindow)
        self.checkbox_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.checkbox_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.checkbox_frame.setLineWidth(0)
        self.checkbox_frame.setObjectName("checkbox_frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.checkbox_frame)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName("verticalLayout")
        self.reflection_checkbox = QtWidgets.QCheckBox(self.checkbox_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.reflection_checkbox.sizePolicy().hasHeightForWidth())
        self.reflection_checkbox.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.reflection_checkbox.setFont(font)
        self.reflection_checkbox.setChecked(True)
        self.reflection_checkbox.setObjectName("reflection_checkbox")
        self.verticalLayout.addWidget(self.reflection_checkbox)
        self.transmission_checkbox = QtWidgets.QCheckBox(self.checkbox_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.transmission_checkbox.sizePolicy().hasHeightForWidth())
        self.transmission_checkbox.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.transmission_checkbox.setFont(font)
        self.transmission_checkbox.setChecked(True)
        self.transmission_checkbox.setObjectName("transmission_checkbox")
        self.verticalLayout.addWidget(self.transmission_checkbox)
        self.absorption_checkbox = QtWidgets.QCheckBox(self.checkbox_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.absorption_checkbox.sizePolicy().hasHeightForWidth())
        self.absorption_checkbox.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.absorption_checkbox.setFont(font)
        self.absorption_checkbox.setChecked(True)
        self.absorption_checkbox.setObjectName("absorption_checkbox")
        self.verticalLayout.addWidget(self.absorption_checkbox)
        self.layers_checkbox = QtWidgets.QCheckBox(self.checkbox_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.layers_checkbox.sizePolicy().hasHeightForWidth())
        self.layers_checkbox.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setItalic(False)
        self.layers_checkbox.setFont(font)
        self.layers_checkbox.setObjectName("layers_checkbox")
        self.verticalLayout.addWidget(self.layers_checkbox)
        spacerItem = QtWidgets.QSpacerItem(20, 150, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_2.addWidget(self.checkbox_frame)
        self.frame = QtWidgets.QFrame(ExportWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setContentsMargins(5, 10, 5, 10)
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.simulations_combobox = QtWidgets.QComboBox(self.frame)
        self.simulations_combobox.setObjectName("simulations_combobox")
        self.verticalLayout_2.addWidget(self.simulations_combobox)
        self.simulation_summary = QtWidgets.QTextEdit(self.frame)
        self.simulation_summary.setReadOnly(True)
        self.simulation_summary.setObjectName("simulation_summary")
        self.verticalLayout_2.addWidget(self.simulation_summary)
        self.buttons_frame = QtWidgets.QFrame(self.frame)
        self.buttons_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.buttons_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.buttons_frame.setLineWidth(0)
        self.buttons_frame.setObjectName("buttons_frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.buttons_frame)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.export_all_button = QtWidgets.QPushButton(self.buttons_frame)
        self.export_all_button.setObjectName("export_all_button")
        self.horizontalLayout.addWidget(self.export_all_button)
        self.export_button = QtWidgets.QPushButton(self.buttons_frame)
        self.export_button.setObjectName("export_button")
        self.horizontalLayout.addWidget(self.export_button)
        self.preview_button = QtWidgets.QPushButton(self.buttons_frame)
        self.preview_button.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.preview_button.setObjectName("preview_button")
        self.horizontalLayout.addWidget(self.preview_button)
        self.verticalLayout_2.addWidget(self.buttons_frame)
        self.horizontalLayout_2.addWidget(self.frame)

        self.retranslateUi(ExportWindow)
        QtCore.QMetaObject.connectSlotsByName(ExportWindow)

    def retranslateUi(self, ExportWindow):
        _translate = QtCore.QCoreApplication.translate
        ExportWindow.setWindowTitle(_translate("ExportWindow", "Export Simulation"))
        self.reflection_checkbox.setText(_translate("ExportWindow", "Reflection"))
        self.transmission_checkbox.setText(_translate("ExportWindow", "Transmission"))
        self.absorption_checkbox.setText(_translate("ExportWindow", "Absorption"))
        self.layers_checkbox.setText(_translate("ExportWindow", "Layers"))
        self.export_all_button.setText(_translate("ExportWindow", "Export all"))
        self.export_button.setText(_translate("ExportWindow", "Export"))
        self.preview_button.setText(_translate("ExportWindow", "Preview"))
