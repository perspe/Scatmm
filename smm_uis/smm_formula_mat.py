# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'formula_mat.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Formula(object):
    def setupUi(self, Formula):
        Formula.setObjectName("Formula")
        Formula.resize(923, 670)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Formula.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(Formula)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.material_name_label = QtWidgets.QLabel(self.frame_2)
        self.material_name_label.setObjectName("material_name_label")
        self.horizontalLayout_2.addWidget(self.material_name_label)
        self.mat_name = QtWidgets.QLineEdit(self.frame_2)
        self.mat_name.setObjectName("mat_name")
        self.horizontalLayout_2.addWidget(self.mat_name)
        self.verticalLayout.addWidget(self.frame_2)
        self.method_cb = QtWidgets.QComboBox(self.frame)
        self.method_cb.setObjectName("method_cb")
        self.verticalLayout.addWidget(self.method_cb)
        self.variables_frame = QtWidgets.QFrame(self.frame)
        self.variables_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.variables_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.variables_frame.setObjectName("variables_frame")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.variables_frame)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(5)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.scrollArea = QtWidgets.QScrollArea(self.variables_frame)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 276, 432))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.variable_layout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.variable_layout.setContentsMargins(0, 0, 0, 0)
        self.variable_layout.setSpacing(0)
        self.variable_layout.setObjectName("variable_layout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.variable_layout.addLayout(self.verticalLayout_2)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_3.addWidget(self.scrollArea)
        self.verticalLayout.addWidget(self.variables_frame)
        self.import_button = QtWidgets.QPushButton(self.frame)
        self.import_button.setObjectName("import_button")
        self.verticalLayout.addWidget(self.import_button)
        self.add_db_button = QtWidgets.QPushButton(self.frame)
        self.add_db_button.setObjectName("add_db_button")
        self.verticalLayout.addWidget(self.add_db_button)
        self.horizontalLayout.addWidget(self.frame)
        self.window_frame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.window_frame.sizePolicy().hasHeightForWidth())
        self.window_frame.setSizePolicy(sizePolicy)
        self.window_frame.setMinimumSize(QtCore.QSize(500, 500))
        self.window_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.window_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.window_frame.setLineWidth(2)
        self.window_frame.setObjectName("window_frame")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.window_frame)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(5)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.frame_5 = QtWidgets.QFrame(self.window_frame)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.left_cb_label = QtWidgets.QLabel(self.frame_5)
        self.left_cb_label.setObjectName("left_cb_label")
        self.horizontalLayout_5.addWidget(self.left_cb_label)
        self.left_axis_cb = QtWidgets.QComboBox(self.frame_5)
        self.left_axis_cb.setObjectName("left_axis_cb")
        self.horizontalLayout_5.addWidget(self.left_axis_cb)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.right_axis_label = QtWidgets.QLabel(self.frame_5)
        self.right_axis_label.setObjectName("right_axis_label")
        self.horizontalLayout_5.addWidget(self.right_axis_label)
        self.right_axis_cb = QtWidgets.QComboBox(self.frame_5)
        self.right_axis_cb.setObjectName("right_axis_cb")
        self.horizontalLayout_5.addWidget(self.right_axis_cb)
        self.verticalLayout_4.addWidget(self.frame_5)
        self.frame_4 = QtWidgets.QFrame(self.window_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.plot_layout = QtWidgets.QVBoxLayout()
        self.plot_layout.setSpacing(0)
        self.plot_layout.setObjectName("plot_layout")
        self.horizontalLayout_4.addLayout(self.plot_layout)
        self.verticalLayout_4.addWidget(self.frame_4)
        self.frame_3 = QtWidgets.QFrame(self.window_frame)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.units_label = QtWidgets.QLabel(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.units_label.sizePolicy().hasHeightForWidth())
        self.units_label.setSizePolicy(sizePolicy)
        self.units_label.setObjectName("units_label")
        self.horizontalLayout_3.addWidget(self.units_label)
        self.units_cb = QtWidgets.QComboBox(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.units_cb.sizePolicy().hasHeightForWidth())
        self.units_cb.setSizePolicy(sizePolicy)
        self.units_cb.setObjectName("units_cb")
        self.horizontalLayout_3.addWidget(self.units_cb)
        spacerItem1 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.xmin_label = QtWidgets.QLabel(self.frame_3)
        self.xmin_label.setObjectName("xmin_label")
        self.horizontalLayout_3.addWidget(self.xmin_label)
        self.xmin_value = QtWidgets.QLineEdit(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.xmin_value.sizePolicy().hasHeightForWidth())
        self.xmin_value.setSizePolicy(sizePolicy)
        self.xmin_value.setObjectName("xmin_value")
        self.horizontalLayout_3.addWidget(self.xmin_value)
        self.xmax_label = QtWidgets.QLabel(self.frame_3)
        self.xmax_label.setObjectName("xmax_label")
        self.horizontalLayout_3.addWidget(self.xmax_label)
        self.xmax_value = QtWidgets.QLineEdit(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.xmax_value.sizePolicy().hasHeightForWidth())
        self.xmax_value.setSizePolicy(sizePolicy)
        self.xmax_value.setObjectName("xmax_value")
        self.horizontalLayout_3.addWidget(self.xmax_value)
        self.verticalLayout_4.addWidget(self.frame_3)
        self.horizontalLayout.addWidget(self.window_frame)
        Formula.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Formula)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 923, 22))
        self.menubar.setObjectName("menubar")
        Formula.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Formula)
        self.statusbar.setObjectName("statusbar")
        Formula.setStatusBar(self.statusbar)

        self.retranslateUi(Formula)
        QtCore.QMetaObject.connectSlotsByName(Formula)

    def retranslateUi(self, Formula):
        _translate = QtCore.QCoreApplication.translate
        Formula.setWindowTitle(_translate("Formula", "Add Material from Formula"))
        self.material_name_label.setText(_translate("Formula", "Material Name:"))
        self.import_button.setText(_translate("Formula", "Import"))
        self.add_db_button.setText(_translate("Formula", "Add to Database"))
        self.left_cb_label.setText(_translate("Formula", "Left Axis:"))
        self.right_axis_label.setText(_translate("Formula", "Right Axis:"))
        self.units_label.setText(_translate("Formula", "Units:"))
        self.xmin_label.setText(_translate("Formula", "Emin (eV)"))
        self.xmax_label.setText(_translate("Formula", "Emax (ev):"))
