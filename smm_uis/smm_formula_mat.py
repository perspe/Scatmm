# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'formula_mat.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Formula(object):
    def setupUi(self, Formula):
        Formula.setObjectName("Formula")
        Formula.resize(910, 566)
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
        self.variable_layout = QtWidgets.QVBoxLayout()
        self.variable_layout.setSpacing(2)
        self.variable_layout.setObjectName("variable_layout")
        self.verticalLayout_3.addLayout(self.variable_layout)
        self.verticalLayout.addWidget(self.variables_frame)
        self.add_db_button = QtWidgets.QPushButton(self.frame)
        self.add_db_button.setObjectName("add_db_button")
        self.verticalLayout.addWidget(self.add_db_button)
        self.horizontalLayout.addWidget(self.frame)
        self.window_widget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.window_widget.sizePolicy().hasHeightForWidth())
        self.window_widget.setSizePolicy(sizePolicy)
        self.window_widget.setMinimumSize(QtCore.QSize(500, 500))
        self.window_widget.setObjectName("window_widget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.window_widget)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.plot_layout = QtWidgets.QVBoxLayout()
        self.plot_layout.setSpacing(0)
        self.plot_layout.setObjectName("plot_layout")
        self.verticalLayout_4.addLayout(self.plot_layout)
        self.horizontalLayout.addWidget(self.window_widget)
        Formula.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Formula)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 910, 22))
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
        self.add_db_button.setText(_translate("Formula", "Add to Database"))

