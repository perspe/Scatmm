# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'view_database.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Database(object):
    def setupUi(self, Database):
        Database.setObjectName("Database")
        Database.resize(1245, 550)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Database.sizePolicy().hasHeightForWidth())
        Database.setSizePolicy(sizePolicy)
        Database.setMinimumSize(QtCore.QSize(1200, 550))
        Database.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        Database.setAcceptDrops(True)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Database.setWindowIcon(icon)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Database)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(Database)
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
        self.add_material = QtWidgets.QPushButton(self.frame)
        self.add_material.setObjectName("add_material")
        self.verticalLayout.addWidget(self.add_material)
        self.add_formula = QtWidgets.QPushButton(self.frame)
        self.add_formula.setObjectName("add_formula")
        self.verticalLayout.addWidget(self.add_formula)
        self.rmv_material = QtWidgets.QPushButton(self.frame)
        self.rmv_material.setObjectName("rmv_material")
        self.verticalLayout.addWidget(self.rmv_material)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout.addWidget(self.frame)
        self.frame_2 = QtWidgets.QFrame(Database)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.table_layout = QtWidgets.QVBoxLayout()
        self.table_layout.setObjectName("table_layout")
        self.verticalLayout_3.addLayout(self.table_layout)
        self.horizontalLayout.addWidget(self.frame_2)
        self.widget = QtWidgets.QWidget(Database)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(4)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(500, 500))
        self.widget.setMouseTracking(False)
        self.widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.widget.setObjectName("widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.figure_widget = QtWidgets.QVBoxLayout()
        self.figure_widget.setObjectName("figure_widget")
        self.horizontalLayout_2.addLayout(self.figure_widget)
        self.horizontalLayout.addWidget(self.widget)

        self.retranslateUi(Database)
        QtCore.QMetaObject.connectSlotsByName(Database)

    def retranslateUi(self, Database):
        _translate = QtCore.QCoreApplication.translate
        Database.setWindowTitle(_translate("Database", "Manage Database"))
        self.add_material.setText(_translate("Database", "Add Material"))
        self.add_formula.setText(_translate("Database", "Add from Formula"))
        self.rmv_material.setText(_translate("Database", "Remove Material"))
