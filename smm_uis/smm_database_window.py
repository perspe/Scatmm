# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Designer_UIs/view_database.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Database(object):
    def setupUi(self, Database):
        Database.setObjectName("Database")
        Database.resize(621, 286)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Database.sizePolicy().hasHeightForWidth())
        Database.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Database.setWindowIcon(icon)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Database)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.database_table = QtWidgets.QTableView(Database)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.database_table.sizePolicy().hasHeightForWidth())
        self.database_table.setSizePolicy(sizePolicy)
        self.database_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.database_table.setObjectName("database_table")
        self.database_table.horizontalHeader().setDefaultSectionSize(150)
        self.horizontalLayout.addWidget(self.database_table)
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
        self.rmv_material = QtWidgets.QPushButton(self.frame)
        self.rmv_material.setObjectName("rmv_material")
        self.verticalLayout.addWidget(self.rmv_material)
        self.view_material = QtWidgets.QPushButton(self.frame)
        self.view_material.setObjectName("view_material")
        self.verticalLayout.addWidget(self.view_material)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout.addWidget(self.frame)

        self.retranslateUi(Database)
        QtCore.QMetaObject.connectSlotsByName(Database)

    def retranslateUi(self, Database):
        _translate = QtCore.QCoreApplication.translate
        Database.setWindowTitle(_translate("Database", "Manage Database"))
        self.add_material.setText(_translate("Database", "Add Material"))
        self.rmv_material.setText(_translate("Database", "Remove Material"))
        self.view_material.setText(_translate("Database", "View Material"))
