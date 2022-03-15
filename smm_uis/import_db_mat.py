# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'import_db_mat.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ImportDB(object):
    def setupUi(self, ImportDB):
        ImportDB.setObjectName("ImportDB")
        ImportDB.resize(614, 606)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        ImportDB.setWindowIcon(icon)
        self.verticalLayout = QtWidgets.QVBoxLayout(ImportDB)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_2 = QtWidgets.QFrame(ImportDB)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_2.setLineWidth(0)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout.setObjectName("gridLayout")
        self.choose_file_button = QtWidgets.QPushButton(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.choose_file_button.sizePolicy().hasHeightForWidth())
        self.choose_file_button.setSizePolicy(sizePolicy)
        self.choose_file_button.setObjectName("choose_file_button")
        self.gridLayout.addWidget(self.choose_file_button, 0, 0, 1, 1)
        self.chosen_file_label = QtWidgets.QLabel(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.chosen_file_label.sizePolicy().hasHeightForWidth())
        self.chosen_file_label.setSizePolicy(sizePolicy)
        self.chosen_file_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.chosen_file_label.setFrameShadow(QtWidgets.QFrame.Raised)
        self.chosen_file_label.setText("")
        self.chosen_file_label.setAlignment(QtCore.Qt.AlignCenter)
        self.chosen_file_label.setObjectName("chosen_file_label")
        self.gridLayout.addWidget(self.chosen_file_label, 0, 2, 1, 1)
        self.mat_name_label = QtWidgets.QLabel(self.frame_2)
        self.mat_name_label.setObjectName("mat_name_label")
        self.gridLayout.addWidget(self.mat_name_label, 1, 0, 1, 1)
        self.mat_name_edit = QtWidgets.QLineEdit(self.frame_2)
        self.mat_name_edit.setObjectName("mat_name_edit")
        self.gridLayout.addWidget(self.mat_name_edit, 1, 2, 1, 1)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(ImportDB)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(6)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_2.setContentsMargins(-1, 15, -1, -1)
        self.verticalLayout_2.setSpacing(15)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_4 = QtWidgets.QFrame(self.frame_3)
        self.frame_4.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_2.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.delimiter_label = QtWidgets.QLabel(self.frame_4)
        self.delimiter_label.setObjectName("delimiter_label")
        self.horizontalLayout_2.addWidget(self.delimiter_label)
        self.comma_cb = QtWidgets.QCheckBox(self.frame_4)
        self.comma_cb.setObjectName("comma_cb")
        self.delimiter_group = QtWidgets.QButtonGroup(ImportDB)
        self.delimiter_group.setObjectName("delimiter_group")
        self.delimiter_group.addButton(self.comma_cb)
        self.horizontalLayout_2.addWidget(self.comma_cb)
        self.space_cb = QtWidgets.QCheckBox(self.frame_4)
        self.space_cb.setChecked(True)
        self.space_cb.setObjectName("space_cb")
        self.delimiter_group.addButton(self.space_cb)
        self.horizontalLayout_2.addWidget(self.space_cb)
        self.comma_dot_cb = QtWidgets.QCheckBox(self.frame_4)
        self.comma_dot_cb.setMaximumSize(QtCore.QSize(50, 16777215))
        self.comma_dot_cb.setObjectName("comma_dot_cb")
        self.delimiter_group.addButton(self.comma_dot_cb)
        self.horizontalLayout_2.addWidget(self.comma_dot_cb)
        self.other_cb = QtWidgets.QCheckBox(self.frame_4)
        self.other_cb.setObjectName("other_cb")
        self.delimiter_group.addButton(self.other_cb)
        self.horizontalLayout_2.addWidget(self.other_cb)
        self.other_edit = QtWidgets.QLineEdit(self.frame_4)
        self.other_edit.setEnabled(False)
        self.other_edit.setReadOnly(False)
        self.other_edit.setObjectName("other_edit")
        self.horizontalLayout_2.addWidget(self.other_edit)
        self.verticalLayout_2.addWidget(self.frame_4)
        self.frame_5 = QtWidgets.QFrame(self.frame_3)
        self.frame_5.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_3.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.decimal_label = QtWidgets.QLabel(self.frame_5)
        self.decimal_label.setObjectName("decimal_label")
        self.horizontalLayout_3.addWidget(self.decimal_label)
        self.dot_dec_cb = QtWidgets.QCheckBox(self.frame_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dot_dec_cb.sizePolicy().hasHeightForWidth())
        self.dot_dec_cb.setSizePolicy(sizePolicy)
        self.dot_dec_cb.setChecked(True)
        self.dot_dec_cb.setObjectName("dot_dec_cb")
        self.decimal_group = QtWidgets.QButtonGroup(ImportDB)
        self.decimal_group.setObjectName("decimal_group")
        self.decimal_group.addButton(self.dot_dec_cb)
        self.horizontalLayout_3.addWidget(self.dot_dec_cb)
        self.comma_dec_cb = QtWidgets.QCheckBox(self.frame_5)
        self.comma_dec_cb.setObjectName("comma_dec_cb")
        self.decimal_group.addButton(self.comma_dec_cb)
        self.horizontalLayout_3.addWidget(self.comma_dec_cb)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.ignore_lines_label = QtWidgets.QLabel(self.frame_5)
        self.ignore_lines_label.setObjectName("ignore_lines_label")
        self.horizontalLayout_3.addWidget(self.ignore_lines_label)
        self.ignore_lines_edit = QtWidgets.QLineEdit(self.frame_5)
        self.ignore_lines_edit.setEnabled(True)
        self.ignore_lines_edit.setInputMask("")
        self.ignore_lines_edit.setAlignment(QtCore.Qt.AlignCenter)
        self.ignore_lines_edit.setObjectName("ignore_lines_edit")
        self.horizontalLayout_3.addWidget(self.ignore_lines_edit)
        self.verticalLayout_2.addWidget(self.frame_5)
        self.tableView = QtWidgets.QTableView(self.frame_3)
        self.tableView.setObjectName("tableView")
        self.verticalLayout_2.addWidget(self.tableView)
        self.verticalLayout.addWidget(self.frame_3)
        self.frame = QtWidgets.QFrame(ImportDB)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setLineWidth(0)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.units_label = QtWidgets.QLabel(self.frame)
        self.units_label.setObjectName("units_label")
        self.horizontalLayout.addWidget(self.units_label)
        self.unit_combobox = QtWidgets.QComboBox(self.frame)
        self.unit_combobox.setObjectName("unit_combobox")
        self.horizontalLayout.addWidget(self.unit_combobox)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.import_button = QtWidgets.QPushButton(self.frame)
        self.import_button.setObjectName("import_button")
        self.horizontalLayout.addWidget(self.import_button)
        self.preview_button = QtWidgets.QPushButton(self.frame)
        self.preview_button.setObjectName("preview_button")
        self.horizontalLayout.addWidget(self.preview_button)
        self.verticalLayout.addWidget(self.frame)

        self.retranslateUi(ImportDB)
        QtCore.QMetaObject.connectSlotsByName(ImportDB)

    def retranslateUi(self, ImportDB):
        _translate = QtCore.QCoreApplication.translate
        ImportDB.setWindowTitle(_translate("ImportDB", "Import Material"))
        self.choose_file_button.setText(_translate("ImportDB", "Choose File"))
        self.mat_name_label.setText(_translate("ImportDB", "Material Name"))
        self.delimiter_label.setText(_translate("ImportDB", "Delimiter:"))
        self.comma_cb.setText(_translate("ImportDB", "comma (,)"))
        self.space_cb.setText(_translate("ImportDB", "space ( )"))
        self.comma_dot_cb.setText(_translate("ImportDB", ";"))
        self.other_cb.setText(_translate("ImportDB", "other"))
        self.decimal_label.setText(_translate("ImportDB", "Decimal:"))
        self.dot_dec_cb.setText(_translate("ImportDB", "Dot (.)"))
        self.comma_dec_cb.setText(_translate("ImportDB", "Comma (,)"))
        self.ignore_lines_label.setText(_translate("ImportDB", "Ignore Lines"))
        self.ignore_lines_edit.setText(_translate("ImportDB", "0"))
        self.units_label.setText(_translate("ImportDB", "Units:"))
        self.import_button.setText(_translate("ImportDB", "Import"))
        self.preview_button.setText(_translate("ImportDB", "Preview"))
