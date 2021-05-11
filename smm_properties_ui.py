# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Designer_UIs/properties_ui.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Properties(object):
    def setupUi(self, Properties):
        Properties.setObjectName("Properties")
        Properties.resize(384, 419)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Properties.setWindowIcon(icon)
        self.verticalLayout = QtWidgets.QVBoxLayout(Properties)
        self.verticalLayout.setContentsMargins(20, 20, 20, 20)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(Properties)
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.properties_wavelenght_label = QtWidgets.QLabel(self.groupBox)
        self.properties_wavelenght_label.setObjectName("properties_wavelenght_label")
        self.horizontalLayout_2.addWidget(self.properties_wavelenght_label)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.prop_wav_points = QtWidgets.QLineEdit(self.groupBox)
        self.prop_wav_points.setAlignment(QtCore.Qt.AlignCenter)
        self.prop_wav_points.setObjectName("prop_wav_points")
        self.horizontalLayout_2.addWidget(self.prop_wav_points)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(Properties)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_2.setFlat(False)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout.setVerticalSpacing(2)
        self.gridLayout.setObjectName("gridLayout")
        self.prop_iter_num = QtWidgets.QLineEdit(self.groupBox_2)
        self.prop_iter_num.setAlignment(QtCore.Qt.AlignCenter)
        self.prop_iter_num.setObjectName("prop_iter_num")
        self.gridLayout.addWidget(self.prop_iter_num, 0, 2, 1, 1)
        self.prop_cog1_label = QtWidgets.QLabel(self.groupBox_2)
        self.prop_cog1_label.setObjectName("prop_cog1_label")
        self.gridLayout.addWidget(self.prop_cog1_label, 3, 0, 1, 1)
        self.prop_num_particles = QtWidgets.QLineEdit(self.groupBox_2)
        self.prop_num_particles.setAlignment(QtCore.Qt.AlignCenter)
        self.prop_num_particles.setObjectName("prop_num_particles")
        self.gridLayout.addWidget(self.prop_num_particles, 1, 2, 1, 1)
        self.prop_w = QtWidgets.QLineEdit(self.groupBox_2)
        self.prop_w.setAlignment(QtCore.Qt.AlignCenter)
        self.prop_w.setObjectName("prop_w")
        self.gridLayout.addWidget(self.prop_w, 2, 2, 1, 1)
        self.prop_particle_num_label = QtWidgets.QLabel(self.groupBox_2)
        self.prop_particle_num_label.setObjectName("prop_particle_num_label")
        self.gridLayout.addWidget(self.prop_particle_num_label, 1, 0, 1, 1)
        self.prop_iter_num_label = QtWidgets.QLabel(self.groupBox_2)
        self.prop_iter_num_label.setObjectName("prop_iter_num_label")
        self.gridLayout.addWidget(self.prop_iter_num_label, 0, 0, 1, 1)
        self.prop_w_label = QtWidgets.QLabel(self.groupBox_2)
        self.prop_w_label.setObjectName("prop_w_label")
        self.gridLayout.addWidget(self.prop_w_label, 2, 0, 1, 1)
        self.prop_cog1 = QtWidgets.QLineEdit(self.groupBox_2)
        self.prop_cog1.setAlignment(QtCore.Qt.AlignCenter)
        self.prop_cog1.setObjectName("prop_cog1")
        self.gridLayout.addWidget(self.prop_cog1, 3, 2, 1, 1)
        self.prop_cog2 = QtWidgets.QLineEdit(self.groupBox_2)
        self.prop_cog2.setAlignment(QtCore.Qt.AlignCenter)
        self.prop_cog2.setObjectName("prop_cog2")
        self.gridLayout.addWidget(self.prop_cog2, 5, 2, 1, 1)
        self.prop_cog2_label = QtWidgets.QLabel(self.groupBox_2)
        self.prop_cog2_label.setObjectName("prop_cog2_label")
        self.gridLayout.addWidget(self.prop_cog2_label, 5, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 1, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.frame = QtWidgets.QFrame(Properties)
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setLineWidth(0)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.save_default_button = QtWidgets.QPushButton(self.frame)
        self.save_default_button.setObjectName("save_default_button")
        self.horizontalLayout.addWidget(self.save_default_button)
        self.properties_apply_button = QtWidgets.QPushButton(self.frame)
        self.properties_apply_button.setObjectName("properties_apply_button")
        self.horizontalLayout.addWidget(self.properties_apply_button)
        self.properties_close_button = QtWidgets.QPushButton(self.frame)
        self.properties_close_button.setObjectName("properties_close_button")
        self.horizontalLayout.addWidget(self.properties_close_button)
        self.verticalLayout.addWidget(self.frame)

        self.retranslateUi(Properties)
        QtCore.QMetaObject.connectSlotsByName(Properties)

    def retranslateUi(self, Properties):
        _translate = QtCore.QCoreApplication.translate
        Properties.setWindowTitle(_translate("Properties", "Properties"))
        self.groupBox.setTitle(_translate("Properties", "Simulation"))
        self.properties_wavelenght_label.setText(_translate("Properties", "Wavelenght Points:"))
        self.prop_wav_points.setText(_translate("Properties", "200"))
        self.groupBox_2.setTitle(_translate("Properties", "Particle Swarm"))
        self.prop_iter_num.setText(_translate("Properties", "30"))
        self.prop_cog1_label.setText(_translate("Properties", "Individual Cognition Factor:"))
        self.prop_num_particles.setText(_translate("Properties", "25"))
        self.prop_w.setText(_translate("Properties", "0.55"))
        self.prop_particle_num_label.setText(_translate("Properties", "Number of Particles:"))
        self.prop_iter_num_label.setText(_translate("Properties", "Number of Iterations:"))
        self.prop_w_label.setText(_translate("Properties", "Inertial Weight Factor:"))
        self.prop_cog1.setText(_translate("Properties", "1.2"))
        self.prop_cog2.setText(_translate("Properties", "1.9"))
        self.prop_cog2_label.setText(_translate("Properties", "Social Cognition Factor:"))
        self.save_default_button.setText(_translate("Properties", "Save as default"))
        self.properties_apply_button.setText(_translate("Properties", "Apply"))
        self.properties_close_button.setText(_translate("Properties", "Close"))
