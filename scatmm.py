#!/usr/bin/python3
"""
Main script that combines everything from the SMM method
"""

import os
import sys
import math
import webbrowser
import json

import matplotlib.style as mstyle
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox
from scipy.interpolate import interp1d

from Database.database import Database
from modules.pso import particle_swarm
from modules.scattering_matrix import smm_broadband
from modules.fig_class import PltFigure, FigWidget
from smm_export_window import Ui_ExportWindow
from smm_main_window import Ui_SMM_Window
from smm_properties_ui import Ui_Properties
from db_window import DBWindow

VERSION = "1.2.0"

# Default plot properties
mstyle.use("smm_style")


class OptimizeWorkder(QtCore.QThread):
    """
    Worker thread to perform the optimization algorithm
    """
    updateValueSignal = QtCore.pyqtSignal(int)
    updateTextSignal = QtCore.pyqtSignal(str)
    updateOptButton = QtCore.pyqtSignal(bool)

    def __init__(self, figure_handler, particle_info, lmb, compare_data,
                 theta, phi, pol, ref_medium, trn_medium, thickness,
                 e_array, checks):
        super().__init__()
        self.figure_canvas = figure_handler
        self.figure = self.figure_canvas.axes
        self.lmb = lmb
        self.compare_data = compare_data
        self.theta = theta
        self.phi = phi
        self.pol = pol
        self.ref_medium = ref_medium
        self.trn_medium = trn_medium
        self.thickness = thickness
        self.e_array = e_array
        self.ref_check, self.trn_check, self.abs_check = checks
        self.particle_info = {key: info[1]
                              for key, info in particle_info.items()}
        self.iterator = 0

    def run(self):
        """
        Thread initialization to run the optimization algorithm
        """
        # Disable optimization button
        self.updateOptButton.emit(False)

        def optimize_function(**thicknesses):
            t_array = np.stack([thick_i for thick_i in thicknesses.values()])
            ref, trn = np.apply_along_axis(
                lambda thick: smm_broadband(self.theta,
                                            self.phi,
                                            self.e_array,
                                            thick,
                                            self.lmb,
                                            pol=self.pol,
                                            inc_medium=self.ref_medium,
                                            trn_medium=self.trn_medium), 0,
                t_array)
            if self.ref_check:
                point_error = np.sum((ref - self.compare_data)**2, axis=0)
            elif self.trn_check:
                point_error = np.sum((trn - self.compare_data)**2, axis=0)
            elif self.abs_check:
                point_error = np.sum((1 - ref - trn - self.compare_data)**2,
                                     axis=0)
            self.updateValueSignal.emit(
                int(self.iterator / self.particle_info["n_iter"] * 100))
            self.iterator += 1
            return point_error

        # Datastructure for the particle swarm algorithm
        thick = {
            "thick_" + str(i): [thick_i[0], thick_i[1]]
            for i, thick_i in enumerate(self.thickness)
        }
        lowest_error, best_thick, _, _ = particle_swarm(
            optimize_function,
            thick,
            swarm_properties=(self.particle_info["w"],
                              self.particle_info["c1"],
                              self.particle_info["c2"]),
            n_iter=self.particle_info["n_iter"],
            n_particles=self.particle_info["n_particles"],
            maximize=False)

        self.updateValueSignal.emit(0)
        # Update result on the QTextBrowser widget with the best results
        self.updateTextSignal.emit(f"Best error of: {lowest_error:.3g}")
        self.updateOptButton.emit(True)
        for i, t_i in enumerate(best_thick):
            self.updateTextSignal.emit(f"thick {i+1} = {t_i:.3f}")
        # Reinit the graph and plot the best curve
        ref, trn = smm_broadband(self.theta,
                                 self.phi,
                                 self.e_array,
                                 best_thick,
                                 self.lmb,
                                 pol=self.pol,
                                 inc_medium=self.ref_medium,
                                 trn_medium=self.trn_medium)
        if self.ref_check:
            self.figure.plot(self.lmb, ref, ":")
        elif self.trn_check:
            self.figure.plot(self.lmb, trn, ":")
        elif self.abs_check:
            self.figure.plot(self.lmb, 1-trn-ref, ":")
        self.figure_canvas.draw()


class SMMGUI(QMainWindow):
    def __init__(self):
        """
        Initialize necessary variables for the GUI
        """
        super(SMMGUI, self).__init__()
        self.ui = Ui_SMM_Window()
        self.ui.setupUi(self)
        # Initialize the Main Figure
        self.fig_layout = self.ui.figure_layout
        self.main_canvas = PltFigure(self.fig_layout, "Wavelength (nm)",
                                     "R/T/Abs")
        self.addToolBar(QtCore.Qt.TopToolBarArea,
                        NavigationToolbar2QT(self.main_canvas, self))
        # Alias to add plots to the figure
        self.main_figure = self.main_canvas.axes
        # Initialize database
        self.database = Database(os.path.join("Database", "database"))
        # Initialize list with the buttons in the opt and sim tabs
        self.sim_mat = [self.ui.sim_tab_sim_mat1, self.ui.sim_tab_sim_mat2]
        for sim_mat_i in self.sim_mat:
            sim_mat_i.addItems(self.database.content)
        self.sim_mat_size = [
            self.ui.sim_tab_sim_mat_size1, self.ui.sim_tab_sim_mat_size2
        ]
        self.opt_mat = [self.ui.opt_tab_sim_mat1, self.ui.opt_tab_sim_mat2]
        for opt_mat_i in self.opt_mat:
            opt_mat_i.addItems(self.database.content)
        self.opt_mat_size_min = [
            self.ui.opt_tab_sim_mat_size_min1,
            self.ui.opt_tab_sim_mat_size_min2
        ]
        self.opt_mat_size_max = [
            self.ui.opt_tab_sim_mat_size_max1,
            self.ui.opt_tab_sim_mat_size_max2
        ]
        # Check if last plot was from simulation or optimization
        self.last_plot_sim = True
        # List to store the previous simulation results
        self.sim_results = []
        # Store imported data
        self.imported_data = []
        # Load simulation default properties
        with open("config.json", "r") as config:
            self.global_properties = json.load(config)
        # Initialize main dictionaries for simulations
        self.sim_config = {
            "materials": self.sim_mat,
            "size": self.sim_mat_size,
            "ref_n": self.ui.sim_tab_ref_n,
            "ref_k": self.ui.sim_tab_trn_k,
            "trn_n": self.ui.sim_tab_trn_n,
            "trn_k": self.ui.sim_tab_trn_k
        }
        self.opt_config = {
            "materials": self.opt_mat,
            "size_low": self.opt_mat_size_min,
            "size_high": self.opt_mat_size_max,
            "ref_n": self.ui.opt_tab_ref_n,
            "ref_k": self.ui.opt_tab_trn_k,
            "trn_n": self.ui.opt_tab_trn_n,
            "trn_k": self.ui.opt_tab_trn_k
        }
        self.sim_data = {
            "lmb_min": self.ui.sim_param_lmb_min,
            "lmb_max": self.ui.sim_param_lmb_max,
            "theta": self.ui.sim_param_theta,
            "phi": self.ui.sim_param_phi,
            "ref": self.ui.sim_param_ref_checkbox,
            "trn": self.ui.sim_param_trn_checkbox,
            "ptm": self.ui.sim_param_ptm,
            "pte": self.ui.sim_param_pte,
            "abs": self.ui.sim_param_abs_checkbox
        }
        # Initialize all the UI elements
        self.initializeUI()
        self.show()

    def initializeUI(self):
        """
        Associate all UI elements with specific functions
        """
        self.ui.sim_tab_add_layer_button.clicked.connect(
            lambda: self.add_layer("sim"))
        self.ui.opt_tab_add_layer_button.clicked.connect(
            lambda: self.add_layer("opt"))
        self.ui.sim_tab_rem_layer_button.clicked.connect(
            lambda: self.rmv_layer("sim"))
        self.ui.opt_tab_rem_layer_button.clicked.connect(
            lambda: self.rmv_layer("opt"))
        self.ui.sim_tab_sim_button.clicked.connect(self.simulate)
        self.ui.sim_tab_clear_button.clicked.connect(self.clear_sim_buffer)
        self.ui.sim_tab_export_button.clicked.connect(self.export_simulation)
        self.clear_button = self.ui.sim_tab_clear_button
        self.ui.import_button.clicked.connect(self.import_data)
        self.ui.opt_tab_sim_button.clicked.connect(self.pre_optimize_checks)
        # Connect menu buttons
        self.ui.actionView_Database.triggered.connect(self.view_database)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionProperties.triggered.connect(self.open_properties)
        self.ui.actionHelp.triggered.connect(lambda: webbrowser.open_new_tab(
            os.path.join("Help", "help.html")))
        self.ui.actionAbout.triggered.connect(self.aboutDialog)
        # Set default value for progress bar
        self.ui.opt_progressBar.setValue(0)

    def aboutDialog(self):
        """Show the about dialog"""
        title = "About Scatmm"
        msg = "Graphical interface to interact with"\
            "the transfer matrix method (using scattering"\
            f"matrices\n\nAuthor: Miguel Alexandre\n\nVersion: {VERSION}"
        QMessageBox.about(self, title, msg)

    """ Properties Button and associated functions """

    def update_properties(self):
        """
        Update all the properties - Linked to Apply button in properties window
        """
        try:
            for key in self.global_properties.keys():
                if self.global_properties[key][0] == "int":
                    self.global_properties[key] = ["int", int(
                        self.propertie_values[key].text())]
                elif self.global_properties[key][0] == "float":
                    self.global_properties[key] = ["float", float(
                        self.propertie_values[key].text())]
                else:
                    raise Exception("Unknown type format in config.json")
        except ValueError:
            raise Exception("Mistaken type in config file")
        self.properties_window.close()

    def save_default_properties(self):
        """
        Store default properties in config.json file
        """
        self.update_properties()
        with open("config.json", "w") as config:
            json.dump(self.global_properties, config, indent=2)

    def open_properties(self):
        """
        Open a new window showing the properties
        """
        # Load the properties UI
        self.properties_window = QtWidgets.QTabWidget()
        self.properties_ui = Ui_Properties()
        self.properties_ui.setupUi(self.properties_window)
        self.properties_window.show()
        # Create a dictionary linked to the properties values
        self.propertie_values = {
            "n_particles": self.properties_ui.prop_num_particles,
            "n_iter": self.properties_ui.prop_iter_num,
            "w": self.properties_ui.prop_w,
            "c1": self.properties_ui.prop_cog1,
            "c2": self.properties_ui.prop_cog2,
            "sim_points": self.properties_ui.prop_wav_points
        }
        # Load the properties values from the global properties variable
        for key in self.global_properties.keys():
            self.propertie_values[key].setText(str(
                self.global_properties[key][1]))

        # Atribute actions to the different buttons
        self.properties_ui.properties_close_button.clicked.connect(
            self.properties_window.close)
        self.properties_ui.properties_apply_button.clicked.connect(
            self.update_properties)
        self.properties_ui.save_default_button.clicked.connect(
            self.save_default_properties)

    """ Layer Management """

    def add_layer(self, tab):
        """
        Add a new layer (combobox, hspacer and qlineedit) to a specific tab
        """
        font = QtGui.QFont()
        font.setPointSize(11)
        if tab == "sim":
            # Add a new combobox
            self.sim_mat.append(QtWidgets.QComboBox(self.ui.sim_tab_sim_frame))
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                               QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(3)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(
                self.sim_mat[-1].sizePolicy().hasHeightForWidth())
            self.sim_mat[-1].setSizePolicy(sizePolicy)
            self.sim_mat[-1].setFont(font)
            self.sim_mat[-1].setObjectName(
                f"sim_tab_sim_mat{len(self.sim_mat)+1}")
            self.ui.gridLayout_2.addWidget(self.sim_mat[-1], len(self.sim_mat),
                                           0, 1, 1)
            self.sim_mat[-1].addItems(self.database.content)
            self.sim_mat_size.append(
                QtWidgets.QLineEdit(self.ui.sim_tab_sim_frame))
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                               QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(1)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(
                self.sim_mat_size[-1].sizePolicy().hasHeightForWidth())
            self.sim_mat_size[-1].setSizePolicy(sizePolicy)
            self.sim_mat_size[-1].setFont(font)
            self.sim_mat_size[-1].setAlignment(QtCore.Qt.AlignCenter)
            self.sim_mat_size[-1].setObjectName(
                f"sim_tab_sim_mat_size{len(self.sim_mat_size)+1}")
            self.ui.gridLayout_2.addWidget(self.sim_mat_size[-1],
                                           len(self.sim_mat_size), 2, 1, 1)
        else:
            # Add combobox for the material
            self.opt_mat.append(QtWidgets.QComboBox(self.ui.opt_tab_sim_frame))
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                               QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(3)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(
                self.opt_mat[-1].sizePolicy().hasHeightForWidth())
            self.opt_mat[-1].setFont(font)
            self.opt_mat[-1].setSizePolicy(sizePolicy)
            self.opt_mat[-1].setObjectName(
                f"opt_tab_sim_mat{len(self.opt_mat)+1}")
            self.ui.gridLayout_4.addWidget(self.opt_mat[-1], len(self.opt_mat),
                                           0, 1, 1)
            self.opt_mat[-1].addItems(self.database.content)
            # Add QLineEdit for the min size
            self.opt_mat_size_min.append(
                QtWidgets.QLineEdit(self.ui.opt_tab_sim_frame))
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                               QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(1)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(
                self.opt_mat_size_min[-1].sizePolicy().hasHeightForWidth())
            self.opt_mat_size_min[-1].setSizePolicy(sizePolicy)
            self.opt_mat_size_min[-1].setFont(font)
            self.opt_mat_size_min[-1].setAlignment(QtCore.Qt.AlignCenter)
            self.opt_mat_size_min[-1].setObjectName(
                f"opt_tab_sim_mat_size_min{len(self.opt_mat_size_min)+1}")
            self.ui.gridLayout_4.addWidget(self.opt_mat_size_min[-1],
                                           len(self.opt_mat_size_min), 2, 1, 1)
            # Add QLineEdit for the max size
            self.opt_mat_size_max.append(
                QtWidgets.QLineEdit(self.ui.opt_tab_sim_frame))
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                               QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(1)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(
                self.opt_mat_size_max[-1].sizePolicy().hasHeightForWidth())
            self.opt_mat_size_max[-1].setSizePolicy(sizePolicy)
            self.opt_mat_size_max[-1].setAlignment(QtCore.Qt.AlignCenter)
            self.opt_mat_size_max[-1].setFont(font)
            self.opt_mat_size_max[-1].setObjectName(
                f"opt_tab_sim_mat_size_max{len(self.opt_mat_size_max)+1}")
            self.ui.gridLayout_4.addWidget(self.opt_mat_size_max[-1],
                                           len(self.opt_mat_size_max), 3, 1, 1)

    def rmv_layer(self, tab):
        """
        Remove a layer from a specific tab
        """
        if tab == "sim":
            if len(self.sim_mat) == 1:
                QMessageBox.warning(self, "Error: Number of Layers",
                                    "Minimum number of layers is 1!!",
                                    QMessageBox.Close, QMessageBox.Close)
                return
            self.ui.gridLayout_2.removeWidget(self.sim_mat[-1])
            self.sim_mat[-1].deleteLater()
            del self.sim_mat[-1]
            self.ui.gridLayout_2.removeWidget(self.sim_mat_size[-1])
            self.sim_mat_size[-1].deleteLater()
            del self.sim_mat_size[-1]
        else:
            if len(self.opt_mat) == 1:
                QMessageBox.warning(self, "Error: Number of Layers",
                                    "Minimum number of layers is 1!!",
                                    QMessageBox.Close, QMessageBox.Close)
                return
            self.ui.gridLayout_4.removeWidget(self.opt_mat[-1])
            self.opt_mat[-1].deleteLater()
            del self.opt_mat[-1]
            self.ui.gridLayout_4.removeWidget(self.opt_mat_size_min[-1])
            self.opt_mat_size_min[-1].deleteLater()
            del self.opt_mat_size_min[-1]
            self.ui.gridLayout_4.removeWidget(self.opt_mat_size_max[-1])
            self.opt_mat_size_max[-1].deleteLater()
            del self.opt_mat_size_max[-1]

    """ Aliases to get information necessary for simulation/optimization """

    def get_sim_data(self):
        """
        Get simulation configuration
        """
        try:
            theta = np.radians(float(self.sim_data["theta"].text()))
            phi = np.radians(float(self.sim_data["phi"].text()))
            # TODO: Normalize the polarization vector <08-10-21, Miguel> #
            pol = (complex(self.sim_data["pte"].text()),
                   complex(self.sim_data["ptm"].text()))
            lmb_min = float(self.sim_data["lmb_min"].text())
            lmb_max = float(self.sim_data["lmb_max"].text())
            return theta, phi, pol, lmb_min, lmb_max
        except ValueError:
            title = "Error: Invalid number"
            message = "Invalid simulation parameter"
            QMessageBox.warning(self, title, message, QMessageBox.Close,
                                QMessageBox.Close)
            raise ValueError

    def get_medium_config(self, data_structure):
        """
        Get data for the transmission and reflection media
        """
        try:
            ref_medium_n = float(data_structure["ref_n"].text())
            ref_medium_k = float(data_structure["ref_k"].text())
            ref_medium = ((ref_medium_n + 1j * ref_medium_k)**2, 1)
            trn_medium_n = float(data_structure["trn_n"].text())
            trn_medium_k = float(data_structure["trn_k"].text())
            trn_medium = ((trn_medium_n + 1j * trn_medium_k)**2, 1)
            return ref_medium, trn_medium
        except ValueError:
            title = "Error: Invalid number"
            message = "Invalid parameter in Ref/Transmission regions"
            QMessageBox.warning(self, title, message, QMessageBox.Close,
                                QMessageBox.Close)
            raise ValueError

    def get_material_config(self, lmb, material_structure, tab="sim"):
        """
        Get material configuration
        """
        try:
            thick = []
            material_i = []
            # Get results depending on the respective tab
            if tab == "sim":
                for material, thickness in zip(material_structure["materials"],
                                               material_structure["size"]):
                    thick.append(float(thickness.text()))
                    material_i.append(material.currentText())
                thick = np.array(thick)
            else:
                for material, low_t, upp_t in zip(
                        material_structure["materials"],
                        material_structure["size_low"],
                        material_structure["size_high"]):
                    thick.append([float(low_t.text()), float(upp_t.text())])
                    material_i.append(material.currentText())
                thick = np.array(thick)
            # Build the interpolated n and k arrays from the input materials
            n_array = []
            k_array = []
            for mat_i in material_i:
                db_data = self.database[mat_i]
                interp_n_array = interp1d(db_data[:, 0],
                                          db_data[:, 1],
                                          kind="cubic")
                interp_k_array = interp1d(db_data[:, 0],
                                          db_data[:, 2],
                                          kind="cubic")
                n_array.append(interp_n_array(lmb))
                k_array.append(interp_k_array(lmb))
            n_array = np.stack(n_array)
            k_array = np.stack(k_array)
            # Create the e_array
            e_array = (n_array + 1j * k_array)**2
            return thick, e_array.T, material_i
        except ValueError:
            title = "Error: Material Out of Bounds"
            message = "One of the materials in the simulation is"\
                "undefined for the defined wavelength range"
            QMessageBox.warning(self, title, message, QMessageBox.Close,
                                QMessageBox.Close)
            raise ValueError

    """ Simulate and plot simulation data """

    def simulate(self):
        """
        Script that outputs all the data necessary to run smm_broadband
        """
        # Get all the sim_data values
        try:
            theta, phi, pol, lmb_min, lmb_max = self.get_sim_data()
            lmb = np.linspace(lmb_min, lmb_max,
                              self.global_properties["sim_points"][1])
            # Get all the sim_config_values
            ref_medium, trn_medium = self.get_medium_config(self.sim_config)
            thick, e_array, mat = self.get_material_config(
                lmb, self.sim_config)
            ref, trn = smm_broadband(theta, phi, e_array, thick, lmb, pol,
                                     ref_medium, trn_medium)
            self.sim_results_update(theta, phi, mat, thick, lmb, ref, trn)
            self.sim_plot_data(lmb, ref, trn)
            return ref, trn
        except ValueError:
            pass

    def sim_plot_data(self, lmb, ref, trn):
        """
        Verify which checkboxes are toggled and plot data accordingly
        """
        # Check what data to plot
        ref_check = self.sim_data["ref"].isChecked()
        trn_check = self.sim_data["trn"].isChecked()
        abs_check = self.sim_data["abs"].isChecked()
        simulations = str(len(self.sim_results))
        # Plot according to the checkboxes
        if ref_check:
            self.main_figure.plot(lmb, ref, label="R Sim(" + simulations + ")")
        if trn_check:
            self.main_figure.plot(lmb, trn, label="T Sim(" + simulations + ")")
        if abs_check:
            self.main_figure.plot(lmb, 1 - ref - trn,
                                  label="A Sim(" + simulations + ")")
        self.main_canvas.draw()

    def sim_results_update(self, theta, phi, mat, thick, lmb, ref, trn):
        """
        Update the simulation results
        """
        ident_string = "(" + str(round(math.degrees(theta), 2)) + "," + str(
            round(math.degrees(phi), 2)) + ") "
        for mat_i, thick_i in zip(mat, thick):
            ident_string += "|" + mat_i[:9] + "(" + str(thick_i) + ")"
        self.sim_results.append((ident_string, lmb, ref, trn))
        self.clear_button.setText("Clear (" + str(len(self.sim_results)) + ")")

    def clear_sim_buffer(self):
        """
        Clear all stored simulation values and destroy all open plots
        """
        self.sim_results = []
        self.main_canvas.reinit()
        self.clear_button.setText("Clear")

    """ Export button and associated functions """

    def export_simulation(self):
        """
        Open a new window displaying all the simulated results
        Choose outputs and then ask for directory to output information
        """
        self.export_window = QtWidgets.QTabWidget()
        self.export_ui = Ui_ExportWindow()
        self.export_ui.setupUi(self.export_window)
        self.export_window.show()
        self.chosen_sim = self.export_ui.simulations_combobox
        self.export_checks = [
            self.export_ui.reflection_checkbox,
            self.export_ui.transmission_checkbox,
            self.export_ui.absorption_checkbox
        ]
        export_names = [sim_name for sim_name, _, _, _ in self.sim_results]
        self.chosen_sim.addItems(export_names)
        self.export_ui.export_all_button.clicked.connect(self.export_all)
        self.export_ui.export_button.clicked.connect(self.export)
        self.export_ui.preview_button.clicked.connect(self.preview_export)

    def export(self):
        """
        Export chosen simulation
        """
        savepath = QFileDialog.getSaveFileName(self, "Save File", "",
                                               "Text Files (*.txt)")
        sim_choice = self.chosen_sim.currentText()
        ref_check = self.export_checks[0].isChecked()
        trn_check = self.export_checks[1].isChecked()
        abs_check = self.export_checks[2].isChecked()
        if savepath[0] == "":
            return
        for name, lmb, ref, trn in self.sim_results:
            if name == sim_choice:
                if ref_check and trn_check and abs_check:
                    export_array = np.stack((lmb, ref, trn, 1 - ref - trn),
                                            axis=-1)
                    np.savetxt(savepath[0],
                               export_array,
                               header='Wavelength Ref Trn Abs')
                elif ref_check and trn_check and not abs_check:
                    export_array = np.stack((lmb, ref, trn), axis=-1)
                    np.savetxt(savepath[0],
                               export_array,
                               header='Wavelength Ref Trn')
                elif ref_check and not trn_check and not abs_check:
                    export_array = np.stack((lmb, ref), axis=-1)
                    np.savetxt(savepath[0],
                               export_array,
                               header='Wavelength Ref')
                elif not ref_check and trn_check and abs_check:
                    export_array = np.stack((lmb, trn, 1 - ref - trn), axis=-1)
                    np.savetxt(savepath[0],
                               export_array,
                               header='Wavelength Trn Abs')
                elif not ref_check and not trn_check and abs_check:
                    export_array = np.stack((lmb, 1 - ref - trn), axis=-1)
                    np.savetxt(savepath[0],
                               export_array,
                               header='Wavelength Abs')
                elif not ref_check and trn_check and not abs_check:
                    export_array = np.stack((lmb, trn), axis=-1)
                    np.savetxt(savepath[0],
                               export_array,
                               header='Wavelength Trn')
        self.export_window.raise_()
        self.export_window.setFocus(True)
        self.export_window.activateWindow()
        QMessageBox.information(self, "Successful Export",
                                "Results Exported Successfully!",
                                QMessageBox.Ok, QMessageBox.Ok)

    def export_all(self):
        """
        Export all simulations
        """
        savepath = QFileDialog.getSaveFileName(self, "Save File", "",
                                               "Text Files (*.txt)")
        if savepath[0] == '':
            return
        export_name = os.path.basename(savepath[0])
        export_dir = os.path.dirname(savepath[0])
        ref_check = self.export_checks[0].isChecked()
        trn_check = self.export_checks[1].isChecked()
        abs_check = self.export_checks[2].isChecked()
        if ref_check and trn_check and abs_check:
            for i, (_, lmb, ref, trn) in enumerate(self.sim_results):
                export_path = os.path.join(export_dir,
                                           "Sim" + str(i) + "_" + export_name)
                export_array = np.stack((lmb, ref, trn, 1 - ref - trn),
                                        axis=-1)
                np.savetxt(export_path,
                           export_array,
                           header='Wavelength Ref Trn Abs')
        elif ref_check and trn_check and not abs_check:
            for i, (_, lmb, ref, trn) in enumerate(self.sim_results):
                export_path = os.path.join(export_dir,
                                           "Sim" + str(i) + "_" + export_name)
                export_array = np.stack((lmb, ref, trn), axis=-1)
                np.savetxt(export_path,
                           export_array,
                           header='Wavelength Ref Trn')
        elif ref_check and not trn_check and not abs_check:
            for i, (_, lmb, ref, trn) in enumerate(self.sim_results):
                export_path = os.path.join(export_dir,
                                           "Sim" + str(i) + "_" + export_name)
                export_array = np.stack((lmb, ref), axis=-1)
                np.savetxt(export_path, export_array, header='Wavelength Ref')
        elif not ref_check and trn_check and abs_check:
            for i, (_, lmb, ref, trn) in enumerate(self.sim_results):
                export_path = os.path.join(export_dir,
                                           "Sim" + str(i) + "_" + export_name)
                export_array = np.stack((lmb, trn, 1 - ref - trn), axis=-1)
                np.savetxt(export_path,
                           export_array,
                           header='Wavelength Trn Abs')
        elif not ref_check and not trn_check and abs_check:
            for i, (_, lmb, ref, trn) in enumerate(self.sim_results):
                export_path = os.path.join(export_dir,
                                           "Sim" + str(i) + "_" + export_name)
                export_array = np.stack((lmb, 1 - ref - trn), axis=-1)
                np.savetxt(export_path, export_array, header='Wavelength Abs')
        elif not ref_check and trn_check and not abs_check:
            for i, (_, lmb, ref, trn) in enumerate(self.sim_results):
                export_path = os.path.join(export_dir,
                                           "Sim" + str(i) + "_" + export_name)
                export_array = np.stack((lmb, trn), axis=-1)
                np.savetxt(export_path, export_array, header='Wavelength Trn')

        self.export_window.close()
        QMessageBox.information(self, "Successful Export",
                                "Results Exported Successfully!",
                                QMessageBox.Ok, QMessageBox.Ok)

    def preview_export(self):
        """
        Preview the selected simulation chosen in the combobox
        """
        sim_choice = self.chosen_sim.currentText()
        self.preview_window = FigWidget(f"Preview {sim_choice}")
        self.preview_fig = PltFigure(self.preview_window.layout,
                                     "Wavelength (nm)", "R/T/Abs", width=7)
        self.preview_window.show()
        for name, lmb, ref, trn in self.sim_results:
            if name == sim_choice:
                self.preview_fig.axes.plot(lmb, ref, label="Ref")
                self.preview_fig.axes.plot(lmb, trn, label="Trn")
                self.preview_fig.axes.plot(lmb, 1-ref-trn, label="Abs")
        self.preview_fig.axes.legend(bbox_to_anchor=(1, 1), loc="upper left")
        self.preview_fig.draw()

    """ Optimization functions """

    def update_progress_bar(self, value):
        """Connect the progress bar with the worker thread"""
        return self.ui.opt_progressBar.setValue(value)

    def updateTextBrower(self, text):
        """Connects the textBrowser with the worker thread"""
        return self.ui.opt_res_text.append(text)

    def updateOptButton(self, status):
        """Connects the Optimization button with the worker thread"""
        return self.ui.opt_tab_sim_button.setEnabled(status)

    def pre_optimize_checks(self):
        """
        Perform checks before going into the optimization loop
        """
        self.ui.opt_res_text.clear()
        # Check if only one optimization option is chosen
        title = "Bad Optimization option"
        msg1 = "Only one of Absorption/Reflection/Transmission"\
            "options can be chosen for optimization"
        msg2 = "One of the Absorption/Reflection/Transmission"\
            "options must be chosen for optimization"
        ref_check = self.sim_data["ref"].isChecked()
        trn_check = self.sim_data["trn"].isChecked()
        abs_check = self.sim_data["abs"].isChecked()
        # Check that only one option of Abs/Ref/Trn is chosen
        if ref_check + trn_check + abs_check > 1:
            QMessageBox.warning(self, title, msg1, QMessageBox.Close,
                                QMessageBox.Close)
            self.ui.opt_res_text.append("Optimization Failed")
            return
        elif ref_check + trn_check + abs_check == 0:
            QMessageBox.warning(self, title, msg2, QMessageBox.Close,
                                QMessageBox.Close)
            self.ui.opt_res_text.append("Optimization Failed")
            return
        self.optimize(ref_check, trn_check, abs_check)

    def optimize(self, ref_check, trn_check, abs_check):
        """
        Perform Results Optimization (Splits into a new worker thread)
        """
        try:
            # Get the data for optimization
            lmb = self.imported_data[:, 0].T
            compare_data = self.imported_data[:, 1][:, np.newaxis]
            self.main_canvas.reinit()
            self.main_figure.plot(lmb, compare_data, "-.")
            self.main_canvas.draw()
            theta, phi, pol, _, _ = self.get_sim_data()
            ref_medium, trn_medium = self.get_medium_config(self.opt_config)
            thick, e_array, _ = self.get_material_config(lmb,
                                                         self.opt_config,
                                                         tab="opt")
            # Create a new worker thread to do the optimization
            self.ui.opt_res_text.clear()
            self.ui.opt_res_text.append("Simulation started...")
            self.opt_worker = OptimizeWorkder(self.main_canvas,
                                              self.global_properties,
                                              lmb,
                                              compare_data,
                                              theta,
                                              phi,
                                              pol,
                                              ref_medium,
                                              trn_medium,
                                              thick,
                                              e_array,
                                              (ref_check,
                                                  trn_check,
                                                  abs_check))
            # Connect the new thread with functions from the main thread
            self.opt_worker.updateValueSignal.connect(self.update_progress_bar)
            self.opt_worker.updateTextSignal.connect(self.updateTextBrower)
            self.opt_worker.updateOptButton.connect(self.updateOptButton)
            # Start worker
            self.opt_worker.start()
        except ValueError:
            self.ui.opt_res_text.append("Optimization Failed")
        except TypeError:
            QMessageBox.warning(self, "Import Data missing",
                                "Missing data to perform optimization",
                                QMessageBox.Close, QMessageBox.Close)
            self.ui.opt_res_text.append("Optimization Failed")

    """ Open the Database Window """

    def view_database(self):
        """
        Open a new window to see all the database materials
        """
        self.db_ui = DBWindow(self, self.database)
        self.db_ui.show()

    """ Import data from file """

    def import_data(self):
        """
        Function for import button - import data for simulation/optimization
        """
        filepath = QFileDialog.getOpenFileName(self, 'Open File')
        if filepath[0] == '':
            return
        try:
            data = self.get_data_from_file(filepath[0])
        except ValueError:
            return
        self.imported_data = data[:, [0, 1]]
        self.main_figure.plot(data[:, 0], data[:, 1],
                              '--', label="Import Data")
        self.main_canvas.draw()

    """ Generic function to import data from file """

    def get_data_from_file(self, filepath):
        """
        Get data from file and return it as numpy array
        """
        title = "Invalid Import Format"
        msg = "The imported data has an unacceptable format"
        try:
            data_df = pd.read_csv(filepath, sep="[ ;,:\t]")
            data = data_df.values
        except ValueError:
            QMessageBox.warning(self, title, msg, QMessageBox.Close,
                                QMessageBox.Close)
            raise ValueError
        return data

    def dragEnterEvent(self, event):
        """ Check for correct datatype to accept drops """
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """ Check if only a single file was imported and then
        import the data from that file """
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()
            if len(url) > 1:
                QMessageBox.warning(self, "Multiple Import",
                                    "Only a single file can be imported!!!",
                                    QMessageBox.Close,
                                    QMessageBox.Close)
                return
            data = self.get_data_from_file(str(url[0].toLocalFile()))
            self.imported_data = data[:, [0, 1]]
            self.main_figure.plot(
                data[:, 0], data[:, 1], "--", label="Import Data")
            self.main_canvas.draw()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    SMM = SMMGUI()
    sys.exit(app.exec_())
