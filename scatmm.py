#!/usr/bin/env python3
"""
Main script that combines everything from the SMM method
"""

import json
import logging
import os
import shutil
import sys
from typing import Any, List, Tuple
import uuid
import webbrowser

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QPalette, QRegExpValidator
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout
import appdirs
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.style as mstyle
import numpy as np
from numpy.linalg import norm
import numpy.typing as npt

from modules.database import Database
from modules.fig_class import PltFigure
from modules.pso import particle_swarm
from modules.s_matrix import MatOutsideBounds
from modules.s_matrix import smm_angle, smm_broadband, smm_layer
from modules.s_matrix import Layer3D
from modules.structs import SRes, SType
from smm_uis.smm_main import Ui_SMM_Window
from smm_uis.sim_layer_widget import SimLayerLayout
from smm_uis.opt_layer_widget import OptLayerLayout

VERSION = "3.7.0"

log_config = {
    "format": '%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:'\
            '%(lineno)d:%(message)s',
    "level": logging.DEBUG,
}
logging.basicConfig(**log_config)
logging.getLogger("matplotlib").setLevel(logging.INFO)

# Check if there is a user folder with the config data
# If there isnt... Copy everything to that folder
config_dir = os.path.join(appdirs.user_data_dir(), "scatmm")
if not os.path.isdir(config_dir):
    logging.info("Copying config data to user directory")
    shutil.copytree("config", config_dir)


def find_loc(filename):
    """
    Find the location the first location of filename, following a order of path
    1st - User Local Config (%APPDATA% or ~/.local/share)
    2nd - Fallback to the base path
    """
    config_dir = os.path.join(appdirs.user_data_dir(), "scatmm")
    if os.path.isdir(config_dir):
        logging.info(f"{filename} in {config_dir}")
        return os.path.join(config_dir, filename)
    else:
        logging.info(f"{filename} in local config")
        return os.path.join("config", filename)


with open(os.path.join("config", "config.json"), "r") as global_config:
    global_properties = json.load(global_config)
    logging.debug(f"{global_properties=}")

# Update global config with the user config
with open(find_loc("config.json"), "r") as user_config:
    user_properties = json.load(user_config)
    global_properties.update(user_properties)
    logging.debug(f"{global_properties=}")

# Default plot properties
mstyle.use("smm_style")

class OptimizeWorkder(QtCore.QThread):
    """
    Worker thread to perform the optimization algorithm
    """
    # Connections with widgets from the main thread
    updateValueSignal = QtCore.pyqtSignal(int)
    updateTextSignal = QtCore.pyqtSignal(str)
    updateOptButton = QtCore.pyqtSignal(bool)
    returnOptRes = QtCore.pyqtSignal(object)

    def __init__(self, figure_handler, particle_info, lmb, compare_data, theta,
                 phi, pol, ref_medium, trn_medium, thickness, layer_list,
                 checks):
        super().__init__()
        logging.info("Starting Optimization Worker Thread")
        """ Initialize variables and aliases """
        self.figure_canvas = figure_handler
        self.figure = self.figure_canvas.axes
        self.smm_args = {
            "lmb": lmb,
            "theta": theta,
            "phi": phi,
            "pol": pol,
            "i_med": ref_medium,
            "t_med": trn_medium
        }
        self.compare_data = compare_data
        self.thickness = thickness
        self.layer_list = layer_list
        self.ref_check, self.trn_check, self.abs_check = checks
        self.particle_info = {
            key: info[1]
            for key, info in particle_info.items()
        }
        self.iterator = 0

    def run(self):
        """
        Thread initialization to run the optimization algorithm
        """
        # Disable optimization button
        self.updateOptButton.emit(False)

        def optimize_function(**thicknesses):
            """ Optimization function """
            t_array = np.stack([thick_i for thick_i in thicknesses.values()])
            ref, trn = np.apply_along_axis(
                lambda thick: smm_broadband(
                    self.layer_list, override_thick=thick, **self.smm_args), 0,
                t_array)
            if self.ref_check:
                point_error = np.sum((ref - self.compare_data)**2, axis=0)
            elif self.trn_check:
                point_error = np.sum((trn - self.compare_data)**2, axis=0)
            elif self.abs_check:
                point_error = np.sum((1 - ref - trn - self.compare_data)**2,
                                     axis=0)
            else:
                logging.error("This should not happen!!")
                raise Exception("Unknown optimization variable")
            self.updateValueSignal.emit(
                int(self.iterator / self.particle_info["n_iter"] * 100))
            self.iterator += 1
            return point_error

        # Datastructure for the particle swarm algorithm
        thick = {
            "thick_" + str(i): [thick_i[0], thick_i[1]]
            for i, thick_i in enumerate(self.thickness)
        }
        # Particle swarm algorithm
        try:
            lowest_error, best_thick, _, _ = particle_swarm(
                optimize_function,
                thick,
                swarm_properties=(self.particle_info["w"],
                                  self.particle_info["c1"],
                                  self.particle_info["c2"]),
                n_iter=self.particle_info["n_iter"],
                n_particles=self.particle_info["n_particles"],
                maximize=False)
        except MatOutsideBounds as error:
            self.updateValueSignal.emit(0)
            self.updateOptButton.emit(True)
            self.updateTextSignal.emit(str(error))
            return

        self.updateValueSignal.emit(0)
        # Update result on the QTextBrowser widget with the best results
        self.updateTextSignal.emit(f"Best error of: {lowest_error:.3g}")
        self.updateOptButton.emit(True)
        for i, t_i in enumerate(best_thick):
            self.updateTextSignal.emit(f"thick {i+1} = {t_i:.3f}")
        # Plot the best result
        ref, trn = smm_broadband(self.layer_list,
                                 override_thick=best_thick,
                                 **self.smm_args)
        if self.ref_check:
            self.figure.plot(self.smm_args["lmb"], ref, ":")
        elif self.trn_check:
            self.figure.plot(self.smm_args["lmb"], trn, ":")
        elif self.abs_check:
            self.figure.plot(self.smm_args["lmb"], 1 - trn - ref, ":")
        exp_results = SRes(1, SType.OPT, self.layer_list,
                           self.smm_args["theta"], self.smm_args["phi"],
                           self.smm_args["pol"], self.smm_args["i_med"],
                           self.smm_args["t_med"], self.smm_args["lmb"], ref,
                           trn)
        self.returnOptRes.emit(exp_results)
        self.figure_canvas.draw()


class SMMGUI(QMainWindow):
    def __init__(self):
        """
        Initialize necessary variables for the GUI and aliases to the important
        structures
        """
        super(SMMGUI, self).__init__()
        logging.info("Initializing UI/Connecting buttons to functions")
        self.ui = Ui_SMM_Window()
        self.ui.setupUi(self)
        # Initialize the Main Figure and toolbar
        logging.debug("Initializing variable aliases")
        self.fig_layout = self.ui.figure_layout
        self.main_canvas = PltFigure(self.fig_layout, "Wavelength (nm)",
                                     "R/T/Abs")
        mpl_toolbar = NavigationToolbar2QT(self.main_canvas, self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, mpl_toolbar)
        # Alias to add plots to the figure
        self.main_figure = self.main_canvas.axes
        # Initialize database
        db_file = os.path.join("Database", "database")
        self.database = Database(find_loc(db_file))
        # Create the sim_tab widget
        self._simLayout = QVBoxLayout(self.ui.sim_tab_sim_frame)
        self._simLayout.setContentsMargins(0, 0, 0, 0)
        self.simWidget = SimLayerLayout(self, self.database.content, 2)
        self._simLayout.addWidget(self.simWidget)
        # Create the opt_tab_widget
        self._optLayout = QVBoxLayout(self.ui.opt_tab_sim_frame)
        self._optLayout.setContentsMargins(0, 0, 0, 0)
        self.optWidget = OptLayerLayout(self, self.database.content, 2)
        self._optLayout.addWidget(self.optWidget)
        # Store Simulations
        self.sim_results: List[SRes] = []
        # Store imported data
        self.imported_data: Any = []
        # Window variables
        self.export_ui = None
        self.db_ui = None
        self.properties_window = None
        self.import_window = None
        # Load simulation default properties
        logging.debug("Loading default global properties")
        with open(find_loc("config.json"), "r") as config:
            self.global_properties = json.load(config)
        # Initialize main helper dictionaries for simulations
        self.sim_config = {
            "ref_n": self.ui.sim_tab_ref_n,
            "ref_k": self.ui.sim_tab_ref_k,
            "trn_n": self.ui.sim_tab_trn_n,
            "trn_k": self.ui.sim_tab_trn_k
        }
        self.opt_config = {
            "ref_n": self.ui.opt_tab_ref_n,
            "ref_k": self.ui.opt_tab_ref_k,
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
        Associate all UI elements with action functions
        """
        # Connect Sim and Opt Widgets
        self.ui.sim_tab_add_layer_button.clicked.connect(self.add_sim_layer)
        self.ui.sim_tab_rem_layer_button.clicked.connect(self.rmv_sim_layer)
        self.simWidget.abs_clicked.connect(
            lambda nlayer, state, id: self.plot_abs_layer(nlayer, state, id))
        self.ui.opt_tab_add_layer_button.clicked.connect(self.add_opt_layer)
        self.ui.opt_tab_rem_layer_button.clicked.connect(self.rmv_opt_layer)
        # Add Validators to several entries
        self.ui.sim_param_lmb_max.setValidator(QIntValidator(self))
        self.ui.sim_param_lmb_min.setValidator(QIntValidator(self))
        self.ui.sim_param_theta.setValidator(QIntValidator(0, 89, self))
        self.ui.sim_param_phi.setValidator(QIntValidator(0, 360, self))
        # Validator for the interface inputs
        self._double_validator = QDoubleValidator(self)
        self._double_validator.setLocale(QtCore.QLocale("en_US"))
        double_validators = [
            self.ui.sim_tab_ref_k, self.ui.sim_tab_ref_n,
            self.ui.sim_tab_trn_k, self.ui.sim_tab_trn_n,
            self.ui.opt_tab_ref_k, self.ui.opt_tab_ref_n,
            self.ui.opt_tab_trn_k, self.ui.opt_tab_trn_n
        ]
        for validator in double_validators:
            validator.setValidator(self._double_validator)
        # Regex validator for the polarization vectors
        pol_regex = QtCore.QRegExp("[0-9]+\\.?[0-9]*j?")
        self.ui.sim_param_pte.setValidator(QRegExpValidator(pol_regex))
        self.ui.sim_param_ptm.setValidator(QRegExpValidator(pol_regex))
        # Connect remaining buttons
        self.ui.sim_param_check_angle.stateChanged.connect(self.sim_angle)
        self.ui.sim_param_check_lmb.stateChanged.connect(self.sim_lmb)
        self.ui.sim_tab_sim_button.clicked.connect(self.simulate)
        self.ui.import_button.clicked.connect(self.import_data)
        self.ui.opt_tab_sim_button.clicked.connect(self.pre_optimize_checks)
        # Connect menu buttons
        self.ui.actionView_Database.triggered.connect(self.view_database)
        self.ui.actionClear.triggered.connect(self.clear_sim_buffer)
        self.ui.actionClear_Plots.triggered.connect(self.clear_plot)
        self.ui.actionExport.triggered.connect(self.export_simulation)
        self.ui.actionImport.triggered.connect(self.import_data)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionProperties.triggered.connect(self.open_properties)
        self.ui.actionHelp.triggered.connect(
            lambda: webbrowser.open_new_tab(os.path.join("Help", "help.html")))
        self.ui.actionAbout.triggered.connect(self.aboutDialog)
        self.ui.clear_button.clicked.connect(self.clear_plot)
        logging.debug("All buttons connected to functions")
        # Set default value for progress bar
        self.ui.opt_progressBar.setValue(0)

    def aboutDialog(self):
        """Show the about dialog"""
        logging.info("Open About Dialog Window")
        title = "About Scatmm"
        msg = "Graphical interface to interact with"\
            "the transfer matrix method (using scattering"\
            f"matrices\n\nAuthor: Miguel Alexandre\n\nVersion: {VERSION}"
        QMessageBox.about(self, title, msg)

    """ Simulation Configuration (wavelength or angle) """

    def sim_angle(self, int):
        """ Change simulation to angle simulation """
        logging.info("Clicked Simulation Type Angle Checkbox")
        if self.ui.sim_param_check_lmb.isChecked() and int > 0:
            self.ui.sim_param_check_lmb.setChecked(False)
            # Reinitialize plot
            self.main_canvas.reinit()
            self.main_canvas.draw_axes(xlabel="Angle (θ)")
            self.main_canvas.draw()
            # Disable non-necessary text-boxes
            self.sim_data["lmb_max"].setDisabled(True)
            self.sim_data["theta"].setDisabled(True)
            # Disable absorption checkboxes
            self.simWidget.reinit_abs_checkbox(True)
        elif not self.ui.sim_param_check_lmb.isChecked() and int == 0:
            self.ui.sim_param_check_lmb.setChecked(True)
            self.main_canvas.reinit()
            self.main_canvas.draw_axes(xlabel="Wavelength (nm)")
            self.main_canvas.draw()
            # Enable non-necessary text-boxes
            self.sim_data["lmb_max"].setDisabled(False)
            self.sim_data["theta"].setDisabled(False)
            # Enable absorption checkboxes
            self.simWidget.reinit_abs_checkbox(True)

    def sim_lmb(self, int):
        """ Change simulation to broadband simulation """
        logging.info("Clicked Simulation Type Wavelength Checkbox")
        if self.ui.sim_param_check_angle.isChecked() and int > 0:
            self.ui.sim_param_check_angle.setChecked(False)
            self.main_canvas.reinit()
            self.main_canvas.draw_axes(xlabel="Wavelength (nm)")
            self.main_canvas.draw()
            # Enable non-necessary text-boxes
            self.sim_data["lmb_max"].setDisabled(False)
            self.sim_data["theta"].setDisabled(False)
            # Enable absorption checkboxes
            self.simWidget.reinit_abs_checkbox(False)
        elif not self.ui.sim_param_check_angle.isChecked() and int == 0:
            self.ui.sim_param_check_angle.setChecked(True)
            self.main_canvas.reinit()
            self.main_canvas.draw_axes(xlabel="Angle (θ)")
            self.main_canvas.draw()
            # Disable non-necessary text-boxes
            self.sim_data["lmb_max"].setDisabled(True)
            self.sim_data["theta"].setDisabled(True)
            self.simWidget.reinit_abs_checkbox(True)

    """ Properties Button and associated functions """

    def update_properties(self):
        """
        Update all the properties - Linked to Apply button in properties window
        """
        logging.info("Updating all simulation properties")
        try:
            for key in self.global_properties.keys():
                if self.global_properties[key][0] == "int":
                    self.global_properties[key] = [
                        "int", int(self.propertie_values[key].text())
                    ]
                elif self.global_properties[key][0] == "float":
                    self.global_properties[key] = [
                        "float",
                        float(self.propertie_values[key].text())
                    ]
                else:
                    logging.critical(
                        "Invalid type in config.json.. Should not happen")
                    raise Exception("Unknown type format in config.json")
        except ValueError:
            logging.warning("Invalid configuration in config file")
            raise Exception("Mistaken type in config file")
        logging.info("Closing properties window")
        if self.properties_window is not None:
            self.properties_window.close()
        else:
            raise Exception("Unknown Problem...")

    def save_default_properties(self):
        """
        Store default properties in config.json file
        """
        self.update_properties()
        logging.info("Storing defaults in config file")
        with open(find_loc("config.json"), "w") as config:
            json.dump(self.global_properties, config, indent=2)

    def open_properties(self):
        """
        Open a new window showing the properties
        """
        # Load the properties UI
        from smm_uis.smm_properties_ui import Ui_Properties
        logging.info("Opening properties window")
        self.properties_window = QtWidgets.QTabWidget()
        self.properties_ui = Ui_Properties()
        self.properties_ui.setupUi(self.properties_window)
        self.properties_window.show()
        # Create a dictionary linked to the properties values
        logging.debug("Aliasing Properties text boxes")
        self.propertie_values = {
            "n_particles": self.properties_ui.prop_num_particles,
            "n_iter": self.properties_ui.prop_iter_num,
            "w": self.properties_ui.prop_w,
            "c1": self.properties_ui.prop_cog1,
            "c2": self.properties_ui.prop_cog2,
            "sim_points": self.properties_ui.prop_wav_points
        }
        # Load the properties values from the global properties variable
        logging.debug("Loading global properties")
        for key in self.global_properties.keys():
            self.propertie_values[key].setText(
                str(self.global_properties[key][1]))

        # Atribute actions to the different buttons
        logging.debug("Connect buttons to functions")
        self.properties_ui.properties_close_button.clicked.connect(
            self.properties_window.close)
        self.properties_ui.properties_apply_button.clicked.connect(
            self.update_properties)
        self.properties_ui.save_default_button.clicked.connect(
            self.save_default_properties)

    """ Layer Management """

    def add_sim_layer(self):
        """ Add a new layer to the simulation tab """
        angle_check: bool = self.ui.sim_param_check_angle.isChecked()
        self.simWidget.add_layer(self.database.content, disable=angle_check)

    def rmv_sim_layer(self, index: int = -1):
        """ Remove a specific layer from the stack """
        self.simWidget.rmv_layer(index)

    def add_opt_layer(self):
        """ Add a new layer to the simulation tab """
        self.optWidget.add_layer(self.database.content)

    def rmv_opt_layer(self, index: int = -1):
        """ Remove a specific layer from the stack """
        self.optWidget.rmv_layer(index)

    """ Aliases to get information necessary for simulation/optimization """

    def get_sim_data(self):
        """
        Get simulation configuration
        """
        logging.info("Retrieving Simulation configuration")
        try:
            theta = np.radians(float(self.sim_data["theta"].text()))
            phi = np.radians(float(self.sim_data["phi"].text()))
            lmb_min = float(self.sim_data["lmb_min"].text())
            lmb_max = float(self.sim_data["lmb_max"].text())
            pol = np.array([
                complex(self.sim_data["pte"].text()),
                complex(self.sim_data["ptm"].text())
            ])
            pol /= norm(pol)
            logging.debug(
                f"Sim Config: {theta}:{phi}:{lmb_min}:{lmb_max}:{pol}")
        except ValueError as error:
            raise ValueError(error)
        return theta, phi, pol, lmb_min, lmb_max

    def get_medium_config(
        self, data_structure
    ) -> Tuple[Tuple[complex, complex], Tuple[complex, complex]]:
        """
        Get data for the transmission and reflection media
        """
        logging.info("Retrieve envolvent medium configuration")
        try:
            ref_medium_n = float(data_structure["ref_n"].text())
            ref_medium_k = float(data_structure["ref_k"].text())
            ref_medium = ((ref_medium_n + 1j * ref_medium_k)**2, 1 + 0j)
            trn_medium_n = float(data_structure["trn_n"].text())
            trn_medium_k = float(data_structure["trn_k"].text())
            trn_medium = ((trn_medium_n + 1j * trn_medium_k)**2, 1 + 0j)
            logging.debug(f"Retrieved: {ref_medium}:{trn_medium}")
        except ValueError as error:
            raise ValueError(error)
        return ref_medium, trn_medium

    def get_sim_config(self) -> List[Layer3D]:
        """ Return the thickness from simulation tab """
        layer_list: List[Layer3D] = []
        layer_info: List[Tuple[str, float]] = self.simWidget.layer_info()
        for layer in layer_info:
            db_data: npt.NDArray = self.database[layer[0]]
            layer_list.append(
                Layer3D(layer[0],
                        layer[1],
                        db_data[:, 0],
                        db_data[:, 1],
                        db_data[:, 2],
                        kind='cubic'))
        return layer_list

    def get_opt_config(self) -> Tuple[List[List[float]], List[Layer3D]]:
        """ Get optimization info """
        try:
            layer_info = self.optWidget.layer_info()
        except ValueError as error:
            logging.error(error)
            QMessageBox.warning(self, "Thickness Error",
                                "Optimization thicknesses not defined!!",
                                QMessageBox.Close, QMessageBox.Close)
            raise ValueError
        layer_list: List[Layer3D] = []
        thickness_list: List[List[float]] = []
        for layer_mat, thickness in layer_info:
            db_data = self.database[layer_mat]
            thickness_list.append(list(thickness))
            layer_list.append(
                Layer3D(layer_mat,
                        thickness[0],
                        db_data[:, 0],
                        db_data[:, 1],
                        db_data[:, 2],
                        kind='cubic'))
        return thickness_list, layer_list

    """ Simulation and associated funcitons """

    def simulate(self) -> None:
        """"
        Script that outputs all the data necessary to run smm_broadband
        """
        logging.info("Starting Simulation....")
        # Get all the sim_data values
        try:
            theta, phi, pol, lmb_min, lmb_max = self.get_sim_data()
            lmb = np.linspace(lmb_min, lmb_max,
                              self.global_properties["sim_points"][1])
            # Get all the sim_config_values
            ref_medium, trn_medium = self.get_medium_config(self.sim_config)
            layer_list = self.get_sim_config()
        except ValueError as error:
            logging.error(error)
            QMessageBox.warning(
                self, "Invalid Parameter",
                "Invalid simulation parameter... Check if everything is filled or if all inserted characters are valid"
            )
            return

        # Do the simulation
        try:
            if self.ui.sim_param_check_lmb.isChecked():
                ref, trn = smm_broadband(layer_list, theta, phi, lmb, pol,
                                         ref_medium, trn_medium)
                self.sim_plot_data(lmb, ref, trn)
                results = SRes(len(self.sim_results), SType.WVL, layer_list,
                               theta, phi, pol, ref_medium, trn_medium, lmb,
                               ref, trn)
                self.sim_results.append(results)
            else:
                theta = np.linspace(0, 89,
                                    self.global_properties["sim_points"][1])
                ref, trn = smm_angle(layer_list, np.radians(theta), phi,
                                     lmb_min, pol, ref_medium, trn_medium)
                self.sim_plot_data(theta, ref, trn)
                results = SRes(len(self.sim_results), SType.ANGLE, layer_list,
                               theta, phi, pol, ref_medium, trn_medium,
                               lmb_min, ref, trn)
                self.sim_results.append(results)
            # Reinitialize the absorption checkboxes
            self.simWidget.reinit_abs_checkbox()
            logging.info("Finished Simulation")
        except MatOutsideBounds as message:
            logging.warning("Material Outside Bounds")
            title = "Error: Material Outside of Bounds"
            QMessageBox.warning(self, title, str(message), QMessageBox.Close,
                                QMessageBox.Close)
        if self.export_ui:
            logging.info("Updating exportUI with new simulation")
            self.export_ui.update_sims(self.sim_results)
        # Update the number of simulations in the clear button
        self.ui.tab_widget.setTabText(0, f"Simulation ({len(self.sim_results)})")
        self.simWidget.reinit_abs_checkbox()

    def sim_plot_data(self, x, ref, trn) -> None:
        """
        Verify which checkboxes are toggled and plot data accordingly
        """
        logging.debug("Plotting Simulation Data")
        # Check what data to plot
        ref_check: bool = self.sim_data["ref"].isChecked()
        trn_check: bool = self.sim_data["trn"].isChecked()
        abs_check: bool = self.sim_data["abs"].isChecked()
        simulations = str(len(self.sim_results))
        # Plot according to the checkboxes
        if ref_check:
            logging.debug("Reflection Checkbox detected")
            self.main_figure.plot(x, ref, label="R Sim(" + simulations + ")")
        if trn_check:
            logging.debug("Transmission Checkbox detected")
            self.main_figure.plot(x, trn, label="T Sim(" + simulations + ")")
        if abs_check:
            logging.debug("Absorption Checkbox detected")
            self.main_figure.plot(x,
                                  1 - ref - trn,
                                  label="A Sim(" + simulations + ")")
        self.main_canvas.draw()

    def clear_sim_buffer(self) -> None:
        """
        Clear all stored simulation values and destroy all open plots
        """
        logging.info("Clearing all results from simulation stack")
        self.sim_results = []
        self.main_canvas.reinit()
        self.ui.tab_widget.setTabText(0, "Simulation")
        self.imported_data = []
        self.simWidget.reinit_abs_checkbox()

    def plot_abs_layer(self, layer_index: int, cb_state: bool,
                       id: uuid.UUID) -> None:
        """ Plot absorption for each checked layer """
        logging.info("Plotting absorption for single Layer")
        # Check if there are simulations
        if len(self.sim_results) == 0:
            logging.warning("No simulation has been made yet")
            self.simWidget.reinit_abs_checkbox()
            return
        # Check if the last simulation was not angular
        if self.sim_results[-1].Type == SType.ANGLE:
            logging.warning("Previous simulation was angle...")
            self.simWidget.reinit_abs_checkbox()
            return
        # Get all the data from the last simulations
        theta = self.sim_results[-1].Theta
        phi = self.sim_results[-1].Phi
        pol = self.sim_results[-1].Pol
        lmb = self.sim_results[-1].Lmb
        ref_medium = self.sim_results[-1].INC_MED
        trn_medium = self.sim_results[-1].TRN_MED
        layer_list = self.sim_results[-1].Layers
        if not cb_state:
            self.delete_plot(id)
            self.main_canvas.draw()
            return
        abs = smm_layer(layer_list, layer_index + 1, theta, phi, lmb, pol,
                        ref_medium, trn_medium)
        # Define a random ID for each partiular layer absorption and plot
        self.main_figure.plot(
            lmb,
            abs,
            "--",
            gid=id,
            label=f"A{len(self.sim_results)}-Layer:{layer_index}")
        self.main_canvas.draw()

    def clear_plot(self) -> None:
        """ Clear all the plots without removing imported data """
        self.main_canvas.reinit()
        if len(self.imported_data) > 1:
            self.main_figure.plot(self.imported_data[:, 0],
                                  self.imported_data[:, 1],
                                  '--',
                                  label="Import Data",
                                  gid="Imported_Data")
        self.main_canvas.draw()

    def delete_plot(self, id) -> None:
        """ Delete a specific plot defined by guid """
        for plt in self.main_figure.lines:
            if plt.get_gid() == id:
                logging.debug(f"Plot found with gid: {id}... Deleting...")
                plt.remove()

    """ Call  Export UI """

    def export_simulation(self) -> None:
        """
        Open a new window displaying all the simulated results
        Choose outputs and then ask for directory to output information
        """
        logging.info("Opening Export Simulation Window")
        if len(self.sim_results) > 0:
            from smm_uis.export_window import ExpWindow
            self.export_ui = ExpWindow(self, self.sim_results)
            self.export_ui.show()
        else:
            logging.warning("No Simulation detected...")
            QMessageBox.warning(self, "Simulation Required",
                                "No Simulation available to export!!",
                                QMessageBox.Close, QMessageBox.Close)

    """ Optimization functions """

    def update_progress_bar(self, value: int):
        """Connect the progress bar with the worker thread"""
        logging.debug(f"Updating progress bar to {value}")
        return self.ui.opt_progressBar.setValue(value)

    def updateTextBrowser(self, text: str):
        """Connects the textBrowser with the worker thread"""
        logging.debug("Updating Optimization text browser")
        return self.ui.opt_res_text.append(text)

    def updateResults(self, results) -> None:
        logging.debug("Added optimization results to list")
        results.update_ID(len(self.sim_results))
        self.sim_results.append(results)

    def updateOptButton(self, status) -> None:
        """Connects the Optimization button with the worker thread"""
        logging.debug(f"Updating Optimization/Simulation button: {status}")
        self.ui.opt_tab_sim_button.setEnabled(status)
        self.ui.sim_tab_sim_button.setEnabled(status)

    def pre_optimize_checks(self) -> None:
        """
        Perform checks before going into the optimization loop
        """
        logging.info("Starting pre optimization checks...")
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
            logging.warning("Multiple optimization observables selected")
            QMessageBox.warning(self, title, msg1, QMessageBox.Close,
                                QMessageBox.Close)
            self.ui.opt_res_text.append("Optimization Failed")
            return
        elif ref_check + trn_check + abs_check == 0:
            logging.warning("No simulation observables selected")
            QMessageBox.warning(self, title, msg2, QMessageBox.Close,
                                QMessageBox.Close)
            self.ui.opt_res_text.append("Optimization Failed")
            return
        self.optimize(ref_check, trn_check, abs_check)

    def optimize(self, ref_check, trn_check, abs_check) -> None:
        """
        Perform Results Optimization (Splits into a new worker thread)
        """
        # Check if there is imported data
        if self.imported_data is None or len(self.imported_data) == 0:
            logging.warning("Missing Imported Data for Optimization")
            QMessageBox.warning(self, "Import Data missing",
                                "Missing data to perform optimization",
                                QMessageBox.Close, QMessageBox.Close)
            self.ui.opt_res_text.append("Optimization Failed")
            return
        if self.ui.sim_param_check_angle.isChecked():
            logging.warning("Wrong Simulation type detected")
            error = "Optimization only available for Wavelength type" +\
                " simulations\n\n Change?"
            change = QMessageBox.question(self,
                                          "Invalid Simulation Mode",
                                          error,
                                          QMessageBox.Yes | QMessageBox.No,
                                          defaultButton=QMessageBox.Yes)
            if change == QMessageBox.Yes:
                logging.debug("Changing Simulation Type")
                self.ui.sim_param_check_lmb.setChecked(True)
            return
        # Get all the optimization info
        try:
            theta, phi, pol, lmb_min, lmb_max = self.get_sim_data()
            ref_medium, trn_medium = self.get_medium_config(self.opt_config)
            thick, layer_list = self.get_opt_config()
        except ValueError as error:
            logging.error(error)
            return

        try:
            # Get the data for optimization
            lmb = self.imported_data[:, 0].T
            compare_data = self.imported_data[:, 1][:, np.newaxis]
            # Adjust the imported file to adhere to lower and upper wvl limits
            wvl_mask = (lmb > lmb_min) & (lmb < lmb_max)
            lmb = lmb[wvl_mask]
            compare_data = compare_data[wvl_mask]
            self.main_canvas.reinit()
            self.main_figure.plot(lmb, compare_data, "--")
            self.main_canvas.draw()
            # Create a new worker thread to do the optimization
            self.ui.opt_res_text.clear()
            self.ui.opt_res_text.append("Simulation started...")
            self.opt_worker = OptimizeWorkder(
                self.main_canvas, self.global_properties, lmb, compare_data,
                theta, phi, pol, ref_medium, trn_medium, thick, layer_list,
                (ref_check, trn_check, abs_check))
            # Connect the new thread with functions from the main thread
            self.opt_worker.updateValueSignal.connect(self.update_progress_bar)
            self.opt_worker.updateTextSignal.connect(self.updateTextBrowser)
            self.opt_worker.updateOptButton.connect(self.updateOptButton)
            self.opt_worker.returnOptRes.connect(self.updateResults)
            # Start worker
            self.opt_worker.start()
        # The ValueError and Exception are handled inside the calling functions
        except MatOutsideBounds as error:
            logging.warning(error)
            title = "Error: Material Out of Bounds"
            error = "One of the materials in the simulation is "\
                "undefined for the defined wavelength range"
            QMessageBox.warning(self, title, error, QMessageBox.Close,
                                QMessageBox.Close)
        except (ValueError, Exception) as error:
            logging.warning(error)

    """ Connection to the Database Window """

    def view_database(self) -> None:
        """
        Open a new window to see all the database materials
        """
        from smm_uis.db_window import DBWindow
        logging.info("Calling the Database Window")
        self.db_ui = DBWindow(self, self.database)
        self.db_ui.db_updated.connect(self.update_mat_cb)
        self.db_ui.show()

    def update_mat_cb(self):
        self.simWidget.update_cb_items(self.database.content)
        self.optWidget.update_cb_items(self.database.content)

    """ Import data from file """

    def import_data(self):
        """
        Function for import button - import data for simulation/optimization
        """
        from smm_uis.imp_window import ImpPrevWindow, ImpFlag
        logging.info("Import Button Clicked")
        self.import_window = ImpPrevWindow(self,
                                           imp_flag=ImpFlag.BUTTON
                                           | ImpFlag.DATA | ImpFlag.NONAME)
        self.import_window.imp_clicked.connect(self._import)
        self.import_window.show()

    def _import(self, import_data, name):
        logging.debug(f"Imported data detected:\n {import_data=}\n{name=}")
        if self.import_window is None:
            logging.critical("Unknown error")
            return
        self.imported_data = import_data.values[:, [0, 1]]
        self.delete_plot("Imported_Data")
        self.main_figure.plot(self.imported_data[:, 0],
                              self.imported_data[:, 1],
                              '--',
                              label="Import Data",
                              gid="Imported_Data")
        self.main_canvas.draw()
        self.import_window.close()

    """ Drag and Drop Functionality """

    def dragEnterEvent(self, event) -> None:
        """ Check for correct datatype to accept drops """
        logging.debug("Drag event Detected")
        if event.mimeData().hasUrls():
            logging.debug("Acceptable event data...")
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, a0: QtGui.QDragMoveEvent) -> None:
        return super().dragMoveEvent(a0)

    def dropEvent(self, event) -> None:
        """ Check if only a single file was imported and then
        import the data from that file """
        from smm_uis.imp_window import ImpPrevWindow, ImpFlag
        logging.debug("Handling Drop event")
        if not event.mimeData().hasUrls():
            return
        url = event.mimeData().urls()
        if len(url) > 1:
            logging.warning("Invalid number of files dragged..")
            QMessageBox.warning(self, "Multiple Import",
                                "Only a single file can be imported!!!",
                                QMessageBox.Close, QMessageBox.Close)
            return
        filepath = str(url[0].toLocalFile())
        if os.path.isdir(filepath):
            logging.info("Invalid File Type")
            return
        self.import_window = ImpPrevWindow(self,
                                           imp_flag=ImpFlag.DRAG
                                           | ImpFlag.DATA,
                                           filepath=filepath)
        self.import_window.imp_clicked.connect(self._import)
        self.import_window.show()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        """ Clean all possible open windows """
        if self.properties_window is not None:
            logging.info("Properties Window still opened... Closing...")
            self.properties_window.close()
        if self.db_ui is not None:
            logging.info("Database Window still opened... Closing...")
            self.db_ui.close()
        if self.import_window is not None:
            logging.info("Import Window still opened... Closing...")
            self.import_window.close()
        # Save the final global_properties
        # with open(find_loc("config.json"), "w") as end_config:
        #     json.dump(global_properties, end_config, indent=2)
        return super().closeEvent(a0)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Update matplotlib color to match qt application
    color = app.palette().color(QPalette.Background)
    mpl.rcParams["axes.facecolor"] = f"{color.name()}"
    mpl.rcParams["figure.facecolor"] = f"{color.name()}"
    SMM = SMMGUI()
    sys.exit(app.exec_())
