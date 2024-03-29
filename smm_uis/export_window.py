import numpy as np
import logging
import os

# from scatmm import SType
from .smm_export_gui import Ui_ExportWindow
from modules.fig_class import FigWidget, PltFigure
from modules.structs import SType
from modules.s_matrix import smm_layer

from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox


class ExpWindow(QWidget):
    """Main class for the export window"""

    def __init__(self, parent, sim_results):
        """Initialize elements of the main window"""
        self.sim_results = sim_results
        self.parent = parent
        super(ExpWindow, self).__init__()
        self.ui = Ui_ExportWindow()
        self.ui.setupUi(self)
        self.chosen_sim = self.ui.simulations_combobox
        self.export_checks = [
            self.ui.reflection_checkbox,
            self.ui.transmission_checkbox,
            self.ui.absorption_checkbox,
            self.ui.layers_checkbox,
        ]
        export_names = [sim_name.repr for sim_name in self.sim_results]
        self.chosen_sim.addItems(export_names)
        self.initializeUI()
        self.update_sim_info()

    def initializeUI(self):
        """Connect Ui Elements to Functions"""
        self.ui.export_all_button.clicked.connect(self.export_all)
        self.ui.export_button.clicked.connect(self.export)
        self.ui.preview_button.clicked.connect(self.preview_export)
        self.chosen_sim.activated.connect(self.update_sim_info)

    def update_sims(self, sim_results):
        logging.info("Updating simulation info in export window")
        self.sim_results = sim_results
        export_names = [sim_name.repr for sim_name in self.sim_results]
        self.chosen_sim.clear()
        self.chosen_sim.addItems(export_names)

    def update_sim_info(self):
        """Update the simulation information in the QText Edit"""
        sim_choice = self.chosen_sim.currentText()
        summary = ""
        for sim in self.sim_results:
            print([sim == sim_i for sim_i in self.sim_results])
            if sim.repr == sim_choice:
                logging.debug(f"Chosen {sim}")
                summary = sim.description()
                break
        logging.debug(summary)
        self.ui.simulation_summary.setText(summary)

    """ Button functions """

    def export(self):
        """
        Export chosen simulation
        """
        savepath = QFileDialog.getSaveFileName(
            self, "Save File", "", "Text Files (*.txt)"
        )
        sim_choice = self.chosen_sim.currentText()
        ref_check = self.export_checks[0].isChecked()
        trn_check = self.export_checks[1].isChecked()
        abs_check = self.export_checks[2].isChecked()
        layers_check = self.export_checks[3].isChecked()
        if savepath[0] == "":
            return
        for result in self.sim_results:
            header = ""
            if result.repr == sim_choice:
                # Start building the export array
                xlabel, xdata = result.xinfo()
                header += xlabel
                export_array = xdata
                export_array = export_array[:, np.newaxis]
                if ref_check:
                    export_array = np.concatenate(
                        (export_array, result.Ref[:, np.newaxis]), axis=1
                    )
                    header += " Ref"
                if trn_check:
                    export_array = np.concatenate(
                        (export_array, result.Trn[:, np.newaxis]), axis=1
                    )
                    header += " Trn"
                if abs_check:
                    export_array = np.concatenate(
                        (export_array, (1 - result.Ref - result.Trn)[:, np.newaxis]),
                        axis=1,
                    )
                    header += " Abs"
                if result.Type == SType.ANGLE:
                    np.savetxt(savepath[0], export_array, header=header)
                    break
                if layers_check and result.Type is not SType.ANGLE:
                    for i in range(len(result.Layers)):
                        abs_i = smm_layer(
                            result.Layers,
                            i + 1,
                            result.Theta,
                            result.Phi,
                            result.Lmb,
                            result.Pol,
                            result.INC_MED,
                            result.TRN_MED,
                        )
                        export_array = np.concatenate(
                            (export_array, abs_i[:, np.newaxis]), axis=1
                        )
                        exp_header = result.Layers[i].name.replace(" ", "_")
                        header += f" Abs_{exp_header}"
                np.savetxt(savepath[0], export_array, header=header)
                break
        QMessageBox.information(
            self,
            "Successful Export",
            "Results Exported Successfully!",
            QMessageBox.Ok,
            QMessageBox.Ok,
        )
        self.raise_()
        self.setFocus(True)
        self.activateWindow()

    def export_all(self):
        """
        Export all simulations
        """
        export_dir = QFileDialog.getExistingDirectory(self, "Save To:")
        if not export_dir:
            logging.debug("No Directry provided... Ignoring")
            return
        logging.info(f"Export path: {export_dir}")
        ref_check = self.export_checks[0].isChecked()
        trn_check = self.export_checks[1].isChecked()
        abs_check = self.export_checks[2].isChecked()
        layers_check = self.export_checks[3].isChecked()
        for result in self.sim_results:
            header = ""
            xlabel, xdata = result.xinfo()
            header += xlabel
            export_array = xdata
            export_array = export_array[:, np.newaxis]
            if ref_check:
                logging.debug("Exporting Reflection....")
                export_array = np.concatenate(
                    (export_array, result.Ref[:, np.newaxis]), axis=1
                )
                header += " Ref"
            if trn_check:
                logging.debug("Exporting Transmission....")
                export_array = np.concatenate(
                    (export_array, result.Trn[:, np.newaxis]), axis=1
                )
                header += " Trn"
            if abs_check:
                logging.debug("Exporting Absorption....")
                export_array = np.concatenate(
                    (export_array, (1 - result.Ref - result.Trn)[:, np.newaxis]), axis=1
                )
                header += " Abs"
            if layers_check and result.Type is not SType.ANGLE:
                logging.debug("Exporting Layers....")
                for i in range(len(result.Layers)):
                    abs_i = smm_layer(
                        result.Layers,
                        i + 1,
                        result.Theta,
                        result.Phi,
                        result.Lmb,
                        result.Pol,
                        result.INC_MED,
                        result.TRN_MED,
                    )
                    export_array = np.concatenate(
                        (export_array, abs_i[:, np.newaxis]), axis=1
                    )
                    exp_header = result.Layers[i].name.replace(" ", "_")
                    header += f" Abs_{exp_header}"
            # Replace unwnated " " and "|" from the export name
            id = result.repr.replace("|", "_").replace(" ", "")
            export_path = os.path.join(export_dir, id)
            np.savetxt(export_path, export_array, header=header)
        self.close()
        QMessageBox.information(
            self,
            "Successful Export",
            "Results Exported Successfully!",
            QMessageBox.Ok,
            QMessageBox.Ok,
        )

    def preview_export(self):
        """
        Preview the selected simulation chosen in the combobox
        """
        sim_choice = self.chosen_sim.currentText()
        self.preview_window = FigWidget(f"Preview {sim_choice}")
        for result in self.sim_results:
            if result.repr == sim_choice:
                xlabel, xdata = result.xinfo()
                self.preview_fig = PltFigure(
                    self.preview_window.layout, xlabel, "R/T/Abs", width=7
                )
                self.preview_window.show()
                self.preview_fig.axes.plot(xdata, result.Ref, label="Ref")
                self.preview_fig.axes.plot(xdata, result.Trn, label="Trn")
                self.preview_fig.axes.plot(
                    xdata, 1 - result.Ref - result.Trn, label="Abs"
                )
        self.preview_fig.axes.legend(bbox_to_anchor=(1, 1), loc="upper left")
        self.preview_fig.draw()

    def closeEvent(self, event):
        logging.debug("Closing Export Window / Restore export_ui to None")
        event.accept()
        self.parent.export_ui = None
