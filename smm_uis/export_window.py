import numpy as np
import os

# from scatmm import SType
from .smm_export_window import Ui_ExportWindow
from modules.fig_class import FigWidget, PltFigure
from modules.structs import SType

from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox


class ExpWindow(QWidget):
    """ Main class for the export window """

    def __init__(self, parent, sim_results):
        """ Initialize elements of the main window """
        self.sim_results = sim_results
        self.parent = parent
        super(ExpWindow, self).__init__()
        self.ui = Ui_ExportWindow()
        self.ui.setupUi(self)
        self.chosen_sim = self.ui.simulations_combobox
        self.export_checks = [
                self.ui.reflection_checkbox,
                self.ui.transmission_checkbox,
                self.ui.absorption_checkbox
        ]
        export_names = [sim_name.ID for sim_name in self.sim_results]
        self.chosen_sim.addItems(export_names)
        self.initializeUI()

    def initializeUI(self):
        """ Connect Ui Elements to Functions """
        self.ui.export_all_button.clicked.connect(self.export_all)
        self.ui.export_button.clicked.connect(self.export)
        self.ui.preview_button.clicked.connect(self.preview_export)

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
        header = ""
        if savepath[0] == "":
            return
        for result in self.sim_results:
            if result.ID == sim_choice:
                # Start building the export array
                if result.Type == SType.WVL:
                    export_array = result.Lmb
                    header += "Wavelength(nm)"
                elif result.Type == SType.ANGLE:
                    export_array = result.Theta
                    header += "Angle"
                export_array = export_array[:, np.newaxis]
                if ref_check:
                    export_array = np.concatenate(
                        (export_array, result.Ref[:, np.newaxis]), axis=1)
                    header += " Ref"
                if trn_check:
                    export_array = np.concatenate(
                        (export_array, result.Trn[:, np.newaxis]), axis=1)
                    header += " Trn"
                if abs_check:
                    export_array = np.concatenate(
                        (export_array,
                            (1-result.Ref-result.Trn)[:, np.newaxis]), axis=1)
                    header += " Abs"
                np.savetxt(savepath[0], export_array, header=header)
        QMessageBox.information(self, "Successful Export",
                                "Results Exported Successfully!",
                                QMessageBox.Ok, QMessageBox.Ok)
        self.raise_()
        self.setFocus(True)
        self.activateWindow()

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
        header = ""
        for result in self.sim_results:
            # Start building the export array
            if result.Type == SType.WVL:
                export_array = result.Lmb
                header += "Wavelength(nm)"
            elif result.Type == SType.ANGLE:
                export_array = result.Theta
                header += "Angle"
            export_array = export_array[:, np.newaxis]
            if ref_check:
                export_array = np.concatenate(
                    (export_array, result.Ref[:, np.newaxis]), axis=1)
                header += " Ref"
            if trn_check:
                export_array = np.concatenate(
                    (export_array, result.Trn[:, np.newaxis]), axis=1)
                header += " Trn"
            if abs_check:
                export_array = np.concatenate(
                    (export_array,
                        (1-result.Ref-result.Trn)[:, np.newaxis]), axis=1)
                header += " Abs"
            export_path = os.path.join(export_dir,
                                       export_name+result.ID+".txt")
            np.savetxt(export_path, export_array, header=header)
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
        for result in self.sim_results:
            if result.ID == sim_choice:
                if result.Type == SType.WVL:
                    xlabel = "Wavelength (nm)"
                    xdata = result.Lmb
                elif result.Type == SType.ANGLE:
                    xlabel = "Angle (θ)"
                    xdata = result.Theta
                self.preview_fig = PltFigure(self.preview_window.layout,
                                             xlabel, "R/T/Abs", width=7)
                self.preview_window.show()
                self.preview_fig.axes.plot(xdata, result.Ref, label="Ref")
                self.preview_fig.axes.plot(xdata, result.Trn, label="Trn")
                self.preview_fig.axes.plot(xdata, 1-result.Ref-result.Trn,
                                           label="Abs")
        self.preview_fig.axes.legend(bbox_to_anchor=(1, 1), loc="upper left")
        self.preview_fig.draw()
