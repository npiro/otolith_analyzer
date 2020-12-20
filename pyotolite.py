from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QTreeWidgetItem
from PyQt5.Qt import QStandardItemModel, QStandardItem
from PyQt5.QtGui import QFont, QColor
#from pyqtgraph import PlotWidget
#import pyqtgraph as pg

import sys, os
from datasets import Datasets
from gui import PandasModel, TableModel, get_checked_items, get_checked_items_in_subtree, plot_labeled_dataset, plot_labeled_dataset_field
from labelling import predict_labels_on_selected_datasets, load_model, auto_select_elements

def load_dir_dialog(widget, text_box):
    # get download_path from lineEdit
    download_path = text_box.text()

    # open select folder dialog
    fname = QFileDialog.getExistingDirectory(
        widget, 'Select a directory', download_path)

    if fname:
        # Returns pathName with the '/' separators converted to separators that are appropriate for the underlying operating system.
        # On Windows, toNativeSeparators("c:/winnt/system32") returns
        # "c:\winnt\system32".
        fname = QDir.toNativeSeparators(fname)

    if os.path.isdir(fname):
        text_box.setText(fname)


def load_file_dialog(widget, text_box):
    # get download_path from lineEdit
    path = text_box.text()

    # open select filename dialog
    fname, _ = QFileDialog.getOpenFileName(
        widget, 'Select a file', path)
    print (fname)
    if fname:
        fname = QDir.toNativeSeparators(fname)


    text_box.setText(fname)


class StandardItem(QStandardItem):
    def __init__(self, txt='', font_size=12, set_bold=False, color=QColor(0, 0, 0), is_file = False):
        super().__init__()

        fnt = QFont('Open Sans', font_size)
        fnt.setBold(set_bold)

        self.setEditable(False)
        self.setForeground(color)
        self.setFont(fnt)
        self.setText(txt)
        self.is_file = is_file


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('mainwindow.ui', self)
        self.show()
        self.load_dir_but.clicked.connect(lambda: load_dir_dialog(self, self.load_dir_lineEdit))
        self.load_drift_file_but.clicked.connect(lambda: load_file_dialog(self, self.load_drift_file_lineEdit))
        self.load_calibration_file_but.clicked.connect(lambda: load_file_dialog(self, self.load_calibration_file_lineEdit))

        def select_model_path(self, line_edit):
            self.model = None
            return load_dir_dialog(self, line_edit)
        self.select_model_path_but.clicked.connect(
            lambda: select_model_path(self, self.model_path_lineEdit))

        self.load_data.clicked.connect(self.load_data_func)
        self.label_data_but.clicked.connect(self.label_data_func)
        self.load_prefix_tables()
        self.datasets = Datasets()
        self.model = None

        self.labeled_dataset = None
        self.original_dataset = None


    def load_data_func(self):
        calibration_prefixes, sample_prefixes = self.get_prefixes_from_tables()

        # Load drift data
        drift_path = self.load_drift_file_lineEdit.text()
        self.datasets.read_drift_data(drift_path)

        drift_data = self.datasets.get_drift_data()
        self.driftDataModel = PandasModel(drift_data)
        self.driftDataTableView.setModel(self.driftDataModel)
        self.driftDataTableView.show()

        # Fill session tree
        session_files = self.datasets.get_session_dict()
        unique_sessions = self.datasets.get_sessions()

        tree = self.sessionTreeWidget
        tree.clear()
        
        for s in unique_sessions:
            session_item = QTreeWidgetItem(tree)
            session_item.setText(0, s)
            session_item.setFlags(session_item.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

            files = session_files[s]
            for f in files:
                file_item = QTreeWidgetItem(session_item)
                file_item.setFlags(file_item.flags() | Qt.ItemIsUserCheckable)
                file_item.setText(0, f)
                file_item.setCheckState(0, Qt.Unchecked)

        tree.setHeaderLabel("Session")

        self.sessionTreeWidget.doubleClicked.connect(self.onSessionTreeDoubleClick)

        # Load calibration data
        calibration_path = self.load_calibration_file_lineEdit.text()
        self.datasets.read_calibration_data(calibration_path)

        calibration_data = self.datasets.get_calibration_data()
        self.calibrationDataModel = PandasModel(calibration_data)
        self.calibrationDataTableView.setModel(self.calibrationDataModel)
        self.calibrationDataTableView.show()

        # Load datasets
        datasets_path = self.load_dir_lineEdit.text()
        self.datasets.read_datasets(datasets_path)

        # Load elements tree
        tree = self.labelElementsTreeWidget
        tree.clear()
        elements = [e for e in next(iter(self.datasets.get_datasets().values())).columns if e.lower() not in ['time', 'delay']]
        # automatically determine top elements to select for labelling
        data = next(iter(self.datasets.get_datasets().values()))
        selected_elements = auto_select_elements(data, num_to_select = 5)
        for e in elements:
            item = QTreeWidgetItem(tree)
            item.setText(0, e)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if e in selected_elements: item.setCheckState(0, Qt.Checked)
            else: item.setCheckState(0, Qt.Unchecked)


    def onSessionTreeDoubleClick(self,index):
        item = self.sessionTreeWidget.selectedIndexes()[0]
        is_file = not item.model().hasChildren(index)
        if is_file:
            filename = index.data()
            if filename not in self.datasets.get_datasets().keys():
                raise ValueError(f"File {filename} does not exist.\n Existing files are {self.datasets.get_datasets().keys()}")
            dataset = self.datasets.get_dataset(filename)
            self.datasetModel = PandasModel(dataset)
            self.sessionsDataTableView.setModel(self.datasetModel)
            self.sessionsDataTableView.show()


    def label_data_func(self):
        selected_datasets = get_checked_items_in_subtree(self.sessionTreeWidget)
        selected_elements = get_checked_items(self.labelElementsTreeWidget)
        model_path = self.model_path_lineEdit.text()
        if not self.model:
            self.model = load_model(model_path)

        select_per_sample = self.perSampleSelectionCheckBox.isChecked()
        self.datasets.labeled_datasets = predict_labels_on_selected_datasets(self.datasets,
                                                                             selected_datasets,
                                                                             selected_elements,
                                                                             self.model,
                                                                             select_per_sample)

        # Fill labeled session tree

        labeled_datasets = self.datasets.get_labeled_datasets()

        unique_sessions = labeled_datasets.keys()

        tree = self.labeledSessionTreeWidget

        tree.clear()
        for s in unique_sessions:
            session_item = QTreeWidgetItem(tree)
            session_item.setText(0, s)
            session_item.setFlags(session_item.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

            files = labeled_datasets[s]
            for f in files:
                file_item = QTreeWidgetItem(session_item)
                file_item.setFlags(file_item.flags() | Qt.ItemIsUserCheckable)
                file_item.setText(0, f)
                file_item.setCheckState(0, Qt.Unchecked)

        tree.setHeaderLabel("Session")

        self.labeledSessionTreeWidget.doubleClicked.connect(self.onLabeledSessionTreeDoubleClick)


        import pickle
        with open("labeled_dataset.pkl", "wb") as f:
            pickle.dump(self.datasets.get_labeled_datasets(), f)

    def onLabeledSessionTreeDoubleClick(self,index):
        item = self.labeledSessionTreeWidget.selectedIndexes()[0]
        is_file = not item.model().hasChildren(index)
        if is_file:
            filename = index.data()
            labeled_datasets = self.datasets.get_labeled_datasets()
            labeled_files = {f: d for s in labeled_datasets.keys() for f, d in labeled_datasets[s].items()}
            if filename not in labeled_files.keys():
                raise ValueError(f"File {filename} does not exist.\n Existing files are {labeled_files.keys()}")
            self.labeled_dataset = labeled_files[filename]

            self.original_dataset = self.datasets.data_files[filename]
            fields = self.original_dataset.columns
            tree = self.plotFieldTreeWidget
            tree.clear()
            for f in fields:
                item = QTreeWidgetItem(tree)
                item.setText(0, f)

            plot_labeled_dataset(self.labeled_dataset, self.labeledPlotWidget)
            self.plotFieldTreeWidget.doubleClicked.connect(self.onPlotFieldTreeDoubleClick)

    def onPlotFieldTreeDoubleClick(self, index):
        if self.labeled_dataset is not None:
            item = self.labeledSessionTreeWidget.selectedIndexes()[0]
            field = index.data()
            labels = self.labeled_dataset[1].argmax(axis=2)[0,:]
            plot_labeled_dataset_field(self.original_dataset, labels, field, self.labeledPlotWidget)

    def get_prefixes_from_tables(self):

        table = self.calibrationPrefixTableWidget
        calibration_prefixes = [table.item(row, 0).text() for row in range(table.rowCount())]
        calibration_prefixes = [text for text in calibration_prefixes if len(text) > 0]

        table = self.samplePrefixTableWidget
        sample_prefixes = [table.item(row, 0).text() for row in range(table.rowCount())]
        sample_prefixes = [text for text in sample_prefixes if len(text) > 0]

        return calibration_prefixes, sample_prefixes

    def load_prefix_tables(self):
        default_calibration_prefixes = ["FEBS", "NIES", "MACS3", "NIST612", "NIST614", "NIST616"]
        default_sample_prefixes = ["SL1"]

        self.calibrationPrefixTableWidget.setRowCount(len(default_calibration_prefixes))
        self.calibrationPrefixTableWidget.setColumnCount(1)
        self.samplePrefixTableWidget.setRowCount(len(default_sample_prefixes))
        self.samplePrefixTableWidget.setColumnCount(1)

        for i, prefix in enumerate(default_calibration_prefixes):
            self.calibrationPrefixTableWidget.setItem(i, 0, QTableWidgetItem(prefix))

        for i, prefix in enumerate(default_sample_prefixes):
            self.samplePrefixTableWidget.setItem(i, 0, QTableWidgetItem(prefix))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()
