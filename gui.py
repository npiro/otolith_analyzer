from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import numpy as np

class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])

class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None


def get_checked_items(treeWidget):
    checked = dict()
    root = treeWidget.invisibleRootItem()
    signal_count = root.childCount()

    for i in range(signal_count):
        signal = root.child(i)
        checked_sweeps = list()
        num_children = signal.childCount()

        for n in range(num_children):
            child = signal.child(n)

            if child.checkState(0) == QtCore.Qt.Checked:
                checked_sweeps.append(child.text(0))

        checked[signal.text(0)] = checked_sweeps

    return checked

def plot_labeled_dataset(dataset, plot_widget):

    y = dataset[0][0,:,0]
    t = np.array(range(y.shape[0]))
    c = dataset[1].argmax(axis=2)[0,:]

    plt = plot_widget

    plt.clear()
    # draw a scatter plot with selected points in yellow
    cmap = {0: (0, 0, 200), 1: (255, 0, 0), 2: (0, 255, 0)}
    brushes = [pg.mkBrush(cmap[x]) for x in c]
    plt.plot(t, y, pen=None, symbol='o', symbolBrush=brushes)

    # draw vertical ticks marking the position of selected points
    #tick_x = np.array(t)[c]
    #ticks = pg.VTickGroup(tick_x, yrange=[0, 0.1], pen={'color': 'w', 'width': 5})
    #plt.addItem(ticks)

    # add a vertical line with marker at the bottom for each selected point
    #for tx in tick_x:
    #    l = plt.addLine(x=tx, pen=(50, 150, 50), markers=[('^', 0, 10)])


    #plot_widget.plot(t,y)

def plot_labeled_dataset_field(original_dataset, labels, field, plot_widget):

    y = original_dataset[field]
    t = original_dataset["time"]
    c = labels

    plt = plot_widget

    plt.clear()
    # draw a scatter plot with selected points in yellow
    cmap = {0: (0, 0, 200), 1: (255, 0, 0), 2: (0, 255, 0)}
    brushes = [pg.mkBrush(cmap[x]) for x in c]
    plt.plot(t, y, pen=None, symbol='o', symbolBrush=brushes)