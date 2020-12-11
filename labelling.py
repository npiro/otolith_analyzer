import numpy as np
import scipy as sp
from scipy import interpolate
import tensorflow as tf
import tensorflow.keras

def predict_labels_on_selected_datasets(datasets, selected_datasets, selected_elements, model):
    selected_datasets = {s:
                            {f:datasets.get_dataset(f) for f in selected_datasets[s]}
                             for s in selected_datasets.keys() if len(selected_datasets[s]) > 0
                         }
    fields = ["time"]+selected_elements
    labeled_datasets = {s:
                            {f: (prepare_input_data(selected_datasets[s][f].loc[:,fields]),
                                 predict(selected_datasets[s][f].loc[:,fields], model)) for f in selected_datasets[s]}
                             for s in selected_datasets.keys()
                       }
    return labeled_datasets

def prepare_input_data(df):
    """
    Prepares data into the right format for model
    :param df: Dataframe containing the mass spectrometer data
    :return:
    """
    df.columns = df.columns.str.lower()
    df['delay'] = df['time'].diff().fillna(df['time'][0])

    x = df.values
    sh = x.shape
    x = x.reshape((1,sh[0],sh[1]))
    x = np.stack([x.mean(axis=2), x.std(axis=2)], axis=2)
    m = np.mean(x, axis = (0,1))
    s = np.std(x, axis = (0,1))
    x_norm = (x-m)/s
    return x_norm



def prepare_input_data_v2(df):
    feature_cols_ = [d for d in df.columns if d not in ['delay','Time','time','label']]
    x = df[feature_cols_].values
    times = df.Time if 'Time' in df.columns else df.time

    sh = x.shape
    x = x.reshape((1, sh[0], sh[1]))
    x = np.stack([x.mean(axis=2), x.std(axis=2)], axis=2)

    tf = np.linspace(0, max(times), 120)
    x_new = np.transpose(np.stack([[interpolate.interp1d(times, x[0, :, 0], fill_value="extrapolate",
                                                            kind="slinear")(tf),
                                    interpolate.interp1d(times, x[0, :, 1], fill_value="extrapolate",
                                                            kind="slinear")(tf)] for n in range(x.shape[0])], axis=2))
    m = np.mean(x_new, axis=(0, 1))
    s = np.std(x_new, axis=(0, 1))
    x_norm = (x_new - m) / s

    return x_norm

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict(data, model):

    X = prepare_input_data_v2(data)
    y = model.predict(X)
    return(y)


