import numpy as np
import tensorflow as tf
import tensorflow.keras

def predict_labels_on_selected_datasets(datasets, selected_datasets, model_path):
    model = load_model(model_path)
    selected_datasets = {s:
                            {f:datasets.get_dataset(f) for f in selected_datasets[s]}
                             for s in selected_datasets.keys() if len(selected_datasets[s]) > 0
                         }

    labeled_datasets = {s:
                            {f: (prepare_input_data(selected_datasets[s][f]), predict(selected_datasets[s][f], model)) for f in selected_datasets[s]}
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

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict(data, model):

    X = prepare_input_data(data)
    y = model.predict(X)
    return(y)


