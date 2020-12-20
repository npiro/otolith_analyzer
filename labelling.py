import numpy as np
import scipy as sp
from scipy import interpolate
import tensorflow as tf
import tensorflow.keras
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

def auto_select_elements(data, num_to_select = 5):
    def standardize_data(X):
        ss = StandardScaler()
        return ss.fit_transform(X)

    def remove_outliers(X):
        """
        Remove outliers
        df: dataframe
        c: column to apply to
        """

        db = DBSCAN(eps=0.5)
        db_fit = db.fit(X)
        mask = db_fit.labels_ != -1

        return X[mask]

    def compute_range_ratio(X):
        sc = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')
        sc_fit = sc.fit(X)
        pp = sc_fit.labels_
        x0 = X[pp == 0][:, 0]
        x1 = X[pp == 1][:, 0]
        x2 = X[pp == 2][:, 0]
        x_range = max(X[:, 0]) - min(X[:, 0])
        return (max(x0) - min(x0)) / x_range + (max(x1) - min(x1)) / x_range + (max(x2) - min(x2)) / x_range

    def compute_mean_distance(X):
        mean = X.mean()
        upper = X[X > mean]
        lower = X[X <= mean]
        return (upper.mean() - lower.mean()) / upper.std()

    def compute_metric(df, c, threshold=1.05):
        X = df[["time", c]].values
        X = standardize_data(X)
        X = remove_outliers(X)

        rr = compute_range_ratio(X)
        dd = compute_mean_distance(X)

        metric = dd * int(rr < threshold)
        return metric

    df = data
    cols = [c for c in df.columns if c.lower() not in ['time','delay']]
    metrics = {c: compute_metric(df, c) for c in cols}
    order = sorted(metrics, key=metrics.get, reverse=True)
    return order[:num_to_select]

def predict_labels_on_selected_datasets(datasets, selected_datasets, selected_elements, model, select_per_sample = False):
    selected_datasets = {s:
                            {f:datasets.get_dataset(f) for f in selected_datasets[s]}
                             for s in selected_datasets.keys() if len(selected_datasets[s]) > 0
                         }


    if select_per_sample:
        fields = {s:
                     {f:["time"]+auto_select_elements(datasets.get_dataset(f), num_to_select = 5) for f in selected_datasets[s]}
                      for s in selected_datasets.keys() if len(selected_datasets[s]) > 0
                 }
    else:
        fields = {s:
                      {f: ["time"]+selected_elements for f in
                       selected_datasets[s]}
                  for s in selected_datasets.keys() if len(selected_datasets[s]) > 0
                  }

    labeled_datasets = {s:
                            {f: (prepare_input_data_v2(selected_datasets[s][f].loc[:,fields[s][f]]),
                                 predict(selected_datasets[s][f].loc[:,fields[s][f]], model)) for f in selected_datasets[s]}
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


