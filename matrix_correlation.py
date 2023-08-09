import joblib
import numpy as np
from epoch_fitting import timeseries_decimated
from MCCA import MCCA

sensor = {'lw1': '/data2/jpanzay1/cachedir/sensorweight_lw1_decimated.pkl',
          'cable_spool': '/data2/jpanzay1/cachedir/sensorweight_cable_decimated.pkl',
          'easy_money': '/data2/jpanzay1/cachedir/sensorweight_easy_decimated.pkl',
          'black_willow': '/data2/jpanzay1/cachedir/sensorweight_black_decimated.pkl',
          'all': '/data2/jpanzay1/cachedir/sensorweight_all_decimated.pkl'}

mcca_scores = {'lw1': '/data2/jpanzay1/cachedir/mccascores_lw1_decimated.pkl',
               'cable_spool': '/data2/jpanzay1/cachedir/mccascores_cable_decimated.pkl',
               'easy_money': '/data2/jpanzay1/cachedir/mccascores_easy_decimated.pkl',
               'black_willow': '/data2/jpanzay1/cachedir/mccascores_black_decimated.pkl',
               'all': '/data2/jpanzay1/cachedir/mccascores_all_decimated.pkl'}


def extract_timeseries(file_key):
    with open(timeseries_decimated[file_key],'rb') as file:
        X = np.array(joblib.load(file))
    return X


if __name__ == "__main__":
    X = extract_timeseries('lw1')
    # define the MCCA parameters and obtain the scores
    mcca = MCCA(n_components_pca=50, n_components_mcca=24, reg=False, r=0, pca_only=False)
    scores = mcca.obtain_mcca(X)
    # invert the weights so that the matrix multiplication can be done (n,k)*(k,m) = (n,m)
    pca_w_inverse = np.linalg.pinv(mcca.weights_pca)
    cc_w_inverse = np.linalg.pinv(mcca.weights_mcca)
    # our final result of the multiplication should be sensors * cc
    sensor_weight = np.matmul(cc_w_inverse, pca_w_inverse)

    with open(sensor['lw1'], "wb") as sensor_file:
        joblib.dump(sensor_weight, sensor_file)

    with open(mcca_scores['lw1'], "wb") as scores_file:
        joblib.dump(scores, scores_file)
