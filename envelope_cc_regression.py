from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import ceil
import numpy as np
from scipy.stats import zscore
import joblib
from envelope_correlation import envelope_file, mcca_scores_corrected
import matplotlib.pyplot as plt


def normalized_beta(envelope, scores):
    # The length of our time series of CC scores and the envelope should be the same, in our case there can be some
    # differences of a couple of samples. We take the difference between the arrays and substract the difference from
    # the envelope.
    difference = ceil((np.shape(envelope)[0] - np.shape(scores)[1]))
    envelope_rs = envelope[:int(len(envelope))-int(difference)]
    # Create arrays representing the predictor variable (x) and the response variable (y)
    x = zscore(np.mean(scores[:, :, :], axis=0), axis=0)
    y = envelope_rs
    # Create a linear regression object
    regression = LinearRegression()
    # Fit the model on the training set
    regression.fit(x, y)
    betas = regression.coef_
    # Compute the normalization factor
    norm_factor = np.sqrt(np.sum(betas ** 2))
    # Normalize the betas
    R = betas / norm_factor
    return R


def get_rmse(envelope, scores):
    # The length of our time series of CC scores and the envelope should be the same, in our case there can be some
    # differences of a couple of samples. We take the difference between the arrays and substract the difference from
    # the envelope.
    difference = ceil((np.shape(envelope)[0] - np.shape(scores)[1]))
    envelope_rs = envelope[:int(len(envelope)) - int(difference)]
    # Create arrays representing the predictor variable (x) and the response variable (y)
    x = zscore(np.mean(scores[:, :], axis=0), axis=0)
    y = envelope_rs
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y, test_size=0.25, random_state=42)
    # Fit the linear regression model on the training data
    reg = LinearRegression().fit(X_train, y_train)
    # Predict values for the test set
    y_pred = reg.predict(X_test)
    # Calculate the RMSE for the test set
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse_test

if __name__ == "__main__":
    with open(envelope_file['lw1'], 'rb') as e_file:
        lw1 = np.array(joblib.load(e_file))
    #
    with open(envelope_file['cable_spool'], 'rb') as cable_file:
        cable_spool = np.array(joblib.load(cable_file))
    #
    with open(envelope_file['easy_money'], 'rb') as easy_file:
        easy_money = np.array(joblib.load(easy_file))
    #
    with open(envelope_file['black_willow'], 'rb') as black_file:
        black_willow = np.array(joblib.load(black_file))
    #
    with open(mcca_scores_corrected['all'], "rb") as file:
        scores = np.array(joblib.load(file))
    #
    envelope = np.concatenate((lw1, cable_spool, easy_money, black_willow))
    R = normalized_beta(envelope, scores)
    # Create a bar plot of R
    fig, ax = plt.subplots()
    ax.bar(range(len(R)), R, color='skyblue')
    # Add axis labels and a title
    x_T = np.arange(0, 24, 4)
    x_L = np.arange(1, 24, 4)
    plt.xticks(x_T, x_L)
    ax.set_xlabel('Canonical Components')
    ax.set_ylabel('Normalized beta coefficient')
    ax.set_title('Normalized beta coefficients of the linear regression \n between canonical components and sound envelope')
    # Display the plot
    plt.show()
    cc = np.arange(0, 24).tolist()
    rmse_all = []
    for c in cc:
        rmse_cc = get_rmse(envelope, scores[:, :, c])
        rmse_all.append(rmse_cc)
    rmse_array = np.asarray(rmse_all)
    # Normalize the RMSE values using Min-Max normalization
    normalized_rmse = (rmse_array - rmse_array.min()) / (rmse_array.max() - rmse_array.min())
    fig_1, ax_1 = plt.subplots()
    ax_1.bar(range(len(rmse_array)), rmse_array, color='pink')
    # Add axis labels and a title
    ax_1.set_xlabel('Canonical Components')
    ax_1.set_ylabel('Normalized RMSE')
    ax_1.set_title('Normalized RMSE for the linear regression model fit for \n predicting sound envelope using canonical components')
    plt.xticks(x_T, x_L)
    plt.ylim([0, 1.1])
    # Display the plot
    plt.show()
