import joblib
import numpy as np
from math import ceil
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
from envelope_wav_file import envelope_file_decimated
from matrix_correlation import mcca_scores

envelope_file = {'lw1': '/data2/jpanzay1/cachedir/story_lw1.pkl',
                 'cable_spool': '/data2/jpanzay1/cachedir/story_cable_spool.pkl',
                 'easy_money': '/data2/jpanzay1/cachedir/story_easy_money.pkl',
                 'black_willow': '/data2/jpanzay1/cachedir/story_black_willow.pkl'}

mcca_scores_corrected = {'lw1': '/data2/jpanzay1/cachedir/scores_mcca_24_corrected_lw1.pkl',
                         'cable_spool': '/data2/jpanzay1/cachedir/scores_mcca_24_corrected_cable.pkl',
                         'easy_money': '/data2/jpanzay1/cachedir/scores_mcca_24_corrected_easy.pkl',
                         'black_willow': '/data2/jpanzay1/cachedir/scores_mcca_24_corrected_willow.pkl',
                         'all': '/data2/jpanzay1/cachedir/scores_mcca_24_corrected.pkl'}


def get_corr_coef(envelope, scores):
    # The length of our time series of CC scores and the envelope should be the same, in our case there can be some
    # differences of a couple of samples. We take the difference between the arrays and substract the difference from
    # the envelope.
    difference = ceil((np.shape(envelope)[0] - np.shape(scores)[1]))
    envelope_rs = envelope[:int(len(envelope)) - int(difference)]
    subjects = np.arange(0, 39).tolist()
    cc = np.arange(0, 24).tolist()
    cc_corr = []
    cc_pv = []
    # We iterate over all the canonical components we extracted, and we have to remember that in our array of CC we
    # have in the first dimension subjects, and in the third dimension the components.
    for r in cc:
        subject_corr = []
        subject_pvalue = []
        for i in subjects:
            # Using scipy we can get the pearson coefficient between the envelope, and the scores for one subject and
            # one canonical component. We iterate this for the same subject over all components and then for all
            # subjects.
            correlation, pvalue = scipy.stats.pearsonr(envelope_rs, scores[i, :, r])
            subject_corr.append(correlation)
            subject_pvalue.append(pvalue)
        cc_corr.append(np.vstack(np.array(subject_corr)))
        cc_pv.append(np.vstack(np.array(subject_pvalue)))
    # This variable will contain the correlation scores and p values between all subjects and all canonical
    # components with the sound envelope.
    total_corr = np.hstack(np.array(cc_corr))
    total_pv = np.hstack(np.array(cc_pv))
    return total_corr, total_pv


def coef_t_test(total_corr):
    # We are calculating a t test between all subjects for the same canonical component, we want to check if the
    # distribution across all subjects is similar for the same canonical component.
    t_stat, p_val = scipy.stats.ttest_1samp(total_corr, popmean=0, axis=0)
    # Save the resulting scores and pv values as a data frame
    t_results = pd.DataFrame(np.vstack((t_stat, p_val)), columns=list(np.arange(1, 25, 1)))
    labels = pd.DataFrame(['t test value', 'p value'])
    t_results.insert(0, 'Canonical Components', labels)
    # Save the data frame as an excel table
    return t_results.to_excel(r'/data2/jpanzay1/cachedir/t_test_results.xlsx', index=False)


if __name__ == "__main__":
    # Extract the timeseries of cc scores
    with open(mcca_scores_corrected['all'], "rb") as file:
        scores = np.array(joblib.load(file))
    # Extract the envelope of the sound and concatenate it into one continuous array, to match the CC scores.
    with open(envelope_file['lw1'], 'rb') as e_file:
        lw1 = np.array(joblib.load(e_file))
    with open(envelope_file['cable_spool'], 'rb') as cable_file:
        cable_spool = np.array(joblib.load(cable_file))
    with open(envelope_file['easy_money'], 'rb') as easy_file:
        easy_money = np.array(joblib.load(easy_file))
    with open(envelope_file['black_willow'], 'rb') as black_file:
        black_willow = np.array(joblib.load(black_file))
    envelope = np.concatenate((lw1, cable_spool, easy_money, black_willow))
    # Calculate our correlation coefficients
    total_corr, total_pv = get_corr_coef(envelope, scores)
    # Define the ticks for our plot, we have 39 subjects and 24 canonical components
    y_T = np.arange(0, 39, 4)
    y_L = np.arange(1, 39, 4)
    x_T = np.arange(0, 24, 3)
    x_L = np.arange(1, 24, 3)
    # Plot a scaled image of the correlation coefficients
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(total_corr, vmin=-0.11, vmax=0.11)
    plt.colorbar(im)
    plt.xticks(x_T, x_L)
    plt.yticks(y_T, y_L)
    plt.xlabel('Canonical Components')
    plt.ylabel('Subjects')
    plt.title('Correlation coefficients of \n sound envelope with canonical components', loc='center')
    plt.show()
    # Do the t test and save the results
    coef_t_test(total_corr)


