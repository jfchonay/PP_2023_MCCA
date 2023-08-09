import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from epoching import save_filename
from logistic_regression import getting_parameters

save_image = "/data2/jchonay/cachedir/ttestsub-{0}_ses-{1}"


def ttest_classes(epoch_file, image_file):
    # extract the information of the epochs and extract the shape parameters
    X, y = getting_parameters(epoch_file)
    n, nchans, ntimes = X.shape
    X = X.reshape(n, -1)
    X_verb = []
    X_noun = []
    # use a loop to separate the epochs according to its class, either verb or noun
    for index, label in enumerate(y):
        if label == 0:
            X_verb.append(X[index])
        elif label == 1:
            X_noun.append(X[index])
    # save these classes as arrays
    X_verb = np.asarray(X_verb)
    X_noun = np.asarray(X_noun)
    # do a independent different variance t test on these two classes
    stats, pvalue = ttest_ind(X_noun, X_verb, axis=0, equal_var=False)
    # reshape the results so that they resembled the epoch data
    stats_reshape = stats.reshape(nchans, ntimes)
    pvalue_reshape = pvalue.reshape(nchans, ntimes)
    # create a boolean matrix that is True where the p value of the t test is lower than 0.05,
    # thus rejecting the null hypothesis
    significant_pvalue = np.where(pvalue_reshape <= 0.05, pvalue_reshape, 0)
    significant_pvalue = significant_pvalue.astype(dtype=bool)
    # using the imshow function to plot the t test values and overlapping another image that shows where
    # these results are significant using the boolean matrix that we created
    # RGBA overlay matrix
    overlay = np.zeros((nchans, ntimes, 4))
    # setting the red channel of the overlay to 1
    overlay[..., 0] = 1
    # setting the alpha to our boolean matrix, so it is transparent except when its TRUE
    overlay[..., 3] = significant_pvalue
    # creating the subplot
    fig, ax = plt.subplots(1, 1)
    # defining our imshow plot of the t test values
    im = ax.imshow(stats_reshape, interpolation='nearest')
    plt.colorbar(im)
    plt.xlabel('Time points')
    plt.ylabel('Channels')
    # adding the layer of significant p values
    h = ax.imshow(overlay, interpolation='nearest', visible=True)
    plt.savefig(image_file, format='svg')


if __name__ == "__main__":
    # define the number of subjects and sessions, also the window size
    subjects = [str(sub).rjust(2, "0") for sub in range(1, 28)]
    sessions = ["0", "1"]
    for subject in subjects:
        print(subject)
        for session in sessions:
            # if there is no session for our subject skip to the next session
            epoch_file = save_filename.format(subject, session)
            if not os.path.isfile(epoch_file):
                print("missing session: ", session)
                continue
            image_file = save_image.format(subject, session)
            if os.path.isfile(image_file):
                print("graph already exists")
                continue
            print(session)
            # use our functions
            ttest_classes(epoch_file, image_file)