import os
import joblib
import numpy as np
from epoch_story import save_epoch

# define a file name to save our whole timeseries
timeseries_decimated = {'lw1': '/data2/jpanzay1/cachedir/timeseries_lw1_decimated.pkl',
                        'cable_spool': '/data2/jpanzay1/cachedir/timeseries_cable_decimated.pkl',
                        'easy_money': '/data2/jpanzay1/cachedir/timeseries_easy_decimated.pkl',
                        'black_willow': '/data2/jpanzay1/cachedir/timeseries_black_decimated.pkl'}


def get_session_timeseries(epoch_file):
    # create an empty 3D array in which we can append our timeseries
    #session_timeseries = np.zeros((1, 1, 208))
    story = []
    # open the file and extract the information
    with open(epoch_file, "rb") as file:
        epoch = joblib.load(file)
        # our epochs are saved in a list format, we need to go to each epoch and extract the information
        for idx, event in enumerate(epoch[3]):
            print(idx)
            for segment in event:
                data = segment * 1e13
                # swap the axes so that the channels go to the third dimension
                n_by_time = np.swapaxes(data, 0, 1)
                # stack all the epochs over the second dimension so that we have the whole timeseries for a session
                story.append(n_by_time)
                #session_timeseries = np.hstack((session_timeseries, n_by_time))
    session_timeseries = np.concatenate((np.array(story[0]), np.array(story[1]), np.array(story[2]), np.array(story[3]),
                                         np.array(story[4]), np.array(story[5]), np.array(story[6]), np.array(story[7]),
                                         np.array(story[8]), np.array(story[9]), np.array(story[10]), np.array(story[11])))
    # save the length of the timeseries
    timeseries_length = session_timeseries.shape[0]
    return session_timeseries, timeseries_length


# def all_subjects_timeseries(timeseries, lengths):
#     # transform our length list into an array
#     lengths_array = np.array(lengths)
#     maximum_length = lengths_array.max()
#     # find the indices for when the length matches the biggest length present, the one we want to maintain
#     indexes = np.where(lengths_array == maximum_length)
#     subjects_timeseries = []
#     # iterate over the indices that match our condition and extract just that information
#     for idx in indexes[0]:
#         good_timeseries = np.array(timeseries[idx])
#         subjects_timeseries.append(good_timeseries)
#     with open(timeseries_file, "wb") as file:
#         joblib.dump(subjects_timeseries, file)


if __name__ == "__main__":
    # give the values of the number of subjects and sessions
    subjects = [str(sub).rjust(2, "0") for sub in range(1, 28)]
    sessions = ["0", "1"]
    n_by_timeseries = []
    all_lengths = []
    # obtaining the epochs for each session of each subject by calling the function
    for subject in subjects:
        print(subject)
        for session in sessions:
            print(session)
            epoch_file = save_epoch.format(subject, session)
            # if the file does not exist the function will end
            if not os.path.isfile(epoch_file):
                print("missing session: ", session)
                continue
            session_timeseries, timeseries_length = get_session_timeseries(epoch_file)
            n_by_timeseries.append(session_timeseries)
            all_lengths.append(timeseries_length)
    # get the aligned data of our subjects
    with open(timeseries_decimated['black_willow'], "wb") as file:
        joblib.dump(n_by_timeseries, file)