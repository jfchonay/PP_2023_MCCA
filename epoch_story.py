import os
import mne_bids
from mne_bids import read_raw_bids, BIDSPath
from pandas import DataFrame
import pandas as pd
import numpy as np
import mne
import joblib
import ancpbids

# setting the path to our raw data
path = '/data2/jpanzay1/meg-masc/bids_anonym/'
save_epoch = '/data2/jpanzay1/cachedir/decimated_epochs-{0}_ses-{1}.pkl'


def get_epochs_aligned(subject, session, epochfile):
    task_lw1 = []
    task_cable = []
    task_easy = []
    task_black = []
    # iterate over all tasks
    for task in tasks:
        bids_path = BIDSPath(
                    subject=subject,
                    session=str(session),
                    task=str(task),
                    datatype="meg",
                    root=path)
        try:
            raw = read_raw_bids(bids_path)
        except FileNotFoundError:
            print("missing", task)
            continue
        # read the raw object created by the bids path
        raw = raw.pick_types(meg=True, misc=False, eeg=False, eog=False, ecg=False)
        raw.load_data().filter(0.5, 30.0, n_jobs=1)
        # meta is going to be saved as a list.
        meta = list()
        # we are going to use the annotations to obtain the events,
        # but first we need to transform the annotations from arrays to data frame.
        for annot in raw.annotations:
            # we look at the description part of the annotations to creat the meta list.
            # the variable d will take each annotation separately
            d = eval(annot.pop("description"))
            for k, v in annot.items():
                assert k not in d.keys()
                d[k] = v
            meta.append(d)
        meta = DataFrame(meta)
        meta["intercept"] = 1.0
        # we are going to use the cut points in the story to separate each task and obtain each segment,
        # for this reason we look at the higher sound_id value which tells us how many cut points exist for this task.
        sound_array = np.arange(meta.sound_id.max()+1)
        # then iterating over all the different values of sound_id will allow us to separate our data
        for id in sound_array:
            # we extract the information for a specific value of the sound_id
            task_section = meta.loc[meta.sound_id == id]
            # we determine the size in seconds of this segment, by using the start and end information
            # provided for the segment
            section_size = task_section.start.max() - task_section.start.min() # in seconds
            # we determine what is the first index of this section, as this first index contains the onset value
            first_index = task_section.index.values[task_section.index.values.argsort()[0]]
            # with the index we can obtain the onset value and multiply it by the sampling frequency
            onset = task_section.onset[first_index] * raw.info['sfreq']
            # create an object event that the function Epochs can understand
            event = np.c_[onset, 1, 1].astype(int)
            # use the information to create the epochs
            epoch_section = mne.Epochs(
                raw,
                event,
                tmin=0,
                tmax=section_size,
                baseline=(0, 0), # we are not applying any baseline
                decim=50, # decimation of 50 fold to have 20 Hz
                preload=True,
            )
            # depending on the story that was used we save the epochs in a specific list
            if meta['story'][0] == 'lw1':
                task_lw1.append(epoch_section)
            elif meta['story'][0] == 'cable_spool_fort':
                task_cable.append(epoch_section)
            elif meta['story'][0] == 'easy_money':
                task_easy.append(epoch_section)
            elif meta['story'][0] == 'The_Black_Willow':
                task_black.append(epoch_section)
    # get all the epochs for a session
    session_epochs = []
    # we check if all the stimuli were present and only under this condition we create the file
    if task_lw1 and task_cable and task_easy and task_black:
        # we put all of our sessions in the same order
        session_epochs = [task_lw1, task_cable, task_easy, task_black]
        with open(epochfile, "wb") as file:
            joblib.dump(session_epochs, file)
    else:
        print('Warning: file not created because task is missing')


if __name__ == "__main__":
    # give the values of the number of subjects and sessions
    subjects = [str(sub).rjust(2, "0") for sub in range(1, 28)]
    sessions = ["0", "1"]
    layout = ancpbids.BIDSLayout(path)
    tasks = layout.get_tasks()
    # obtaining the epochs for each session of each subject by calling the function
    for subject in subjects:
        print(subject)
        for session in sessions:
            epochfile = save_epoch.format(subject, session)
            # if the file already exists the function will end
            if os.path.isfile(epochfile):
                print("file already exists")
                continue
            get_epochs_aligned(subject, session, epochfile)
            print("done")