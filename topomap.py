import mne
import matplotlib.pyplot as plt
import numpy as np
import joblib
import mne_bids

# In this scrpit we want to visualize our canonical components using the layout of the MEG sensors
scores_files = {'lw1': '/data2/jpanzay1/cachedir/sensor_weight_corrected_lw1.pkl',
               'cable_spool': '/data2/jpanzay1/cachedir/sensor_weight_corrected_cable.pkl',
               'easy_money': '/data2/jpanzay1/cachedir/sensor_weight_corrected_easy.pkl',
               'black_willow': '/data2/jpanzay1/cachedir/sensor_weight_corrected_willow.pkl'}


with open(scores_files['lw1'], "rb") as file:
    sensor_weights = np.array(joblib.load(file))
# Access the raw data information to oextract the channel layout
path = '/data2/jpanzay1/meg-masc/bids_anonym/'
raw = mne_bids.read_raw_bids(mne_bids.BIDSPath(
            subject='02',
            session=str('0'),
            task=str('0'),
            datatype="meg",
            root=path))
# Pick only the magnetometers, in our case the first 208 channels.
ch_pick = np.arange(0, 208).tolist()
info_meg = mne.pick_info(raw.info, ch_pick, True)
n_subjects = [0, 1, 2]
cc_selected = [0, 1, 2]
# Set up the figure and axes handle
fig, axes = plt.subplots(np.size(n_subjects), np.size(cc_selected), squeeze=False)
# Iterate over how many CC we want and plot them
for r in cc_selected:
    axes[0, r].set_title(r'cc = %d' % cc_selected[r])
    for i in n_subjects:
        # Using the mne function we input the sensor weights and the info object.
        mne.viz.plot_topomap(sensor_weights[i, r, :], info_meg, show=False, axes=axes[i, r])
for i in n_subjects:
    axes[i, 0].set_ylabel('Subject %d' % (i + 1))
plt.show()