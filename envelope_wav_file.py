import numpy as np
import scipy.io.wavfile as wav
from scipy import stats
import librosa as lbr
from scipy.signal import hilbert, resample, decimate
import matplotlib.pyplot as plt
import joblib

# create a dictionary to easily access all the wav files
story_files = {'lw1': '/data2/jpanzay1/meg-masc/bids_anonym/stimuli/audio/lw1_{0}.wav',
               'cable_spool': '/data2/jpanzay1/meg-masc/bids_anonym/stimuli/audio/cable_spool_fort_{0}.wav',
               'easy_money': '/data2/jpanzay1/meg-masc/bids_anonym/stimuli/audio/easy_money_{0}.wav',
               'black_willow': '/data2/jpanzay1/meg-masc/bids_anonym/stimuli/audio/the_black_willow_{0}.wav'}

envelope_file_decimated = {'lw1': '/data2/jpanzay1/cachedir/decimated_lw1.pkl',
                 'cable_spool': '/data2/jpanzay1/cachedir/decimated_cable.pkl',
                 'easy_money': '/data2/jpanzay1/cachedir/decimated_easy.pkl',
                 'black_willow': '/data2/jpanzay1/cachedir/decimated_black.pkl'}

def get_mel_spectrogram(filename, log=True, sr=22050, hop_length=512, **kwargs):
    #    '''Returns the (log) Mel spectrogram of a given wav file, the sampling rate
    #    of that spectrogram and names of the frequencies in the Mel spectrogram
    #
    #    Parameters
    #    ----------
    #    filename : str, path to wav file to be converted
    #    sr : int, sampling rate for wav file
    #         if this differs from actual sampling rate in wav it will be resampled
    #    log : bool, indicates if log mel spectrogram will be returned
    #    kwargs : additional keyword arguments that will be
    #             transferred to librosa's melspectrogram function
    #
    #    Returns
    #    -------
    #    a tuple consisting of the Melspectrogram of shape (time, mels), the
    #    repetition time in seconds, and the frequencies of the Mel filters in Hertz
    #    '''
    wav, _ = lbr.load(filename, sr=sr)
    melspecgrams = lbr.feature.melspectrogram(y=wav, sr=sr, hop_length=hop_length,
                                              **kwargs)
    if log:
        melspecgrams[np.isclose(melspecgrams, 0)] = np.finfo(melspecgrams.dtype).eps
        melspecgrams = np.log(melspecgrams)
    log_dict = {True: 'Log ', False: ''}
    freqs = lbr.core.mel_frequencies(
        **{param: kwargs[param] for param in ['n_mels', 'fmin', 'fmax', 'htk']
           if param in kwargs})
    freqs = ['{0:.0f} Hz ({1}Mel)'.format(freq, log_dict[log]) for freq in freqs]
    return melspecgrams.T, sr / hop_length, freqs


if __name__ == "__main__":
    output_folder = '/data2/jpanzay1/cachedir/'
    segments = np.arange(0, 12)
    story = []
    for segment in segments:
        wav_file = story_files['black_willow'].format(segment)
        print("Converting ", wav_file)
        # sig2, Fs = lbr.load(wav_file, sr=None, mono=True)
        rate, sig = wav.read(wav_file)
        if len(sig.shape) > 1:
            sig = np.mean(sig, axis=1)  # convert a WAV from stereo to mono
        # set parameters
        winlen = int(np.rint(rate * 0.1))  # 37485 Window length std 850 ms
        overlap = int(np.rint(rate * 0.0509))
        hoplen = winlen - overlap  # 661 hop_length
        nfft = winlen  # standard is = winlen = 1102 ... winlen*2 = 2204 ... nfft = the FFT size. Default for speech
        # processing is 512.
        nmel = 32  # n = the number of cepstrum to return = the number of filters in the filterbank
        lowfreq = 100  # lowest band edge of mel filters. In Hz
        highfreq = 8000  # highest band edge of mel filters. In Hz
        noay = 3  # subplot: y-dimension
        noax = 1  # subplot: x-dimension
        #                foursec     = int(np.rint(rate*4.0)) # 4 seconds
        #                start_time  = 0

        config = {"n_fft": nfft, "sr": rate, "win_length": winlen,
                  "hop_length": hoplen, "n_mels": nmel, "fmax": highfreq,
                  "fmin": lowfreq}
        melspec, sr_spec, freqs = get_mel_spectrogram(wav_file, **config)

        zscore_melspec = stats.zscore(melspec, axis=0)
        zband_envelope = np.abs(hilbert(melspec, axis=0))
        # band_envelope = np.abs(hilbert(melspec, axis=0))

        # log_mean_zband_envelope = np.log1p(np.mean(zband_envelope,axis=1))
        # log_mean_band_envelope = np.log1p(np.mean(band_envelope,axis=1))

        mean_zband_envelope = np.mean(zband_envelope, axis=1)

        # new_rate = 22000
        # n_samples = round(len(mean_zband_envelope) * float(new_rate) / sr_spec)
        # data = resample(mean_zband_envelope, n_samples, axis=0)
        # runs = [0, 1, 2]
        # factors = [2, 5, 11]
        # for r in runs:
        #     data = decimate(data, factors[r])
        # mean_band_envelope = np.mean(band_envelope,axis=1)
        # scipy decimate function
        # 22050 to 200
        # resample to 44.5k to 44k and then to 200
        # first factoreasy_money 11, then 5, last 4
        story.append(mean_zband_envelope)
    envelope = np.hstack(story)
    with open(envelope_file_decimated['black_willow'], 'wb') as file:
        joblib.dump(envelope, file)
