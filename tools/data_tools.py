import tools.audio_utils as au

from pydub import AudioSegment
import numpy as np
import os
import copy

# converts mp3 to mono-channel 1D wave
#  Input:
#    mp3_path: path to mp3 file
#    t_start,t_stop: time endpoints in seconds to retrieve audio
#       if unspecified, whole audio will be used
#  Output:
#    audio_segment: full pydub mp3 AudioSegment
#    wave: mono-channel (1D) waveform extracted from mp3 with given time bounds
#  Notes:
#    No error checking; make sure path and time endpoints make sense
#
def mono_wave_from_mp3(mp3_path, sample_rate=None,t_start=None, t_stop=None):
    # load audio segment
    audio_segment = AudioSegment.from_mp3(mp3_path)
    # set frame rate
    if (sample_rate != None):
        if (audio_segment.frame_rate != sample_rate):
            audio_segment.set_frame_rate(sample_rate)

    # get mono wave
    wave = mono_wave(audio_segment, t_start, t_stop)
    return audio_segment, wave


# gets mono-channel waveform from pydub audio segment. Uses averaging as conversion function
#   Input:
#      audio_segment: pydub audio segment
#      t_start,t_stop: time endpoints to retrieve audio. No input given = use entire segment
#   Output:
#      wave: 1d mono-channel waveform
#   Note:
#      assumes audio_segment.raw_data is in 16 bit format
#      assumes audio_segment.raw_data is in interleaved format [chan1 chan2 chan1 chan2 ...etc]
#      this is at least true for audio segments loaded from mp3 (maybe in general?)
def mono_wave(audio_segment, t_start=None, t_stop=None):
    if (t_start == None):
        t_start = 0
    if (t_stop == None):
        t_stop = np.floor(audio_segment.duration_seconds).astype(int)

    # audio properties
    channels = audio_segment.channels  # num of channels
    sr = audio_segment.frame_rate  # sample rate (Hz)

    # convert raw audio data to numpy array
    raw_data = np.fromstring(audio_segment.raw_data, np.int16)

    # get sample from time bounds
    raw_sample = raw_data[t_start * channels * sr:t_stop * channels * sr]

    # convert to mono channel with averaging
    wave = np.zeros(raw_sample.size // channels, dtype=np.int16)
    for c in range(channels):
        wave += (raw_sample[c::channels] / channels).astype(np.int16)

    return wave


# gets fma paths given file index and unzipped fma_[subset].zip dir
#   Input:
#      index: numeric file index (number part of filename) of fma mp3
#      fma_dir: file path to unzipped fma_subset.zip dir
#   Output:
#      paths: list of file paths generated from indices
#   Note:
#      NO error checking, wont tell you if file doesn't exist
def fma_paths_from_indices(index, fma_dir):
    paths = []
    for i in range(index.size):
        # tracks are seperated into folders of 1000 songs ordered sequentially
        folder_num = index[i] // 1000
        # folder/file names are 3/6 digits respectively padded with 0
        paths.append(os.path.join(fma_dir, "{:03}/{:06}.mp3".format(folder_num,index[i])))
    return paths


# Creates spectrogram dataset
def spectrogram_dataset(mp3_paths,
                        nspecs_file,
                        spec_len,
                        sample_rate = 44100,
                        fft_size = 2048,
                        step_size = 128,
                        log_scale=True,
                        spec_threshold = 0,
                        quiet =True):

    ds = {'data': [],
          'mp3_paths': [],
          'sample_rate': sample_rate,
          'log_scale': log_scale,
          'spec_len': spec_len,
          'spec_per_file': nspecs_file,
          'spec_threshold': spec_threshold,
          'fft_size': fft_size,
          'step_size': step_size}

    assert isinstance(spec_len, int), "spec_len should be int"
    assert isinstance(sample_rate , int), "sample_rate should be int"
    assert isinstance(nspecs_file , int), "nspecs_file should be int"
    assert isinstance(fft_size , int), "fft_size should be int"
    assert isinstance(step_size , int), "step_size should be int"

    npaths = len(mp3_paths)
    for p in range(npaths):
        path = mp3_paths[p]
        if (not quiet):
            print("Processing {}/{}: {}".format(p+1,npaths,path))

        # Load song and get mono-channel wave
        audio_seg, full_wave = mono_wave_from_mp3(path, sample_rate=sample_rate)
        assert spec_len <= audio_seg.duration_seconds, "spectrogram length exceeds duration {}".format(path)
        # audio segment time-offset for successive spectrograms
        time_offset = spec_len
        if (spec_len * nspecs_file > audio_seg.duration_seconds):
            time_offset = np.floor((audio_seg.duration_seconds - spec_len) / (nspecs_file-1)).astype(int)
        sample_offset = time_offset * sample_rate
        sample_len = spec_len * sample_rate
        # create spectrograms from wave
        for s in range(nspecs_file):
            beg = (sample_offset * s).astype(int)
            end = (beg + sample_len).astype(int)
            #calculate and add spectrogram
            # note pretty spectrogram gives rows = time, we transpose it here
            ds['data'].append(au.pretty_spectrogram(full_wave[beg:end].astype('float64'), fft_size=fft_size,
                           step_size=step_size, log=log_scale, thresh=spec_threshold).transpose().astype(np.float32))
            ds['mp3_paths'].append(path)  # path spec is generated from

        pass
    return ds

# converts spectrogram dataset to mel-spectrogram dataset
def melspec_dataset(spec_ds, freq_components = 128, shorten = 1, low_freq = 2, high_freq =22000, quiet=True):

    mel_ds = {'data': [],
              'mp3_paths': spec_ds['mp3_paths'].copy(),
              'sample_rate': spec_ds['sample_rate'],
              'log_scale': spec_ds['log_scale'],
              'spec_len': spec_ds['spec_len'],
              'spec_per_file': spec_ds['spec_per_file'],
              'spec_threshold': spec_ds['spec_threshold'],
              'fft_size': spec_ds['fft_size'],
              'step_size': spec_ds['step_size'],
              'freq_components' : freq_components,
              'shorten_factor' : shorten,
              'low_freq': low_freq,
              'high_freq': high_freq,
              'mel_filter' : [],
              'inv_mel_filter' : []}

    # create_mel_filter gives both the forward and inverse mel filters
    mel_filter, mel_inversion_filter = au.create_mel_filter(fft_size=mel_ds['fft_size'],
                                                            n_freq_components=freq_components,
                                                            start_freq=low_freq,
                                                            end_freq=high_freq,
                                                            samplerate=mel_ds['sample_rate'])
    mel_ds['mel_filter'] = mel_filter
    mel_ds['inv_mel_filter'] = mel_inversion_filter
    nspecs_file = mel_ds['spec_per_file']

    nspecs = len(spec_ds['data'])
    for s in range(nspecs):
        if ((not quiet) and (np.mod(s,nspecs_file) == 0)):
            print("Processing {}/{}: {}".format(s//nspecs_file +1,nspecs//nspecs_file,mel_ds['mp3_paths'][s]))
        #create mel-spectrogram
        #  note that spectrogram_dataset transposed so that rows = frequencies, we need to transpose again here
        #  the output of make_mel will have rows=frequencies
        debug = au.make_mel(spec_ds['data'][s].transpose(), mel_filter, shorten_factor=shorten)
        mel_ds['data'].append(au.make_mel(spec_ds['data'][s].transpose(), mel_filter, shorten_factor=shorten))

    return mel_ds