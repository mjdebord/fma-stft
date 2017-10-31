# Processing
import scipy.signal as sig
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

# Free music archive utilities
import fma.utils # loads dotenv dotenv.load_dotenv(dotenv.find_dotenv())

# General
import os

if __name__ == "__main__":

    AUDIO_DIR = os.environ.get('AUDIO_DIR')
    METADATA_DIR = os.environ.get('METADATA_DIR')

    # print("Audio Dir: {}\nMetadata Dir: {}".format(AUDIO_DIR,METADATA_DIR))
    #
    # tracks = fma.utils.load(os.path.join(METADATA_DIR, 'tracks.csv'))
    # genres = fma.utils.load(os.path.join(METADATA_DIR, 'genres.csv'))

    # 'Small' Subset of tracks
    tracks_small = fma.utils.load(os.path.join(METADATA_DIR, 'tracks_small.csv'))
    # 'Instrumental' Subset
    instrumental = tracks_small['track','genre_top'] == 'Instrumental'
    # 'Electronic' Subset
    electronic = tracks_small['track', 'genre_top'] == 'Electronic'

    # make truncated metadata csvs
    tracks_instrumental = tracks_small[instrumental]
    tracks_electronic = tracks_small[electronic]
    # tracks_instrumental.to_csv('instrumental.csv')
    # tracks_electronic.to_csv('electronic.csv')

    instrumental_list = tracks_instrumental.index.values
    electronic_list = tracks_electronic.index.values

    #folder
    folder_num = instrumental_list[4] // 1000
    path = os.path.join(AUDIO_DIR,"{:03}/{:06}.mp3".format(folder_num,instrumental_list[4]))

    sound = AudioSegment.from_mp3(path)
    #sound.export("simple.mp3",format='mp3')
    raw_data = sound.raw_data

    sample_rate = sound.frame_rate
    sample_size = sound.sample_width
    channels = sound.channels

    sample_data = np.fromstring(raw_data, np.int16)
    sample_mp3 = np.array([sample_data[0::2].copy(), sample_data[1::2].copy()]).transpose()
    sample_mp3 = sample_mp3[:sample_rate*5, :]

    # Convert to mono channel
    sample_mono = (sample_mp3[:, 0] / 2 + sample_mp3[:, 1] / 2).astype(np.int16)

    # stft
    f, t, Zxx = sig.stft(sample_mono, sample_rate)  # using full sample rate

    # show spectrogram
    plt.figure()
    plt.pcolormesh(t, f, np.abs(Zxx))  # , vmin=0, vmax=mono.max() / (2 * np.sqrt(2)))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    # plt.yscale('log')
    plt.show()

    q = 3

    # #do the same thing but for mp3
    # sound = AudioSegment.from_mp3("Kalipluche-ascent.mp3")
    # #sound.export("simple.mp3",format='mp3')
    # raw_data = sound.raw_data
    #
    # sample_rate = sound.frame_rate
    # sample_size = sound.sample_width
    # channels = sound.channels
    #
    # sample_data = np.fromstring(raw_data, np.int16)
    # sample_mp3 = np.array([sample_data[0::2].copy(), sample_data[1::2].copy()]).transpose()
    # sample_mp3 = sample_mp3[:sample_rate*20, :]
    #
    # # Convert to mono channel
    # sample_mono = (sample_mp3[:, 0] / 2 + sample_mp3[:, 1] / 2).astype(np.int16)
    #
    # # stft
    # f, t, Zxx = sig.stft(sample_mono, sample_rate)  # using full sample rate
    #
    # #Random transform;
    # Zxx = np.conjugate(Zxx) #conjugate signal; sounds...evil
    # #Zxx = Zxx * np.exp(1j*np.pi) #inverted signal, sound same on its own, does cool things when played w/ original signal
    #
    # # inverse stft
    # _, x_rec = sig.istft(Zxx, sample_rate)
    #
    # # normalize amplitude range
    # mult = (sample_mono.max() - sample_mono.min()) / (x_rec.max() - x_rec.min())
    # x_out = (x_rec * mult).astype(np.int16)
    #
    # # write mp3 file
    # x_outb = x_out.tobytes()
    # out = AudioSegment(x_outb,frame_rate=sample_rate,sample_width=2,channels=1)
    # out.export("mp3Recon.mp3",format='mp3')
    #
    # # write funny mp3
    # funx = np.ravel(np.column_stack((xout,x_out)))
    # funout = AudioSegment(funx.tobytes(),frame_rate=sample_rate,sample_width=2,channels=2)
    # funout.export("inv.mp3",format='mp3')
    #
    # # show spectrogram
    # # plt.figure()
    # # plt.pcolormesh(t, f, np.abs(Zxx))  # , vmin=0, vmax=mono.max() / (2 * np.sqrt(2)))
    # # plt.title('STFT Magnitude')
    # # plt.ylabel('Frequency (Hz)')
    # # plt.xlabel('Time (sec)')
    # # # plt.yscale('log')
    # # plt.show()

    pass