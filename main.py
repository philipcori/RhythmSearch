import librosa
import os

def main():
    data_dir = os.path.join(os.getcwd(), 'data\\')
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    y, sr = librosa.load(data_files[0])

    # y, sr = librosa.load(librosa.util.example_audio_file())
    notes = librosa.onset.onset_detect(y=y, sr=sr)
    notes = librosa.frames_to_time(notes, sr=sr)
    oenv = librosa.onset.onset_strength(y, sr=sr)
    # tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    # ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    # ac_global = librosa.util.normalize(ac_global)
    # # Estimate the global tempo for display purposes
    # tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]
    x = range(notes.shape[0])
    y = [x if (x > 10) else 0 for x in notes]

    tempogram = librosa.feature.tempogram(y, sr)

    print(tempogram.shape)



if __name__ == '__main__':
    main()
