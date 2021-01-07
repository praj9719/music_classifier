import librosa
import numpy as np
import pickle

tests = [
    'data/blues/blues.00006.wav',
    'data/disco/disco.00001.wav',
    'data/classical/classical.00004.wav',
    'data/rock/rock.00005.wav',
    'data/pop/pop.00007.wav'
]

res = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock"
}

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

for test in tests:
    y, sr = librosa.load(test, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    parameters = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        parameters += f' {np.mean(e)}'
    parameters = parameters.split()

    print(f"Path: {test} Prediction: {res[model.predict([parameters])[0]]}")

