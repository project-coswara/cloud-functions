import librosa


class MFCC:
    def __init__(self, src_path):
        self.src_path = src_path
        self.y, self.sr = librosa.load(self.src_path, mono=True)
        self.features = None

    def extract(self, n_mfcc=25):
        mfcc = librosa.feature.mfcc(self.y, sr=self.sr, n_mfcc=n_mfcc)
        self.features = mfcc.mean(axis=1).T
