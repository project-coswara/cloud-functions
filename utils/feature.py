import numpy as np
import torch
import torchaudio

from utils.common import parse_config


def compute_sad(sig, fs, threshold=0.0001, sad_start_end_sil_length=100, sad_margin_length=50):
    """
        Compute threshold based sound activity
    """
    # Leading/Trailing margin
    sad_start_end_sil_length = int(sad_start_end_sil_length * 1e-3 * fs)
    # Margin around active samples
    sad_margin_length = int(sad_margin_length * 1e-3 * fs)

    sample_activity = np.zeros(sig.shape)
    sample_activity[np.power(sig, 2) > threshold] = 1
    sad = np.zeros(sig.shape)
    for i in range(sample_activity.shape[1]):
        if sample_activity[0, i] == 1:
            sad[0, i - sad_margin_length:i + sad_margin_length] = 1
    sad[0, 0:sad_start_end_sil_length] = 0
    sad[0, -sad_start_end_sil_length:] = 0
    return sad


class FeatureExtractor:
    def __init__(self, config_file):
        args = parse_config(config_file)["default"]
        self.feature_type = args["feature_type"]
        assert self.feature_type in ["MFCC", "MelSpec", "logMelSpec"], (
            "Expected the feature_type to be MFCC / MelSpec / logMelSpec"
        )

        self.sample_rate = int(args["resampling_rate"])
        self.mel_args = {
            "n_fft": int(float(args['window_size']) * 1e-3 * self.sample_rate),
            "n_mels": int(args['n_mels']),
            "f_max": int(args['f_max']),
            "hop_length": int(float(args['hop_length']) * 1e-3 * self.sample_rate)
        }
        self.n_mfcc = int(args["n_mfcc"]) if "n_mfcc" in args else 0
        self.compute_deltas = bool(args["compute_deltas"]) if "compute_deltas" in args else False
        self.compute_delta_deltas = bool(args["compute_delta_deltas"]) if "compute_delta_deltas" in args else False

    def extract(self, audio_loc):
        s = self._read_audio(audio_loc)
        features = self.get_extractor()(s)
        if self.feature_type == 'logMelSpec':
            features = torchaudio.functional.amplitude_to_DB(features, multiplier=10, amin=1e-10, db_multiplier=0)
        final_features = features
        if self.compute_deltas or self.compute_delta_deltas:
            delta_features = torchaudio.functional.compute_deltas(features)
            final_features = [features, delta_features]
            if self.compute_delta_deltas:
                double_delta_features = torchaudio.functional.compute_deltas(delta_features)
                final_features.append(double_delta_features)
            final_features = torch.cat(final_features, dim=0)
        return final_features.T

    def get_extractor(self):
        if self.feature_type == "MFCC":
            feature_args = {"sample_rate": self.sample_rate, "n_mfcc": self.n_mfcc, "melkwargs": self.mel_args}
            return torchaudio.transforms.MFCC(**feature_args)
        else:
            feature_args = {**{"sample_rate": self.sample_rate}, **self.mel_args}
            return torchaudio.transforms.MelSpectrogram(**feature_args)

    def _read_audio(self, filepath):
        """
        This code does the following:
            1. Read audio,
            2. Resample the audio if required,
            3. Perform waveform normalization,
            4. Compute sound activity using threshold based method
            5. Discard the silence regions
        """
        s, fs = torchaudio.load(filepath)
        if fs != self.sample_rate:
            s, fs = torchaudio.sox_effects.apply_effects_tensor(s, fs, [['rate', str(self.sample_rate)]])
        s = s / torch.max(torch.abs(s))
        sad = compute_sad(s.numpy(), self.sample_rate)
        s = s[np.where(sad == 1)]
        return s