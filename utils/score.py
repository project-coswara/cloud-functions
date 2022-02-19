import json
import multiprocessing as mp
import numpy as np
import os
import pickle
import torch

from utils.common import parse_config
from utils.constants import AUDIO_KEYS
from utils.feature import FeatureExtractor


def score_symptoms(local_loc: str):
    with open(os.path.join("data/models/symptoms.pkl"), 'rb') as f:
        model = pickle.load(f)["classifier"]

    with open(os.path.join(local_loc, "metadata.json")) as f:
        metadata = json.load(f)

    with open(os.path.join("data/configs/symptom_keys")) as f:
        symptom_keys = [line.strip() for line in f.readlines()]

    symptoms_data = np.array([metadata[key] * 1 if key in metadata else 0 for key in symptom_keys], ndmin=2)
    return model.predict_proba(symptoms_data)[0][1]


def score_audio(local_loc: str, audio_key: str):
    training_config = parse_config("data/configs/train_config")

    # feature extraction and normalization
    feature_extractor = FeatureExtractor(config_file="data/configs/feature_config")
    features = feature_extractor.extract(os.path.join(local_loc, f"{audio_key}.wav"))
    if training_config['training_dataset'].get('apply_mean_norm', False):
        features = features - torch.mean(features, dim=0)
    if training_config['training_dataset'].get('apply_var_norm', False):
        features = features / torch.std(features, dim=0)
    input_features = features.to('cpu')

    seg_mode = training_config['training_dataset'].get('mode', 'file')
    if seg_mode == 'file':
        input_features = [input_features]
    elif seg_mode == 'segment':
        segment_length = int(training_config['training_dataset'].get('segment_length', 300))
        segment_hop = int(training_config['training_dataset'].get('segment_hop', 10))
        input_features = [
            input_features[i:i + segment_length, :]
            for i in range(0, max(1, features.shape[0] - segment_length), segment_hop)
        ]
    else:
        raise ValueError('Unknown Eval model')

    # evaluation
    model = torch.load(f"data/models/{audio_key}.mdl", map_location='cpu')
    model.eval()
    with torch.no_grad():
        output = model.predict_proba(input_features)
    return sum(output)[0].item() / len(output)


def score_fn(local_loc: str, key: str):
    if key == "metadata.json":
        return score_symptoms(local_loc=local_loc)
    else:
        return score_audio(local_loc=local_loc, audio_key=key)


def score_user(local_loc: str, num_threads: int = 1):
    score_args = [(local_loc, "metadata.json")] + [(local_loc, audio_key) for audio_key in AUDIO_KEYS]

    with mp.Pool(processes=num_threads) as pool:
        scores = pool.starmap(score_fn, score_args)

    return sum(scores) / len(scores)
