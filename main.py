import configparser
import json
import os
import multiprocessing as mp
from utils.utils import *

from google.cloud import firestore, storage

firestore_db = firestore.Client()
storage_client = storage.Client()
storage_bucket = storage_client.bucket("project-coswara.appspot.com")


# %%
# Cloud function triggers
def hello_world(request):
    return "Hello world! Welcome to Coswara cloud functions..."


# %%
def infer_audio(FE, model_path, audio_path, config):
    # Load model, use CPU for inference
    model = torch.load(model_path, map_location='cpu')
    model.eval()

    # Prepare features
    F = FE.extract(audio_path)
    if config['training_dataset'].get('apply_mean_norm', False): F = F - torch.mean(F, dim=0)
    if config['training_dataset'].get('apply_var_norm', False): F = F / torch.std(F, dim=0)
    feat = F.to('cpu')

    # Input mode
    seg_mode = config['training_dataset'].get('mode', 'file')
    if seg_mode == 'file':
        feat = [feat]
    elif seg_mode == 'segment':
        segment_length = int(config['training_dataset'].get('segment_length', 300))
        segment_hop = int(config['training_dataset'].get('segment_hop', 10))
        feat = [feat[i:i + segment_length, :] for i in range(0, max(1, F.shape[0] - segment_length), segment_hop)]
    else:
        raise ValueError('Unknown eval model')

    with torch.no_grad():
        output = model.predict_proba(feat)

    # Average the scores of all segments from the input file
    score = sum(output)[0].item() / len(output)
    return score


# %%
def infer_symptoms(symptoms_keys, model_path, data_dict):
    with open(model_path, 'rb') as pickle_file:
        model_info = pickle.load(pickle_file)
    # model_info = pickle.load(model_path,'rb')
    classifier = model_info['classifier']

    f = [data_dict[key] * 1 for key in symptoms_keys]
    sc = classifier.predict_proba(np.array(f, ndmin=2))
    return sc[0][1]


# %%
def do_one_infer(audio_loc, audio_key, local_audio_loc, FE, config):
    blob = storage_bucket.blob(f"{audio_loc}/{audio_key}.wav")
    local_audio_path = os.path.join(local_audio_loc, f"{audio_key}.wav")
    blob.download_to_filename(local_audio_path)

    model_path = f"models/{audio_key}/models/final.mdl"
    return infer_audio(FE=FE,
                       model_path=model_path,
                       audio_path=local_audio_path,
                       config=config)


# %%
def compute_score(data, context):
    """Triggered by a change to a Firestore document to compute score.
    Args:
        data (dict): The event payload.
        context (google.cloud.functions.Context): Metadata for the event.
    """

    print(f"Function triggered by change to: {context.resource}")
    user_id = context.resource.rsplit("/")[-1]
    date_string = data["value"]["fields"]["dS"]["stringValue"]
    audio_loc = f"COLLECT_DATA/{date_string}/{user_id}"
    local_audio_loc = f"/tmp/{user_id}"
    os.makedirs(local_audio_loc, exist_ok=True)

    firestore_db.collection("SCORE_DATA").document(user_id).update({
        "m": "Processing Audio"
    })

    # For acoustic categories
    config = configparser.ConfigParser()
    config.read("models/train_config")

    # Feature extractor
    feature_config = configparser.ConfigParser()
    feature_config.read("models/feature_config")
    FE = feature_extractor(feature_config['default'])

    audio_key_list = [
        "breathing-shallow",
        "breathing-deep",
        "cough-shallow",
        "cough-heavy",
        "vowel-a",
        "vowel-e",
        "vowel-o",
        "counting-normal",
        "counting-fast"]

    # Compute scores for audio categories
    p = 3  # Number of parallel processes
    num_audio = 9
    with mp.Pool(processes=p) as pool:
        category_scores = pool.starmap(do_one_infer,
                                       zip([audio_loc] * num_audio, audio_key_list, [local_audio_loc] * num_audio,
                                           [FE] * num_audio, [config] * num_audio))

    # Compute score for Symptoms category
    model_path = 'models/symptoms/model.pkl'
    symptom_keys = [line.strip() for line in open('models/symptom_keys').readlines()]
    ####  Populate the data_dict variable here. 
    # Read the json metadata file and convert the values as True/False.
    # If a field is empty, then populate it as False

    # File - web-app/functions/src/index.ts:81
    blob = storage_bucket.blob(f"{audio_loc}/metadata.json")
    local_metadata_path = os.path.join(local_audio_loc, f"metadata.json")
    blob.download_to_filename(local_metadata_path)

    f = open(local_metadata_path)
    metadata = json.load(f)
    f.close()

    data_dict = {}
    for key in symptom_keys:
        if key in metadata:
            data_dict[key] = metadata[key]
        else:
            data_dict[key] = False

    score = infer_symptoms(symptoms_keys=symptom_keys,
                           model_path=model_path,
                           data_dict=data_dict)
    category_scores.append(score)

    # Final score
    final_score = sum(category_scores) / len(category_scores)
    firestore_db.collection("SCORE_DATA").document(user_id).update({
        "m": "Finished",
        "score": final_score
    })
