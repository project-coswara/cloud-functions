import os
import pickle

from google.cloud import firestore, storage

from utils.feature import MFCC


firestore_db = firestore.Client()
storage_client = storage.Client()
storage_bucket = storage_client.bucket("project-coswara.appspot.com")


# Cloud function triggers
def hello_world(request):
    return "Hello world! Welcome to Coswara cloud functions..."


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

    all_features = []
    for audio_key in [
        "breathing-shallow",
        "breathing-deep",
        "cough-shallow",
        "cough-heavy",
        "vowel-a",
        "vowel-e",
        "vowel-o",
        "counting-normal",
        "counting-fast"
    ][:1]:
        blob = storage_bucket.blob(f"{audio_loc}/{audio_key}.wav")
        local_audio_path = os.path.join(local_audio_loc, f"{audio_key}.wav")
        blob.download_to_filename(local_audio_path)
        mfcc = MFCC(local_audio_path)
        mfcc.extract()
        all_features.append(mfcc)

    firestore_db.collection("SCORE_DATA").document(user_id).update({
        "m": "Evaluating"
    })

    with open("./model.pkl", "rb") as mp:
        model = pickle.load(mp)

    model_input = all_features[0].features.reshape([1, -1])
    final_score = model.predict_proba(model_input)[0][0]

    firestore_db.collection("SCORE_DATA").document(user_id).update({
        "m": "Finished",
        "score": final_score
    })
