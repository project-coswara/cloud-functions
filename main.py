import datetime
import os

from google.cloud import firestore, storage

from utils.score import score_user
from utils.storage import download_data_to_disk

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
    cloud_loc = f"COLLECT_DATA/{date_string}/{user_id}"
    local_loc = f"/tmp/{user_id}"
    os.makedirs(local_loc, exist_ok=True)

    # download audio to local directory
    firestore_db.collection("SCORE_DATA").document(user_id).update({
        "m": "Processing Audio",
        "sT": datetime.datetime.now(),
    })
    download_data_to_disk(bucket=storage_bucket, cloud_loc=cloud_loc, local_loc=local_loc)

    # get scores
    firestore_db.collection("SCORE_DATA").document(user_id).update({
        "m": "Computing Scores",
    })
    final_score = score_user(local_loc=local_loc, num_threads=5)

    # push score to database
    firestore_db.collection("SCORE_DATA").document(user_id).update({
        "m": "Finished",
        "score": final_score,
        "fT": datetime.datetime.now(),
    })
