import os

from utils.constants import AUDIO_KEYS


def download_data_to_disk(bucket, cloud_loc, local_loc):
    download_args = [
        (bucket.blob(f"{cloud_loc}/{audio_key}.wav"), os.path.join(local_loc, f"{audio_key}.wav"))
        for audio_key in AUDIO_KEYS
    ] + [
        (bucket.blob(f"{cloud_loc}/metadata.json"), os.path.join(local_loc, "metadata.json"))
    ]
    for blob, local_path in download_args:
        blob.download_to_filename(local_path)
