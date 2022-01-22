# cloud-functions
Contains code for computation of scores.


### Deployment

#### hello_world http endpoint

```
gcloud functions deploy hello_world --project project-coswara --runtime python37 --trigger-http --allow-unauthenticated
```

#### compute_score firestore endpoint

```
export PROJECT_ID="project-coswara" && gcloud functions deploy compute_score \
 --project ${PROJECT_ID} \
 --region europe-west1 \
 --runtime python37 \
 --memory 2048MB \
 --trigger-event "providers/cloud.firestore/eventTypes/document.create" \
 --trigger-resource "projects/${PROJECT_ID}/databases/(default)/documents/SCORE_DATA/{sessionId}"
```