# Continous Integration and Continous Deployment

## Breast Cancer Prediction Pipeline

### Integrated with Google Cloud Trigger

To deploy the pipeline, make another commit to the pipeline, and it will run Cloud Build which will compile the pipeline and deploy it to the endpoint.
Along with this, a final image of the pipeline called "PipelineOutputs/final.json" will be available in Google Cloud Storage Bucket associated with the project. This file can be used to deploy the pipeline again at any time.

### Description of Files

pipeline.py:
Contains the Actual Training pipeline in the form of a python code.

cloudbuild.yaml:
A file required by GCP to build use pipeline.py in order to create final.json and deploy to endpoint.

requirements.txt:
A file containing all the required libraries for the code.
