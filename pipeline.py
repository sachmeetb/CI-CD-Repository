from datetime import datetime
from typing import NamedTuple

import google.cloud.aiplatform as aip

from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import Dataset, Input, Model, Output, component
from kfp.v2.dsl import ClassificationMetrics

# Project ID
# Val = !gcloud config list --format 'value(core.project)'
# print(Val)
PROJECT_ID = "optical-hexagon-350508"
REGION = "us-west1"
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
SERVICE_ACCOUNT = "559647087083-compute@developer.gserviceaccount.com"
BUCKET_NAME = "gs://demo-account"
PIPELINE_ROOT = "{}/pipeline_root/".format(BUCKET_NAME)
PIPELINE_JSON_FILE = "final.json"
PIPELINE_EXPERIMENT_NAME = "mainscoringpipeline" + TIMESTAMP
MODEL_DISPLAY_NAME = "main-model"

aip.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)


@component(
    base_image="gcr.io/ml-pipeline/google-cloud-pipeline-components:latest",
    packages_to_install=["scikit-learn", "pandas", "numpy"],
)
def df_load(output: Output[Dataset]):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer

    dataframe = load_breast_cancer()
    df = pd.DataFrame(
        data=np.c_[dataframe["data"], dataframe["target"]],
        columns=np.append(dataframe["feature_names"], "target"),
    )
    df.to_csv(output.path, index=False)


# Transformation


@component(
    base_image="gcr.io/ml-pipeline/google-cloud-pipeline-components:latest",
    packages_to_install=["scikit-learn", "pandas", "numpy"],)
def transform_df(dataset: Input[Dataset], output_scaled: Output[Dataset]):

    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(dataset.path)
    X = df.loc[:, df.columns != "target"]
    scaler = StandardScaler().fit(X)
    df_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

    df_scaled["target"] = df["target"]
    # TODO: Return DF for next component or save it to file and return URL
    df_scaled.to_csv(output_scaled.path, index=False)


# Train Test Split


@component(
    base_image="gcr.io/ml-pipeline/google-cloud-pipeline-components:latest",
    packages_to_install=["scikit-learn", "pandas", "numpy"],
)
def train_test_split(
    dataset: Input[Dataset],
    output_X_train: Output[Dataset],
    output_X_test: Output[Dataset],
    output_y_train: Output[Dataset],
    output_y_test: Output[Dataset],
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df_scaled = pd.read_csv(dataset.path)
    X_train, X_test, y_train, y_test = train_test_split(
        df_scaled.loc[:, df_scaled.columns != "target"],
        df_scaled["target"],
        test_size=0.2,
        random_state=1,
    )
    X_train.to_csv(output_X_train.path, index=False)
    X_test.to_csv(output_X_test.path, index=False)
    y_train.to_csv(output_y_train.path, index=False)
    y_test.to_csv(output_y_test.path, index=False)


# Algorithm 1 : K Nearest Neighbours


@component(
    base_image="gcr.io/ml-pipeline/google-cloud-pipeline-components:latest",
    packages_to_install=["scikit-learn", "pandas", "numpy"],
)
def KNN(
    input_X: Input[Dataset],
    input_y: Input[Dataset],
    output: Output[Model],
    output_metrics: Output[ClassificationMetrics],
):
    import os
    import pickle

    import pandas as pd
    from sklearn import model_selection
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier

    X_train = pd.read_csv(input_X.path)
    y_train = pd.read_csv(input_y.path)
    model_1 = KNeighborsClassifier(n_neighbors=35)
    model_1.fit(X_train, y_train)
    predictions = model_selection.cross_val_predict(model_1, X_train, y_train, cv=3)
    output_metrics.log_confusion_matrix(
        ["0", "1"],
        confusion_matrix(
            y_train, predictions
        ).tolist(),  # .tolist() to convert np array to list.
    )
    artifact_filename = "model_1.pkl"
    # Save model artifact to local filesystem (doesn't persist)
    local_path = artifact_filename
    with open(output.path, "wb") as model_file:
        pickle.dump(model_1, model_file)


# Algorithm 2 : Logistic Regression


@component(
    base_image="gcr.io/ml-pipeline/google-cloud-pipeline-components:latest",
    packages_to_install=["scikit-learn", "pandas", "numpy"],
)
def log_reg(
    input_X: Input[Dataset],
    input_y: Input[Dataset],
    output: Output[Model],
    output_metrics: Output[ClassificationMetrics],
):
    import os
    import pickle

    import pandas as pd
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, roc_curve
    from sklearn.model_selection import cross_val_predict

    X_train = pd.read_csv(input_X.path)
    y_train = pd.read_csv(input_y.path)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    model_2 = log_reg
    predictions = model_selection.cross_val_predict(model_2, X_train, y_train, cv=3)
    output_metrics.log_confusion_matrix(
        ["0", "1"],
        confusion_matrix(
            y_train, predictions
        ).tolist(),  # .tolist() to convert np array to list.
    )
    artifact_filename = "model_2.pkl"
    # Save model artifact to local filesystem (doesn't persist)
    local_path = artifact_filename
    with open(output.path, "wb") as model_file:
        pickle.dump(model_2, model_file)


# Algorithm 3 : Random Forest


@component(
    base_image="gcr.io/ml-pipeline/google-cloud-pipeline-components:latest",
    packages_to_install=["scikit-learn", "pandas", "numpy"],
)
def forest(
    input_X: Input[Dataset],
    input_y: Input[Dataset],
    output: Output[Model],
    output_metrics1: Output[ClassificationMetrics],
    output_metrics2: Output[ClassificationMetrics],
):
    import os
    import pickle

    import pandas as pd
    from sklearn import model_selection
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, roc_curve
    from sklearn.model_selection import cross_val_predict

    X_train = pd.read_csv(input_X.path)
    y_train = pd.read_csv(input_y.path)
    forest = RandomForestClassifier(n_estimators=50, random_state=0)
    forest.fit(X_train, y_train)
    model_3 = forest
    predictions = model_selection.cross_val_predict(model_3, X_train, y_train, cv=3)
    output_metrics1.log_confusion_matrix(
        ["0", "1"],
        confusion_matrix(
            y_train, predictions
        ).tolist(),  # .tolist() to convert np array to list.
    )
    # y_scores = cross_val_predict(model_3, X_train, y_train, cv=3, method="predict_proba")
    # fpr, tpr, thresholds = roc_curve(
    #     y_true=y_train, y_score=y_scores[:,1], pos_label=True
    # )
    # output_metrics2.log_roc_curve(fpr, tpr)
    artifact_filename = "model_3.pkl"
    # Save model artifact to local filesystem (doesn't persist)
    local_path = artifact_filename
    with open(output.path, "wb") as model_file:
        pickle.dump(model_3, model_file)


# Algorithm 4 : Support Vector Machine


@component(
    base_image="gcr.io/ml-pipeline/google-cloud-pipeline-components:latest",
    packages_to_install=["scikit-learn", "pandas", "numpy"],
)
def svm(
    input_X: Input[Dataset],
    input_y: Input[Dataset],
    output: Output[Model],
    output_metrics: Output[ClassificationMetrics],
):
    import os
    import pickle

    import pandas as pd
    from sklearn import model_selection
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_predict
    from sklearn.svm import SVC

    X_train = pd.read_csv(input_X.path)
    y_train = pd.read_csv(input_y.path)
    svm = SVC()
    svm.fit(X_train, y_train)
    model_4 = svm
    predictions = model_selection.cross_val_predict(model_4, X_train, y_train, cv=3)
    output_metrics.log_confusion_matrix(
        ["0", "1"],
        confusion_matrix(
            y_train, predictions
        ).tolist(),  # .tolist() to convert np array to list.
    )
    artifact_filename = "model_4.pkl"
    # Save model artifact to local filesystem (doesn't persist)
    local_path = artifact_filename
    with open(output.path, "wb") as model_file:
        pickle.dump(model_4, model_file)


# Crowning of Champion and Challenger


@component(
    base_image="gcr.io/ml-pipeline/google-cloud-pipeline-components:latest",
    packages_to_install=["scikit-learn", "pandas", "numpy"],
)
def crowning(
    input_model_1: Input[Model],
    input_model_2: Input[Model],
    input_model_3: Input[Model],
    input_model_4: Input[Model],
    input_X_test: Input[Dataset],
    input_y_test: Input[Dataset],
    output_champion: Output[Model],
    output_challenger: Output[Model],
) -> NamedTuple(
    "Outputs",
    [
        ("output_champion_location", str),
        ("output_challenger_location", str),
    ],
):
    import os
    import pickle

    import pandas as pd
    from google.cloud import storage

    model_1 = pickle.load(open(input_model_1.path, "rb"))
    model_2 = pickle.load(open(input_model_2.path, "rb"))
    model_3 = pickle.load(open(input_model_3.path, "rb"))
    model_4 = pickle.load(open(input_model_4.path, "rb"))
    models = [model_1, model_2, model_3, model_4]
    scores = []
    X_test = pd.read_csv(input_X_test.path)
    y_test = pd.read_csv(input_y_test.path)
    for i in range(1, 5):
        scores.append(models[i - 1].score(X_test, y_test))

    max_score = max(scores)
    print("champion = model_", scores.index(max_score))
    champion = models[scores.index(max_score)]
    scores.remove(max_score)
    models.remove(champion)

    # Crowning the Challenger
    second_max_score = max(scores)
    print("challenger = model_", scores.index(max_score))
    challenger = models[scores.index(max_score)]
    # TODO: Expose Champion and Challenger as APIs
    artifact_filename = "model.pkl"
    # Save model artifact to local filesystem (doesn't persist)
    local_path = artifact_filename

    def save_model_to_bucket(artifact_filename, Output_Model, Model):
        local_path = artifact_filename
        pickle.dump(Model, open(artifact_filename, "wb"))
        # Upload model artifact to Cloud Storage
        model_directory = Output_Model.path
        model_directory_gs = model_directory.replace("/gcs/", "gs://")
        storage_path = os.path.join(model_directory_gs, artifact_filename)
        blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
        blob.upload_from_filename(local_path)

    save_model_to_bucket(artifact_filename, output_champion, champion)
    save_model_to_bucket(artifact_filename, output_challenger, challenger)

    return (
        output_champion.path.replace("/gcs/", "gs://"),
        output_challenger.path.replace("/gcs/", "gs://"),
    )


@component(
    base_image="gcr.io/ml-pipeline/google-cloud-pipeline-components:latest",
    packages_to_install=["scikit-learn", "pandas", "numpy"],
)
def absolute_testing(
    input_champion: Input[Model],
    input_challenger: Input[Model],
    scaled_data: Input[Dataset],
) -> str:
    import os
    import pickle

    import pandas as pd

    champion_path = os.path.join(input_champion.path, "model.pkl")
    challenger_path = os.path.join(input_challenger.path, "model.pkl")
    champion_model = pickle.load(open(champion_path, "rb"))
    challenger_model = pickle.load(open(challenger_path, "rb"))
    df = pd.read_csv(scaled_data.path)
    sampled_df = df.sample(n=60)
    X = sampled_df.loc[:, sampled_df.columns != "target"]
    y = sampled_df.target
    champion_score = champion_model.score(X, y)
    challenger_score = challenger_model.score(X, y)
    if champion_score >= challenger_score:
        print("Champion prevails", champion_score, challenger_score)
        return "champion"
    else:
        print("Challenger prevails", challenger_score, champion_score)
        return "challenger"


@component(
    base_image="gcr.io/ml-pipeline/google-cloud-pipeline-components:latest",
    packages_to_install=["google-cloud-aiplatform"],
)
def import_model(
    project_id: str,
    region: str,
    display_name: str,
    artifact_gcs_bucket: str,
    model: Output[Model],
    location: str,
    serving_container_image_uri: str,
    description: str,
) -> NamedTuple("Outputs", [("display_name", str), ("resource_name", str)]):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)
    model_resp = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_gcs_bucket,
        serving_container_image_uri=serving_container_image_uri,
        description=description,
    )
    model_resp.wait()
    with open(model.path, "w") as f:
        f.write(model_resp.resource_name)
    model.path = f"aiplatform://v1/{model_resp.resource_name}"  # update the resource path to aiplaform://v1 prefix so that off the shelf tasks can consume the output
    DISPLAY_NAME = "breast-cancer"
    MODEL_NAME = "breast-cancer"
    ENDPOINT_NAME = "breast-cancer"

    model_deploy = model_resp.deploy(
        machine_type="n1-standard-4",
        traffic_split={"0": 100},
        deployed_model_display_name=DISPLAY_NAME,
    )

    return (
        model_resp.display_name,
        model_resp.resource_name,
    )


@component
def print_op(msg: str):
    print(msg)


@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    name=PIPELINE_EXPERIMENT_NAME,
)
def pipeline(
    project: str = PROJECT_ID,
    region: str = REGION,
):
    df = df_load()
    df_scaled = transform_df(df.outputs["output"])
    train_test = train_test_split(df_scaled.outputs["output_scaled"])
    model_1 = KNN(
        train_test.outputs["output_X_train"], train_test.outputs["output_y_train"]
    )
    model_2 = log_reg(
        train_test.outputs["output_X_train"], train_test.outputs["output_y_train"]
    )
    model_3 = forest(
        train_test.outputs["output_X_train"], train_test.outputs["output_y_train"]
    )
    model_4 = svm(
        train_test.outputs["output_X_train"], train_test.outputs["output_y_train"]
    )
    final_models = crowning(
        model_1.outputs["output"],
        model_2.outputs["output"],
        model_3.outputs["output"],
        model_4.outputs["output"],
        train_test.outputs["output_X_test"],
        train_test.outputs["output_y_test"],
    )
    absolute_result = absolute_testing(
        final_models.outputs["output_champion"],
        final_models.outputs["output_challenger"],
        df_scaled.outputs["output_scaled"],
    )
    for i in final_models.outputs:
        print(i)
    with dsl.Condition(absolute_result.output == "champion"):
        model_upload_op_1 = import_model(
            region=region,
            project_id=project,
            display_name="uploaded_model_champion",
            serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
            artifact_gcs_bucket=final_models.outputs["output_champion_location"],
            location=region,
            description="final model",
        )
    with dsl.Condition(absolute_result.output == "challenger"):
        model_upload_op_1 = import_model(
            project_id=project,
            region=region,
            display_name="uploaded_model_challenger",
            serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
            artifact_gcs_bucket=final_models.outputs["output_challenger_location"],
            location=region,
            description="final model",
        )


from kfp.v2 import compiler  # noqa: F811

compiler.Compiler().compile(pipeline_func=pipeline, package_path=PIPELINE_JSON_FILE)


job = aip.PipelineJob(
    display_name=PIPELINE_EXPERIMENT_NAME,
    template_path=PIPELINE_JSON_FILE,
    pipeline_root=PIPELINE_ROOT,
    parameter_values={"project": PROJECT_ID, "region": REGION},
)

job.run()
