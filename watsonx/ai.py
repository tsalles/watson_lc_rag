import os

from ibm_watson_machine_learning.foundation_models import Model


def llm(model_id: str, params: dict) -> Model:
    credentials = {
        "url": os.environ.get("WATSON_MACHINE_LEARNING_URL", "https://us-south.ml.cloud.ibm.com"),
        "apikey": os.environ["WATSON_MACHINE_LEARNING_APIKEY"],
    }
    return Model(
        model_id=model_id,
        params=params,
        credentials=credentials,
        project_id=os.environ["WATSONX_PROJECT_ID"],
    )
