import os
import joblib
import numpy as np
import pandas as pd
from io import StringIO


def model_fn(model_dir):
    """Load model from the model_dir — called once when endpoint starts."""
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    return {"model": model, "label_encoder": label_encoder}


def input_fn(request_body, request_content_type):
    """Deserialize input data — called on every prediction request."""
    if request_content_type == "application/json":
        import json
        data = json.loads(request_body)
        return pd.DataFrame([data])
    elif request_content_type == "text/csv":
        return pd.read_csv(StringIO(request_body), header=None)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_artifacts):
    """Run prediction — called after input_fn."""
    model = model_artifacts["model"]
    label_encoder = model_artifacts["label_encoder"]

    # Encode product_category if present
    if "product_category" in input_data.columns:
        input_data["product_category"] = label_encoder.transform(
            input_data["product_category"]
        )

    prediction = model.predict(input_data)
    return prediction


def output_fn(prediction, response_content_type):
    """Serialize output — called after predict_fn."""
    if response_content_type == "application/json":
        import json
        return json.dumps({"prediction": prediction.tolist()})
    elif response_content_type == "text/csv":
        return str(prediction[0])
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
