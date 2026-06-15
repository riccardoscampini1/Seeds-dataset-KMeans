from flask import Blueprint, request, jsonify
import uuid

from src import DataLoader
from .pipeline_store import PIPELINES

loader_bp = Blueprint("loader", __name__)


@loader_bp.route("/load", methods=["POST"])
def load_dataset():
    """
    Carica un dataset e crea una nuova pipeline.

    Body JSON:
    {
        "file_path": "data/seeds_dataset.txt"
    }
    """

    try:

        data = request.get_json()

        if not data:
            return jsonify(
                {"error": "Missing JSON body"}
            ), 400

        file_path = data.get("file_path")

        if not file_path:
            return jsonify(
                {"error": "file_path is required"}
            ), 400

        loader = DataLoader()

        X, y = loader.load_seeds_dataset(file_path)

        pipeline_id = str(uuid.uuid4())

        PIPELINES[pipeline_id] = {
            "X": X,
            "y": y,

            "X_original": X.copy(),
            "y_original": y.copy(),
            
            "X_train": None,
            "X_test": None,
            "y_train": None,
            "y_test": None,
            "model": None,
            "results": {}
        }

        return jsonify(
            {
                "message": "Dataset loaded successfully",
                "pipeline_id": pipeline_id,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "features": list(X.columns),
                "target": y.columns[0]
            }
        )

    except Exception as e:

        return jsonify(
            {"error": str(e)}
        ), 500


@loader_bp.route("/report/<pipeline_id>", methods=["GET"])
def report(pipeline_id):

    pipeline = PIPELINES.get(pipeline_id)

    if pipeline is None:

        return jsonify(
            {"error": "Pipeline not found"}
        ), 404

    X = pipeline["X"]
    y = pipeline["y"]

    return jsonify(
        {
            "pipeline_id": pipeline_id,
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "features": list(X.columns),
            "target": y.columns[0],
            "sample": X.head().to_dict(
                orient="records"
            )
        }
    )