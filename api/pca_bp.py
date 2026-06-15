from flask import Blueprint, jsonify

from src import PCAProcessor
from .pipeline_store import PIPELINES

pca_bp = Blueprint(
    "pca",
    __name__
)


@pca_bp.route(
    "/pca/<pipeline_id>",
    methods=["POST"]
)
def apply_pca(
    pipeline_id
):

    pipeline = PIPELINES.get(
        pipeline_id
    )

    if pipeline is None:

        return jsonify({
            "error":
                "Pipeline not found"
        }), 404

    X_scaled = (
        pipeline["datasets"]
        ["scaled"]
    )

    pca = PCAProcessor()

    pca.configure(
        n_components=0.95,
        random_state=42
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    pipeline["datasets"][
        "pca"
    ] = X_pca

    pipeline["artifacts"][
        "pca"
    ] = pca

    pipeline["results"][
        "pca"
    ] = {

        "n_components":
            pca.n_components_,

        "input_features":
            X_scaled.shape[1],

        "output_features":
            X_pca.shape[1]
    }

    return jsonify({

        "message":
            "PCA completed",

        "input_features":
            X_scaled.shape[1],

        "output_features":
            X_pca.shape[1]
    })


@pca_bp.route(
    "/pca/report/<pipeline_id>",
    methods=["GET"]
)
def pca_report(
    pipeline_id
):

    pipeline = PIPELINES.get(
        pipeline_id
    )

    if pipeline is None:

        return jsonify({
            "error":
                "Pipeline not found"
        }), 404

    return jsonify(
        pipeline["results"]["pca"]
    )