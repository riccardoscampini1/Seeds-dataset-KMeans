from flask import Blueprint, jsonify

from src import FeatureScaler
from .pipeline_store import PIPELINES

scaling_bp = Blueprint(
    "scaling",
    __name__
)


def get_pipeline_or_error(
    pipeline_id
):

    pipeline = PIPELINES.get(
        pipeline_id
    )

    if pipeline is None:

        return None, (
            jsonify(
                {
                    "error":
                        "Pipeline not found"
                }
            ),
            404
        )

    return pipeline, None



@scaling_bp.route(
    "/scaling/<pipeline_id>",
    methods=["POST"]
)
def scale_data(
    pipeline_id
):

    pipeline, error = (
        get_pipeline_or_error(
            pipeline_id
        )
    )

    if error:
        return error

    X = pipeline["X"]

    scaler = FeatureScaler()

    X_scaled = (
        scaler.scale_robust(X)
    )

    scaler_info = (
        scaler.get_scaler_info()
    )

    pipeline["datasets"]["scaled"] = X_scaled

    pipeline["artifacts"]["scaler"] = scaler

    pipeline["results"][
        "scaling"
    ] = scaler_info

    return jsonify({

        "message":
            "Robust scaling applied",

        "scaler_type":
            scaler_info[
                "scaler_type"
            ],

        "n_columns_scaled":
            scaler_info[
                "n_columns_scaled"
            ],

        "shape":
            list(
                X_scaled.shape
            )
    })


@scaling_bp.route(
    "/scaling/report/<pipeline_id>",
    methods=["GET"]
)
def scaling_report(
    pipeline_id
):

    pipeline, error = (
        get_pipeline_or_error(
            pipeline_id
        )
    )

    if error:
        return error

    X = pipeline["datasets"]["scaled"]

    return jsonify({

        "shape":
            list(X.shape),

        "statistics":
            X.describe().to_dict()
    })