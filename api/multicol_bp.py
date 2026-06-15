from flask import Blueprint, jsonify, request

from src import MulticollinearityReducer
from .pipeline_store import PIPELINES

multicol_bp = Blueprint(
    "multicollinearity",
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
            jsonify({
                "error":
                    "Pipeline not found"
            }),
            404
        )

    return pipeline, None


@multicol_bp.route(
    "/multicollinearity/<pipeline_id>",
    methods=["POST"]
)
def reduce_multicollinearity(
    pipeline_id
):

    pipeline, error = (
        get_pipeline_or_error(
            pipeline_id
        )
    )

    if error:
        return error

    X_scaled = (
        pipeline["datasets"]
        ["scaled"]
    )

    threshold = 0.85

    if request.is_json:

        threshold = (
            request.json.get(
                "threshold",
                0.85
            )
        )

    reducer = (
        MulticollinearityReducer(
            threshold=threshold
        )
    )

    X_multicollinearity = (
        reducer.fit_transform(
            X_scaled
        )
    )

    report = reducer.report()

    pipeline["datasets"][
        "multicollinearity"
    ] = X_multicollinearity

    pipeline["artifacts"][
        "multicollinearity_reducer"
    ] = reducer

    pipeline["results"][
        "multicollinearity"
    ] = report

    return jsonify({

        "message":
            "Multicollinearity reduction completed",

        "threshold":
            threshold,

        "original_shape":
            list(
                X_scaled.shape
            ),

        "reduced_shape":
            list(
                X_multicollinearity.shape
            ),

        "columns_removed":
            len(
                report[
                    "cols_to_drop"
                ]
            )
    })


@multicol_bp.route(
    "/multicollinearity/report/<pipeline_id>",
    methods=["GET"]
)
def report(
    pipeline_id
):

    pipeline, error = (
        get_pipeline_or_error(
            pipeline_id
        )
    )

    if error:
        return error

    report = (
        pipeline["results"]
        .get(
            "multicollinearity"
        )
    )

    if report is None:

        return jsonify({
            "error":
                "Run multicollinearity reduction first"
        }), 400

    return jsonify(report)


@multicol_bp.route(
    "/multicollinearity/pairs/<pipeline_id>",
    methods=["GET"]
)
def correlated_pairs(
    pipeline_id
):

    pipeline, error = (
        get_pipeline_or_error(
            pipeline_id
        )
    )

    if error:
        return error

    report = (
        pipeline["results"]
        .get(
            "multicollinearity"
        )
    )

    if report is None:

        return jsonify({
            "error":
                "Run multicollinearity reduction first"
        }), 400

    return jsonify({

        "n_pairs":
            len(
                report["pairs"]
            ),

        "pairs":
            report["pairs"]
    })


@multicol_bp.route(
    "/multicollinearity/summary/<pipeline_id>",
    methods=["GET"]
)
def summary(
    pipeline_id
):

    pipeline, error = (
        get_pipeline_or_error(
            pipeline_id
        )
    )

    if error:
        return error

    report = (
        pipeline["results"]
        .get(
            "multicollinearity"
        )
    )

    X_multi = (
        pipeline["datasets"]
        .get(
            "multicollinearity"
        )
    )

    if X_multi is None:

        return jsonify({
            "error":
                "Run multicollinearity reduction first"
        }), 400

    return jsonify({

        "shape":
            list(
                X_multi.shape
            ),

        "cols_to_keep":
            report[
                "cols_to_keep"
            ],

        "cols_to_drop":
            report[
                "cols_to_drop"
            ]
    })