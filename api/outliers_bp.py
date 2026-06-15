from flask import Blueprint, jsonify, request

from src import EDAAnalyzer
from .pipeline_store import PIPELINES

outliers_bp = Blueprint("outliers", __name__)


def get_pipeline_or_error(pipeline_id):

    pipeline = PIPELINES.get(pipeline_id)

    if pipeline is None:
        return None, (
            jsonify({"error": "Pipeline not found"}),
            404
        )

    return pipeline, None


@outliers_bp.route(
    "/outliers/remove/<pipeline_id>",
    methods=["POST"]
)
def remove_outliers(pipeline_id):

    pipeline, error = get_pipeline_or_error(
        pipeline_id
    )

    if error:
        return error

    X = pipeline["X"]
    y = pipeline["y"]

    analyzer = EDAAnalyzer()

    threshold = request.json.get(
        "iqr_threshold",
        1.5
    ) if request.is_json else 1.5

    X_clean, outlier_report = (
        analyzer.remove_outliers_iqr(
            X,
            iqr_threshold=threshold
        )
    )

    y_clean = y.loc[X_clean.index]

    removed_samples = (
        X.shape[0] - X_clean.shape[0]
    )

    removed_pct = (
        removed_samples / X.shape[0] * 100
    )

    pipeline["X"] = X_clean
    pipeline["y"] = y_clean

    pipeline["results"]["outlier_removal"] = {
        "threshold": threshold,
        "removed_samples": removed_samples,
        "removed_pct": round(
            removed_pct,
            2
        ),
        "report": outlier_report
    }

    return jsonify({
        "message": "Outliers removed successfully",

        "threshold": threshold,

        "samples_before":
            X.shape[0],

        "samples_after":
            X_clean.shape[0],

        "removed_samples":
            removed_samples,

        "removed_pct":
            round(
                removed_pct,
                2
            ),

        "outlier_report":
            outlier_report
    })


@outliers_bp.route(
    "/outliers/distributions/<pipeline_id>",
    methods=["POST"]
)
def comparative_distributions(
    pipeline_id
):

    pipeline, error = get_pipeline_or_error(
        pipeline_id
    )

    if error:
        return error

    results = pipeline["results"].get(
        "outlier_removal"
    )

    if results is None:

        return jsonify({
            "error":
                "Run outlier removal first"
        }), 400

    X_clean = pipeline["X"]

    X_original = pipeline.get(
        "X_original"
    )

    if X_original is None:

        return jsonify({
            "error":
                "Original dataset not found"
        }), 400

    analyzer = EDAAnalyzer()

    fig = (
        analyzer
        .plot_comparative_distributions(
            X_original,
            X_clean,
            title_suffix=(
                "(IQR Outlier Removal)"
            ),
            show=False
        )
    )

    filepath = (
        f"outputs/"
        f"{pipeline_id}"
        f"_outlier_comparison.png"
    )

    fig.savefig(filepath)

    pipeline["results"][
        "comparative_distributions"
    ] = filepath

    return jsonify({
        "message":
            "Comparative plot generated",

        "file":
            filepath
    })


