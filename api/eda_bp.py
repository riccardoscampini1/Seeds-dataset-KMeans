from flask import Blueprint, jsonify

from src import EDAAnalyzer
from .pipeline_store import PIPELINES

eda_bp = Blueprint("eda", __name__)


def get_pipeline_or_error(pipeline_id):

    pipeline = PIPELINES.get(pipeline_id)

    if pipeline is None:
        return None, (
            jsonify({"error": "Pipeline not found"}),
            404
        )

    return pipeline, None


@eda_bp.route("/summary/<pipeline_id>", methods=["GET"])
def summary(pipeline_id):

    pipeline, error = get_pipeline_or_error(pipeline_id)

    if error:
        return error

    X = pipeline["X"]
    y = pipeline["y"]

    analyzer = EDAAnalyzer()

    y_series = y[y.columns[0]]

    summary = analyzer.get_eda_summary(
        X,
        target=y_series
    )

    descriptive_stats = X.describe().to_dict()

    pipeline["results"]["eda_summary"] = summary

    return jsonify({
        "descriptive_statistics": descriptive_stats,
        "summary": summary
    })


@eda_bp.route("/correlation/<pipeline_id>", methods=["POST"])
def correlation(pipeline_id):

    pipeline, error = get_pipeline_or_error(pipeline_id)

    if error:
        return error

    X = pipeline["X"]

    analyzer = EDAAnalyzer()

    fig = analyzer.plot_correlation_matrix(
        X,
        title="Correlation Matrix",
        show=False
    )

    filepath = f"outputs/{pipeline_id}_correlation.png"

    fig.savefig(filepath)

    pipeline["results"]["correlation_matrix"] = filepath

    return jsonify({
        "message": "Correlation matrix generated",
        "file": filepath
    })


@eda_bp.route("/distributions/<pipeline_id>", methods=["POST"])
def distributions(pipeline_id):

    pipeline, error = get_pipeline_or_error(pipeline_id)

    if error:
        return error

    X = pipeline["X"]

    analyzer = EDAAnalyzer()

    fig = analyzer.plot_feature_distributions(
        X,
        show=False
    )

    filepath = f"outputs/{pipeline_id}_distributions.png"

    fig.savefig(filepath)

    pipeline["results"]["distributions"] = filepath

    return jsonify({
        "message": "Feature distributions generated",
        "file": filepath
    })


@eda_bp.route("/boxplots/<pipeline_id>", methods=["POST"])
def boxplots(pipeline_id):

    pipeline, error = get_pipeline_or_error(pipeline_id)

    if error:
        return error

    X = pipeline["X"]

    analyzer = EDAAnalyzer()

    fig = analyzer.plot_boxplots_outliers(
        X,
        show=False
    )

    filepath = f"outputs/{pipeline_id}_boxplots.png"

    fig.savefig(filepath)

    pipeline["results"]["boxplots"] = filepath

    return jsonify({
        "message": "Boxplots generated",
        "file": filepath
    })


