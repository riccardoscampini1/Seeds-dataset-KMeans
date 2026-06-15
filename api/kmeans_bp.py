from flask import Blueprint, jsonify, request

from src import KMeansModel
from sklearn.metrics import adjusted_rand_score

from .pipeline_store import PIPELINES

kmeans_bp = Blueprint("kmeans", __name__)


def get_pipeline(pipeline_id):

    pipeline = PIPELINES.get(pipeline_id)

    if pipeline is None:
        return None, (
            jsonify({"error": "Pipeline not found"}),
            404
        )

    return pipeline, None


@kmeans_bp.route(
    "/kmeans/train/<pipeline_id>",
    methods=["POST"]
)
def train_kmeans(pipeline_id):

    pipeline, error = get_pipeline(pipeline_id)
    if error:
        return error

    data = request.get_json(silent=True) or {}

    dataset_name = data.get("dataset", "pca")

    X = pipeline["datasets"].get(dataset_name)
    y = pipeline.get("y")

    if X is None:
        return jsonify({
            "error": f"Dataset '{dataset_name}' not found"
        }), 400

    print("[INFO] Training K-Means model...")

    model = KMeansModel()

    model.train(X, verbose=True)

    labels = model.predict(X)

    metrics = model.evaluate(X, labels)

    ari = adjusted_rand_score(
        pipeline["y_original"].iloc[:, 0],
        labels
    )

    pipeline["artifacts"][f"kmeans_{dataset_name}"] = model

    pipeline["results"]["kmeans"] = pipeline["results"].get("kmeans", {})

    pipeline["results"]["kmeans"][dataset_name] = {
        "silhouette_score": metrics["silhouette_score"],
        "davies_bouldin_index": metrics["davies_bouldin_index"],
        "inertia": metrics["inertia"],
        "n_clusters": metrics["n_clusters"],
        "ari": ari
    }

    return jsonify({
        "message": "KMeans training completed",
        "dataset": dataset_name,
        "silhouette_score": metrics["silhouette_score"],
        "davies_bouldin_index": metrics["davies_bouldin_index"],
        "inertia": metrics["inertia"],
        "ari": ari
    })


@kmeans_bp.route(
    "/kmeans/elbow/<pipeline_id>",
    methods=["POST"]
)
def elbow(pipeline_id):

    pipeline, error = get_pipeline(pipeline_id)
    if error:
        return error

    data = request.get_json(silent=True) or {}
    dataset_name = data.get("dataset", "pca")

    X = pipeline["datasets"].get(dataset_name)

    model = pipeline["artifacts"].get(
        f"kmeans_{dataset_name}"
    )

    if model is None:
        model = KMeansModel()

    fig = model.plot_elbow(
        X,
        n_clusters_range=(2, 10),
        show=False
    )

    path = f"outputs/{pipeline_id}_kmeans_elbow_{dataset_name}.png"
    fig.savefig(path)

    return jsonify({
        "message": "Elbow curve saved",
        "file": path
    })


@kmeans_bp.route(
    "/kmeans/clusters/<pipeline_id>",
    methods=["POST"]
)
def clusters(pipeline_id):

    pipeline, error = get_pipeline(pipeline_id)
    if error:
        return error

    data = request.get_json(silent=True) or {}
    dataset_name = data.get("dataset", "pca")

    X = pipeline["datasets"].get(dataset_name)

    model = pipeline["artifacts"].get(
        f"kmeans_{dataset_name}"
    )

    if model is None:
        return jsonify({
            "error": "Model not trained yet"
        }), 400

    labels = model.predict(X)

    fig = model.plot_clusters_2d(
        X,
        labels=labels,
        pca_components=0,
        show=False
    )

    path = f"outputs/{pipeline_id}_clusters_{dataset_name}.png"
    fig.savefig(path)

    return jsonify({
        "message": "Cluster plot saved",
        "file": path
    })