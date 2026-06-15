from flask import Blueprint, jsonify
import pandas as pd

from .pipeline_store import PIPELINES

evaluation_bp = Blueprint(
    "evaluation",
    __name__
)


@evaluation_bp.route(
    "/evaluation/kmeans/<pipeline_id>",
    methods=["GET"]
)
def compare_kmeans(pipeline_id):

    pipeline = PIPELINES.get(pipeline_id)

    if pipeline is None:
        return jsonify({
            "error": "Pipeline not found"
        }), 404

    results = pipeline["results"].get("kmeans")

    if results is None:
        return jsonify({
            "error": "KMeans results not found"
        }), 400

    comparison = pd.DataFrame(
        [
            {
                "model": "PCA + K-Means",
                "features": pipeline["datasets"]["pca"].shape[1],
                "silhouette_score": results["pca"]["silhouette_score"],
                "davies_bouldin_index": results["pca"]["davies_bouldin_index"],
                "inertia": results["pca"]["inertia"],
                "adjusted_rand_index": results["pca"]["ari"],
                "n_clusters": results["pca"]["n_clusters"],
            },
            {
                "model": "Multicollinearity + K-Means",
                "features": pipeline["datasets"]["multicollinearity"].shape[1],
                "silhouette_score": results["multicollinearity"]["silhouette_score"],
                "davies_bouldin_index": results["multicollinearity"]["davies_bouldin_index"],
                "inertia": results["multicollinearity"]["inertia"],
                "adjusted_rand_index": results["multicollinearity"]["ari"],
                "n_clusters": results["multicollinearity"]["n_clusters"],
            },
        ]
    )

    return jsonify({
        "comparison": comparison.to_dict(orient="records")
    })    
