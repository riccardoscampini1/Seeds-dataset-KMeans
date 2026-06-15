from flask import Blueprint, request, jsonify
import pandas as pd

from src.eda.eda import EDAAnalyzer  # ipotizzo la tua classe

outliers_bp = Blueprint(
    'outliers',
    __name__,
    url_prefix='/outliers'
)

# inizializzazione globale (puoi anche iniettarla meglio se vuoi DI)
analyzer = EDAAnalyzer()


@outliers_bp.route('/detect', methods=['POST'])
def detect_outliers():
    """
    Detect outliers using IQR method
    """
    data = request.get_json()

    df = pd.DataFrame(data["X"])

    iqr_threshold = data.get("iqr_threshold", 1.5)

    report = analyzer.detect_outliers_iqr(df, iqr_threshold=iqr_threshold)

    return jsonify({
        "status": "success",
        "outlier_report": report
    })


@outliers_bp.route('/remove', methods=['POST'])
def remove_outliers():
    """
    Remove outliers using IQR method (STEP 6 equivalente al tuo main)
    """
    data = request.get_json()

    X = pd.DataFrame(data["X"])
    y = pd.Series(data["y"])

    iqr_threshold = data.get("iqr_threshold", 1.5)

    X_clean, report = analyzer.remove_outliers_iqr(
        X,
        iqr_threshold=iqr_threshold
    )

    y_clean = y.loc[X_clean.index]

    return jsonify({
        "status": "success",
        "before_shape": X.shape[0],
        "after_shape": X_clean.shape[0],
        "removed": int(X.shape[0] - X_clean.shape[0]),
        "removed_pct": float(
            (X.shape[0] - X_clean.shape[0]) / X.shape[0] * 100
        ),
        "outlier_report": report,
        "X_clean": X_clean.to_dict(orient="records"),
        "y_clean": y_clean.to_dict()
    })


@outliers_bp.route('/summary', methods=['POST'])
def outliers_summary():
    """
    Quick summary without modifying dataset
    """
    data = request.get_json()

    X = pd.DataFrame(data["X"])

    report = analyzer.detect_outliers_iqr(X)

    return jsonify({
        "status": "success",
        "n_samples": X.shape[0],
        "outlier_report": report
    })


