
"""
Main entry point for K-Means Clustering Pipeline.

Complete ML pipeline for seeds dataset preprocessing, EDA, and K-Means clustering.
"""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from src import get_logger

# Import pipeline components
from src import DataLoader
from src import EDAAnalyzer
from src import FeatureScaler
from src import PCAProcessor
from src import MulticollinearityReducer
from src import KMeansModel
from src import FeatureEngineering

logger = get_logger(__name__)


def save_figure(fig, output_dir: str, filename: str) -> None:
    """Save and close a matplotlib figure."""
    if fig is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def main():
    """
    Main ML pipeline for K-Means clustering.
    
    Steps:
    1. Load dataset and split into X, y
    2. Exploratory Data Analysis (EDA)
    3. Outlier removal (IQR method)
    4. Feature scaling (Robust Scaler)
    5. Case 1: Dimensionality reduction (PCA) + K-Means
    6. Case 2: Multicollinearity reduction + K-Means
    7. Model comparison
    """
    try:
        figures_dir = os.path.join(project_root, "figures")

        # ══════════════════════════════════════════════════════════════════════════════
        # 1. LOAD DATASET
        # ══════════════════════════════════════════════════════════════════════════════
        print_header("STEP 1: LOADING DATASET")
        
        loader = DataLoader()
        X, y = loader.load_seeds_dataset('data/seeds_dataset.txt')
        y_series = y[y.columns[0]]
        
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"[OK] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Features: {list(X.columns)}")
        print(f"  Target: {y.columns[0]}")
        
        # ══════════════════════════════════════════════════════════════════════════════
        # 2. DESCRIPTIVE STATISTICS AND EDA
        # ══════════════════════════════════════════════════════════════════════════════
        print_header("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
        
        analyzer = EDAAnalyzer()
        
        # Print descriptive statistics
        print("[INFO] Descriptive Statistics:")
        print(X.describe().to_string())
        print()
        
        # Get EDA summary
        summary = analyzer.get_eda_summary(X, target=y_series)
        print("[INFO] Missing values:")
        if summary['missing_values']:
            for col, stats in summary['missing_values'].items():
                print(f"  {col}: {stats['count']} ({stats['pct']}%)")
        else:
            print("  No missing values detected [OK]")
        
        # ══════════════════════════════════════════════════════════════════════════════
        # 3. CORRELATION ANALYSIS
        # ══════════════════════════════════════════════════════════════════════════════
        print_header("STEP 3: CORRELATION MATRIX")
        
        print("[INFO] Saving correlation matrix...")
        fig = analyzer.plot_correlation_matrix(
            X,
            title="Correlation Matrix - Original Data",
            show=False,
        )
        save_figure(fig, figures_dir, "01_correlation_matrix.png")
        
        # ══════════════════════════════════════════════════════════════════════════════
        # 4. FEATURE DISTRIBUTIONS
        # ══════════════════════════════════════════════════════════════════════════════
        print_header("STEP 4: FEATURE DISTRIBUTIONS")
        
        print("[INFO] Saving feature distributions...")
        fig = analyzer.plot_feature_distributions(X, show=False)
        save_figure(fig, figures_dir, "02_feature_distributions.png")
        
        # ══════════════════════════════════════════════════════════════════════════════
        # 5. OUTLIER DETECTION & VISUALIZATION
        # ══════════════════════════════════════════════════════════════════════════════
        print_header("STEP 5: OUTLIER DETECTION & VISUALIZATION")
        
        print("[INFO] Analyzing outliers using IQR method...")
        print("\nBoxplots of original data:")
        fig = analyzer.plot_boxplots_outliers(X, show=False)
        save_figure(fig, figures_dir, "03_original_boxplots.png")
        
        analyzer.print_outlier_report()
        
        # ══════════════════════════════════════════════════════════════════════════════
        # 6. OUTLIER REMOVAL
        # ══════════════════════════════════════════════════════════════════════════════
        print_header("STEP 6: OUTLIER REMOVAL")
        
        print("[INFO] Removing outliers with IQR method...")
        X_clean, outlier_report = analyzer.remove_outliers_iqr(X, iqr_threshold=1.5)
        y_clean = y_series.loc[X_clean.index]

        print("[OK] Outliers removed!")
        print(f"  Before: {X.shape[0]} samples")
        print(f"  After:  {X_clean.shape[0]} samples")
        print(f"  Removed: {X.shape[0] - X_clean.shape[0]} samples ({(X.shape[0] - X_clean.shape[0]) / X.shape[0] * 100:.2f}%)")
        
        print("\nComparative distributions (Before vs After):")
        fig = analyzer.plot_comparative_distributions(
            X,
            X_clean,
            title_suffix="(IQR Outlier Removal)",
            show=False,
        )
        save_figure(fig, figures_dir, "04_distributions_before_after_outlier_removal.png")
        
        # -----
        # FEATURE ENGINEERING (PERIMETRO/AREA, LENGTH/WIDTH)
        # ------
        print_header("FEATURE ENGINEERING (PERIMETER/AREA, LENGTH/WIDTH)")
        print("[INFO] Creating engineered features...")

        fe = FeatureEngineering()

        X_fe = fe.create_ratio_feature(
            X_clean,
            numerator="perimeter",
            denominator="area",
            new_feature_name="perimeter_area_ratio",
            drop_originals=True,
        )

        X_fe = fe.create_ratio_feature(
            X_fe,
            numerator="length_kernel",
            denominator="width_kernel",
            new_feature_name="length_width_ratio",
            drop_originals=True,
        )

        print("[OK] Feature engineering completed!")

        print(f"Original shape: {X_clean.shape}")
        print(f"New shape:      {X_fe.shape}")

        print("\nFeatures:")
        print(list(X_fe.columns))
        print()


        # ══════════════════════════════════════════════════════════════════════════════
        # 7. FEATURE SCALING (ROBUST SCALER)
        # ══════════════════════════════════════════════════════════════════════════════
        print_header("STEP 7: FEATURE SCALING (ROBUST SCALER)")
        
        print("[INFO] Applying Robust Scaling...")
        scaler = FeatureScaler()
        X_scaled = scaler.scale_robust(X_clean)
        
        scaler_info = scaler.get_scaler_info()
        print("[OK] Robust scaling applied!")
        print(f"  Scaler type: {scaler_info['scaler_type']}")
        print(f"  Columns scaled: {scaler_info['n_columns_scaled']}")
        print(f"\nScaled data preview:")
        print(X_scaled.describe().to_string())

        print("[INFO] Applying Robust Scaling to Feature Engineered DF...")

        X_fe_scaled = scaler.scale_robust(X_fe)
        
        scaler_fe_info = scaler.get_scaler_info()
        print("[OK] Robust scaling applied!")
        print(f"  Scaler type: {scaler_fe_info['scaler_type']}")
        print(f"  Columns scaled: {scaler_fe_info['n_columns_scaled']}")
        print(f"\nScaled data preview:")
        print(X_fe_scaled.describe().to_string())
        
        # ══════════════════════════════════════════════════════════════════════════════
        # 8. DIMENSIONALITY REDUCTION (PCA)
        # ══════════════════════════════════════════════════════════════════════════════
        print_header("STEP 8: CASE 1 - PCA + K-MEANS")
        
        print("[INFO] Applying PCA to reduce dimensions...")
        pca = PCAProcessor()
        pca.configure(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        pca.print_report()
        
        print("\n[INFO] Saving PCA variance...")
        fig = pca.plot_explained_variance(show=False)
        save_figure(fig, figures_dir, "05_pca_explained_variance.png")
        
        print("\n[INFO] Saving 2D PCA projection...")
        fig = pca.plot_pca_2d(X_pca, show=False)
        save_figure(fig, figures_dir, "06_pca_2d_projection.png")

        print_header("STEP 8: CASE 2 - FE + PCA + K-MEANS")

        print("[INFO] Applying PCA to reduce dimensions...")
        pca_fe = PCAProcessor()
        pca_fe.configure(n_components=0.95, random_state=42)
        X_fe_pca = pca_fe.fit_transform(X_fe_scaled)
        
        pca_fe.print_report()
        
        print("\n[INFO] Saving PCA variance...")
        fig = pca_fe.plot_explained_variance(show=False)
        save_figure(fig, figures_dir, "05_pca_fe_explained_variance.png")
        
        print("\n[INFO] Saving 2D PCA projection...")
        fig = pca_fe.plot_pca_2d(X_fe_pca, show=False)
        save_figure(fig, figures_dir, "06_pca_fe_2d_projection.png")
        
        # ══════════════════════════════════════════════════════════════════════════════
        # 9. K-MEANS TRAINING
        # ══════════════════════════════════════════════════════════════════════════════
        print("[INFO] Training K-Means model on PCA data...")
        kmeans_pca = KMeansModel()
        kmeans_pca.train(X_pca, verbose=True)
        kmeans_pca.print_summary()

        labels_pca = kmeans_pca.predict(X_pca)

        print("[INFO] Saving PCA K-Means elbow curve...")
        fig = kmeans_pca.plot_elbow(X_pca, n_clusters_range=(2, 10), show=False)
        save_figure(fig, figures_dir, "07_pca_kmeans_elbow_curve.png")

        print("\n[INFO] Saving PCA K-Means clusters in 2D...")
        fig = kmeans_pca.plot_clusters_2d(X_pca, labels=labels_pca, pca_components=0, show=False)
        save_figure(fig, figures_dir, "08_pca_kmeans_clusters_2d.png")

        metrics_pca = kmeans_pca.evaluate(X_pca, labels=labels_pca)
        ari_pca = adjusted_rand_score(y_clean, labels_pca)

        print("[INFO] PCA K-Means Evaluation Results:")
        print(f"  - Silhouette Score:     {metrics_pca['silhouette_score']:.4f} (higher is better, range: -1 to 1)")
        print(f"  - Davies-Bouldin Index: {metrics_pca['davies_bouldin_index']:.4f} (lower is better)")
        print(f"  - Inertia:              {metrics_pca['inertia']:.4f} (lower is better)")
        print(f"  - Number of Clusters:   {metrics_pca['n_clusters']}")
        print(f"  - Adjusted Rand Index:  {ari_pca:.4f} (higher is better, range: -1 to 1)")


        print("[INFO] Training K-Means model on Feature Engineered PCA data...")
        kmeans_fe_pca = KMeansModel()
        kmeans_fe_pca.train(X_fe_pca, verbose=True)
        kmeans_fe_pca.print_summary()

        labels_fe_pca = kmeans_fe_pca.predict(X_fe_pca)

        print("[INFO] Saving Feature Engineered PCA K-Means elbow curve...")
        fig = kmeans_fe_pca.plot_elbow(X_fe_pca, n_clusters_range=(2, 10), show=False)
        save_figure(fig, figures_dir, "07_fe_pca_kmeans_elbow_curve.png")

        print("\n[INFO] Saving Feature Engineered PCA K-Means clusters in 2D...")
        fig = kmeans_fe_pca.plot_clusters_2d(X_fe_pca, labels=labels_fe_pca, pca_components=0, show=False)
        save_figure(fig, figures_dir, "08_fe_pca_kmeans_clusters_2d.png")

        metrics_fe_pca = kmeans_fe_pca.evaluate(X_fe_pca, labels=labels_fe_pca)
        ari_fe_pca = adjusted_rand_score(y_clean, labels_fe_pca)

        print("[INFO] Feature Engineered PCA K-Means Evaluation Results:")
        print(f"  - Silhouette Score:     {metrics_fe_pca['silhouette_score']:.4f} (higher is better, range: -1 to 1)")
        print(f"  - Davies-Bouldin Index: {metrics_fe_pca['davies_bouldin_index']:.4f} (lower is better)")
        print(f"  - Inertia:              {metrics_fe_pca['inertia']:.4f} (lower is better)")
        print(f"  - Number of Clusters:   {metrics_fe_pca['n_clusters']}")
        print(f"  - Adjusted Rand Index:  {ari_fe_pca:.4f} (higher is better, range: -1 to 1)")    

        print_header("STEP 9: CASE 2 - MULTICOLLINEARITY REDUCTION + K-MEANS")

        print("[INFO] Removing multicollinear variables with |corr| > 0.85...")
        multicollinearity = MulticollinearityReducer(threshold=0.85)
        X_multicollinearity = multicollinearity.fit_transform(X_scaled)
        multicollinearity_report = multicollinearity.report()

        print("[INFO] Correlated pairs found:")
        if multicollinearity_report["pairs"]:
            for pair in multicollinearity_report["pairs"]:
                print(f"  - {pair['col_a']} <-> {pair['col_b']}: r = {pair['r']:.4f}")
        else:
            print("  No correlated pairs above threshold.")

        print(f"[INFO] Columns kept among correlated groups: {multicollinearity_report['cols_to_keep']}")
        print(f"[INFO] Columns dropped: {multicollinearity_report['cols_to_drop']}")
        print(f"[OK] Shape after multicollinearity reduction: {X_multicollinearity.shape}")

        print("[INFO] Saving correlation matrix after multicollinearity reduction...")
        fig = analyzer.plot_correlation_matrix(
            X_multicollinearity,
            title="Correlation Matrix - After Multicollinearity Reduction",
            show=False,
        )
        save_figure(fig, figures_dir, "09_multicollinearity_reduced_correlation_matrix.png")

        print("[INFO] Training K-Means model on multicollinearity-reduced data...")
        kmeans_multicollinearity = KMeansModel()
        kmeans_multicollinearity.train(X_multicollinearity, verbose=True)
        kmeans_multicollinearity.print_summary()

        labels_multicollinearity = kmeans_multicollinearity.predict(X_multicollinearity)

        print("[INFO] Saving multicollinearity K-Means elbow curve...")
        fig = kmeans_multicollinearity.plot_elbow(
            X_multicollinearity,
            n_clusters_range=(2, 10),
            show=False,
        )
        save_figure(fig, figures_dir, "10_multicollinearity_kmeans_elbow_curve.png")

        print("\n[INFO] Saving multicollinearity K-Means clusters in 2D...")
        fig = kmeans_multicollinearity.plot_clusters_2d(
            X_multicollinearity,
            labels=labels_multicollinearity,
            pca_components=2,
            show=False,
        )
        save_figure(fig, figures_dir, "11_multicollinearity_kmeans_clusters_2d.png")

        metrics_multicollinearity = kmeans_multicollinearity.evaluate(
            X_multicollinearity,
            labels=labels_multicollinearity,
        )
        ari_multicollinearity = adjusted_rand_score(y_clean, labels_multicollinearity)

        print("[INFO] Multicollinearity K-Means Evaluation Results:")
        print(f"  - Silhouette Score:     {metrics_multicollinearity['silhouette_score']:.4f} (higher is better, range: -1 to 1)")
        print(f"  - Davies-Bouldin Index: {metrics_multicollinearity['davies_bouldin_index']:.4f} (lower is better)")
        print(f"  - Inertia:              {metrics_multicollinearity['inertia']:.4f} (lower is better)")
        print(f"  - Adjusted Rand Index:  {ari_multicollinearity:.4f} (higher is better, range: -1 to 1)")
        print(f"  - Number of Clusters:   {metrics_multicollinearity['n_clusters']}")

        print_header(
    "STEP 9.1: FEATURE ENGINEERING + MULTICOLLINEARITY + K-MEANS"
)
        print(
            "[INFO] Removing multicollinear variables "
            "with |corr| > 0.9..."
        )

        multicollinearity = (
            MulticollinearityReducer(
                threshold=0.9
            )
        )

        X_fe_multicollinearity = (
            multicollinearity.fit_transform(
                X_fe
            )
        )

        multicollinearity_report = (
            multicollinearity.report()
        )


        print("[INFO] Correlated pairs found:")

        if multicollinearity_report["pairs"]:

            for pair in multicollinearity_report["pairs"]:

                print(
                    f"  - "
                    f"{pair['col_a']} "
                    f"<-> "
                    f"{pair['col_b']}: "
                    f"r = {pair['r']:.4f}"
                )

        else:

            print(
            "  No correlated pairs "
                "above threshold."
            )

        print(
            f"[INFO] Columns kept: "
            f"{multicollinearity_report['cols_to_keep']}"
        )

        print(
            f"[INFO] Columns dropped: "
            f"{multicollinearity_report['cols_to_drop']}"
        )

        print(
            f"[OK] Shape after reduction: "
            f"{X_fe_multicollinearity.shape}"
        )


        print(
            "[INFO] Training K-Means model "
            "on engineered data..."
    )

        kmeans_fe_multicollinearity = KMeansModel()

        kmeans_fe_multicollinearity.train(
            X_fe_multicollinearity,
            verbose=True,
        )   

        kmeans_fe_multicollinearity.print_summary()

        labels_fe = kmeans_fe_multicollinearity.predict(
            X_fe_multicollinearity
        )

        print(
            "[INFO] Saving elbow curve..."
        )

        fig = kmeans_fe_multicollinearity.plot_elbow(
            X_fe_multicollinearity,
            n_clusters_range=(2, 10),
            show=False,
    )

        save_figure(
            fig,
            figures_dir,
            "12_fe_multicollinearity_elbow_curve.png",
        )


        print(
            "\n[INFO] Saving clusters..."
        )

        fig = kmeans_fe_multicollinearity.plot_clusters_2d(
            X_fe_multicollinearity,
            labels=labels_fe,
            pca_components=2,
            show=False,
        )

        save_figure(
            fig,
            figures_dir,
            "13_fe_multicollinearity_clusters_2d.png",
        )

        metrics_fe_multicollinearity = kmeans_fe_multicollinearity.evaluate(
            X_fe_multicollinearity,
            labels=labels_fe,
        )

        ari_fe_multicollinearity = adjusted_rand_score(
            y_clean,
            labels_fe,
        )

        print(
            f"""
        [INFO] Feature Engineering + Multicollinearity K-Means Results:
        - Silhouette Score:     {metrics_fe_multicollinearity['silhouette_score']:.4f}
        - Davies-Bouldin Index: {metrics_fe_multicollinearity['davies_bouldin_index']:.4f}
        - Inertia:              {metrics_fe_multicollinearity['inertia']:.4f}
        - Adjusted Rand Index:  {ari_fe_multicollinearity:.4f}
        - Number of Clusters:   {metrics_fe_multicollinearity['n_clusters']}
        """
        )



        print_header("PIPELINE SUMMARY")
        
        # ══════════════════════════════════════════════════════════════════════════════
        # 10. CLUSTERING VISUALIZATION
        # ══════════════════════════════════════════════════════════════════════════════
        
        
        # ══════════════════════════════════════════════════════════════════════════════
        # 11. EVALUATION METRICS
        # ══════════════════════════════════════════════════════════════════════════════
        
        
        # ══════════════════════════════════════════════════════════════════════════════
        # PIPELINE SUMMARY
        # ══════════════════════════════════════════════════════════════════════════════
        print("[OK] Pipeline completed successfully!\n")
        print("[INFO] Summary Statistics:")
        print(f"  Original dataset:      {X.shape}")
        print(f"  After outlier removal: {X_clean.shape}")
        print(f"  After PCA (0.95 var):  {X_pca.shape}")
        print(f"  After multicollinearity reduction: {X_multicollinearity.shape}")
        print()
        print("[INFO] Model Comparison:")
        comparison = pd.DataFrame(
            [
                {
                    "model": "PCA + K-Means",
                    "features": X_pca.shape[1],
                    "silhouette_score": metrics_pca["silhouette_score"],
                    "davies_bouldin_index": metrics_pca["davies_bouldin_index"],
                    "inertia": metrics_pca["inertia"],
                    "adjusted_rand_index": ari_pca,
                    "n_clusters": metrics_pca["n_clusters"],
                },
                {
                    "model": "Multicollinearity + K-Means",
                    "features": X_multicollinearity.shape[1],
                    "silhouette_score": metrics_multicollinearity["silhouette_score"],
                    "davies_bouldin_index": metrics_multicollinearity["davies_bouldin_index"],
                    "inertia": metrics_multicollinearity["inertia"],
                    "adjusted_rand_index": ari_multicollinearity,
                    "n_clusters": metrics_multicollinearity["n_clusters"],
                },
                {
                    "model": "FE + PCA + K-Means",
                    "features": X_fe_pca.shape[1],
                    "silhouette_score": metrics_fe_pca["silhouette_score"],
                    "davies_bouldin_index": metrics_fe_pca["davies_bouldin_index"],
                    "inertia": metrics_fe_pca["inertia"],
                    "adjusted_rand_index": ari_fe_pca,
                    "n_clusters": metrics_fe_pca["n_clusters"],
                },
                {
                    "model": "FE + Multicollinearity + K-Means",
                    "features": X_fe_multicollinearity.shape[1],
                    "silhouette_score": metrics_fe_multicollinearity["silhouette_score"],
                    "davies_bouldin_index": metrics_fe_multicollinearity["davies_bouldin_index"],
                    "inertia": metrics_fe_multicollinearity["inertia"],
                    "adjusted_rand_index": ari_fe_multicollinearity,
                    "n_clusters": metrics_fe_multicollinearity["n_clusters"],
                },
            ]
        )
        print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print("\n")
        
        logger.info("ML pipeline executed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
