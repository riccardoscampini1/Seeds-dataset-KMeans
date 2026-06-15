from flask import Blueprint

home_bp = Blueprint("home", __name__)

@home_bp.route("/", methods=["GET"])
def intro():
    return """
        <h1>--- BENVENUTO NELL'API K-MEANS ---</h1>

        <p>
            REST API per l'analisi del dataset Seeds e la divisione
            dei vari semi in clusters attraverso l'uso di k-means
        </p>

        <h2>CARICAMENTO DEL DATASET:</h2>

        <h3>Dataset Analysis e Plots</h3>
        <ul>
            <li>POST /load</li>
            <li>CARICARE IL DATASET E PRELEVARE L'UUID RESTITUITO, DDA INSERIRE PER TUTTI GLI ALTRI ENDPOINTS</li>
        </ul>
        
        <h2>ENDPOINTS:</h2>

        <h3>Dataset Analysis e Plots</h3>
        <ul>
            <li>GET /summary</li>
            <li>POST /distributions</li>
            <li>POST /correlations</li>
            <li>POST /boxplots</li>
            <li>POST /outliers/remove</li>
            <li>POST /outliers/distributions</li>
        </ul>

        <h3>Scaling</h3>
        <ul>
            <li>POST /scaling</li>
            <li>GET /scaling/report</li>
        </ul>
        
        <h3>PCA</h3>
        <ul>
            <li>POST /pca</li>
            <li>GET /pca/report</li>
        </ul>
        
        <h3>K-means</h3>
        <ul>
            <li>GET /evaluation/kmeans</li>
            <li>POST /kmeans/train</li>
            <li>POST /kmeans/elbow</li>
            <li>POST /kmeans/clusters</li>
        </ul>

        """