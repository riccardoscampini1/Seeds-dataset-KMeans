from flask import Flask
from api.loader_bp import loader_bp
from api.eda_bp import eda_bp
from api.outliers_bp import outliers_bp
from api.scaling_bp import scaling_bp
from api.pca_bp import pca_bp
from api.multicol_bp import multicol_bp
from api.kmeans_bp import kmeans_bp
from api.evaluation_bp import evaluation_bp


app = Flask(__name__)
app.register_blueprint(loader_bp)
app.register_blueprint(eda_bp)
app.register_blueprint(outliers_bp)
app.register_blueprint(scaling_bp)
app.register_blueprint(pca_bp)
app.register_blueprint(multicol_bp)
app.register_blueprint(kmeans_bp)
app.register_blueprint(evaluation_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)