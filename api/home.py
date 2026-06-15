from flask import Blueprint

home_bp = Blueprint("home", __name__)

@home_bp.route("/", methods=["GET"])
def intro():
    return """
    -- BENVENUTO NEL PROGETTO K-MEANS --
    """