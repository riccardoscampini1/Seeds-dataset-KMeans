from flask import Blueprint, request, jsonify
from lib.loader import GenericLoader

loader_bp = Blueprint('loader', __name__, url_prefix='/loader')

_loader: GenericLoader = None


@loader_bp.post('/load')
def load():
    """
    Body JSON supportato:

    {
        "csv_path": "data/train.csv",
        "dataset_id": 1,
        "target_col": "Survived",
        "drop_cols": [...],
        "drop_missing_thresh": 0.6,
        "sep": ";",
        "engine": "python",
        "header": "infer",
        "names": [...]
    }
    """

    global _loader

    body = request.get_json(force=True)

    # mandatory
    csv_path   = body.get('csv_path')
    target_col = body.get('target_col')

    if not csv_path or not target_col:
        return jsonify({
            'error': 'csv_path e target_col sono obbligatori'
        }), 400

    # optional params (nuova firma completa)
    dataset_id = body.get('dataset_id')

    drop_cols = body.get('drop_cols') or []
    thresh    = body.get('drop_missing_thresh')

    sep    = body.get('sep', ';')
    engine = body.get('engine', None)
    header = body.get('header', 'infer')
    names  = body.get('names', None)

    try:
        _loader = GenericLoader(
            target_col=target_col,
            dataset_id=dataset_id,
            csv_path=csv_path,
            drop_cols=drop_cols,
            drop_missing_thresh=thresh,
            sep=sep,
            engine=engine,
            header=header,
            names=names,
        )

        _loader.load()

        return jsonify({
            "status": "success",
            "info": _loader.info()
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@loader_bp.get('/info')
def info():
    if _loader is None:
        return jsonify({'error': 'Dataset non caricato. Chiama prima POST /loader/load'}), 400
    return jsonify(_loader.info()), 200


@loader_bp.get('/missing')
def missing():
    if _loader is None:
        return jsonify({'error': 'Dataset non caricato. Chiama prima POST /loader/load'}), 400
    return jsonify(_loader.missing_report()), 200
