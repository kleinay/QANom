# format data
python qanom/predicate_detector/prepare_qanom_data.py

# train a new model (optional)
sh qanom/predicate_detector/train_nom_id.sh

# predict using a trained model
sh qanom/predicate_detector/predict_nom_id.sh