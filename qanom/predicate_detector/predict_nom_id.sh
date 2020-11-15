export CUDA_VISIBLE_DEVICES=0,1,2,3

python3.6 qanom/predicate_detector/run_nom_id.py \
--data_dir output/predicate_detector \
--tokenizer bert-base-cased \
--model_name_or_path model/predicate_detector/ \
--output_dir output/predicate_detector/ \
--do_predict \
--do_predict_performance \
--max_seq_length 400 \
--labels qanom/predicate_detector/labels.txt