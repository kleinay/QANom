export CUDA_VISIBLE_DEVICES=0,1,2,3

python3.6 qanom/predicate_detector/run_nom_id.py \
--data_dir output/predicate_detector \
--model_type bert \
--tokenizer bert-base-cased \
--model_name_or_path bert-base-cased \
--output_dir model/predicate_detector/ \
--num_train_epochs 5 \
--do_train \
--do_eval \
--do_predict \
--evaluate_during_training \
--max_seq_length 400 \
--labels qanom/predicate_detector/labels.txt