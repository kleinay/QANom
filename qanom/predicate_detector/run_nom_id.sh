#CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 run_nom_id.py --overwrite_cache --data_dir dataset \
#--model_type bert --tokenizer bert-base-cased --model_name_or_path bert-base-cased --output_dir \
#model --num_train_epochs 5 --do_train --do_eval --do_predict --overwrite_output_dir --labels \
#dataset/labels.txt --evaluate_during_training --max_seq_length 400

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 run_nom_id.py --data_dir dataset \
--tokenizer bert-base-cased --model_name_or_path model/checkpoint-500 --output_dir \
model/checkpoint-500 --do_predict --labels dataset/labels.txt --max_seq_length 400