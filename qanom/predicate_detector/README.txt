python prepare_qanom_data.py --csv path_to_csv_from_QANom
python run_nom_id.py --overwrite_cache --data_dir dataset --tokenizer bert-base-cased --output_dir model/ --model_name_or_path model/ --do_predict --max_seq_length 400 --labels labels.txt

python prepare_qanom_data.py --csv dataset/nombank/qanom_annot_nombank_aligned_sample.csv --txt dataset/nombank/test.txt
/09/2020 17:47:38 - INFO - transformers.trainer -   ***** Running Prediction *****
06/09/2020 17:47:38 - INFO - transformers.trainer -     Num examples = 126
06/09/2020 17:47:38 - INFO - transformers.trainer -     Batch size = 8
Prediction: 100%|██████████| 16/16 [01:16<00:00,  4.78s/it]
06/09/2020 17:48:58 - INFO - __main__ -     eval_loss = 0.32233594357967377
06/09/2020 17:48:58 - INFO - __main__ -     eval_accuracy = 0.8756613756613757
06/09/2020 17:48:58 - INFO - __main__ -     eval_precision = 0.8634146341463415
06/09/2020 17:48:58 - INFO - __main__ -     eval_recall = 0.9030612244897959
06/09/2020 17:48:58 - INFO - __main__ -     eval_f1 = 0.8827930174563591


python run_nom_id.py --overwrite_cache --data_dir dataset/nombank/ --tokenizer bert-base-cased --output_dir dataset/nombank/ --model_name_or_path model/ --do_predict --do_predict_performance --max_seq_length 400 --labels labels.txt

QA-NOM
python run_nom_id.py --overwrite_cache --data_dir dataset/qa-nom --tokenizer bert-base-cased --output_dir dataset/qa-nom --model_name_or_path model/ --do_predict --do_predict_performance --max_seq_length 400 --labels labels.txt