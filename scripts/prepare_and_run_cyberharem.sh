python utils/cyberharem_utils.py --output_dir cyber_run_1 # this runs dataset preparation
python utils/config_util.py --json_file configs/multi_training_cat_default.json --dataset_json cyber_run_1/metadata.jsonl --project_name cyberharem_auto2 --outputs_dir results # prepare the config file
# results/cyberharem_auto1/cyberharem_auto1.json
python cat_train_all.py --tuning_config_path results/cyberharem_auto2/cyberharem_auto2.json
