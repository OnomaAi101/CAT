python utils/cyberharem_utils.py # this runs dataset preparation
python utils/config_util.py --json_file configs/multi_training_cat_default.json --dataset_json samples/metadata.jsonl --project_name cyberharem_auto1 --outputs_dir results --train_data_dir samples/CyberHarem # prepare the config file
# results/cyberharem_auto1/cyberharem_auto1.json
python cat_train_all.py --tuning_config_path results/cyberharem_auto1/cyberharem_auto1.json
