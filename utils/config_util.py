import json
import jsonlines
import os

# /share0/DEEPTEXT_LAB/CAT/CAT/configs/multi_training_cat_default.json

def read_json_file(json_file):
    # if jsonl file, read as jsonl and get list
    if json_file.endswith(".jsonl"):
        with jsonlines.open(json_file, 'r') as reader:
            data = list(reader)
        return data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json_file(json_file, data):
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def collect_tokens(dataset_json):
    tokens = []
    for dicts in dataset_json:
        for key, value in dicts.items():
            if key == "text":
                tokens.append(value.split(", ")[0])
    return set(tokens)

def create_new_json_file(json_file, dataset_json, project_name:str, outputs_dir:str):
    json_data = read_json_file(json_file)
    json_data.update({"output_dir" : os.path.abspath(os.path.join(outputs_dir, project_name))})
    os.makedirs(json_data["output_dir"], exist_ok=True)
    # train_data_dir -> parent of dataset_json
    json_data.update({"from_jsonl" : os.path.abspath(dataset_json)})
    # read validation prompt
    validation_prompts = json_data["validation_prompt"] # list
    dataset_json = read_json_file(dataset_json)
    tokens = collect_tokens(dataset_json)
    combined_prompts = []
    for i, prompt in enumerate(validation_prompts):
        for token in tokens:
            combined_prompts.append(f"{token}, {prompt}")
            # without token
            combined_prompts.append(prompt)
    print(f"Combined prompts: {len(combined_prompts)}, {len(tokens)} x {len(validation_prompts)}")
    json_data.update({"validation_prompt" : combined_prompts})
    # write to new json file
    new_json_file = os.path.join(json_data["output_dir"], f"{project_name}.json")
    write_json_file(new_json_file, json_data)
    print(f"New json file created: {new_json_file}")
    return new_json_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create new json file for training')
    parser.add_argument('--json_file', type=str, required=True, help='Path to the json file')
    parser.add_argument('--dataset_json', type=str, required=True, help='Path to the dataset json file')
    parser.add_argument('--project_name', type=str, required=True, help='Project name')
    parser.add_argument('--outputs_dir', type=str, required=True, help='Outputs directory')
    args = parser.parse_args()
    create_new_json_file(args.json_file, args.dataset_json, args.project_name, args.outputs_dir)
