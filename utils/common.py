import json


def read_json(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    return data

def read_jsonl(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data