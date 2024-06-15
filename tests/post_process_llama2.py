import json

with open("/home/ylu130/workspace/REV-reimpl/Yining/generated_rationales/strategyqa/Llama-2-7b-hf_demo=2_raw=True/test.jsonl", "r") as f:
    data = [json.loads(line) for line in f.readlines()]
for item in data:
    item['facts'] = [item['facts'][0].split("\nquestion:")[0]]

with open("/home/ylu130/workspace/REV-reimpl/Yining/generated_rationales/strategyqa/Llama-2-7b-hf_demo=2_raw=True/test.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")