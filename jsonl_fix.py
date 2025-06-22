# jsonl_to_json.py
import json

input_path = "examples.json"  # Assuming it's actually jsonl
output_path = "examples_fixed.json"

with open(input_path, "r", encoding="utf-8") as infile:
    data = [json.loads(line) for line in infile]

with open(output_path, "w", encoding="utf-8") as outfile:
    json.dump(data, outfile, indent=2)

print(f"✅ Converted to valid JSON array at: {output_path}")
