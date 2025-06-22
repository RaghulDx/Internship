import json
import requests
import argparse
import os
import re
import pandas as pd
from difflib import SequenceMatcher

class RAGRetriever:
    def __init__(self, examples_file='examples.json', k=4):
        self.examples_file = examples_file
        self.k = k
        self.examples = self._load_examples()

    def _load_examples(self):
        try:
            with open(self.examples_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading examples: {e}")
            return []

    def retrieve(self, user_query):
        def relevance(q1, q2):
            return SequenceMatcher(None, q1.lower(), q2.lower()).ratio()

        sorted_examples = sorted(self.examples, key=lambda ex: relevance(user_query, ex["question"]), reverse=True)
        return sorted_examples[:self.k]

class PromptBuilder:
    def build_prompt(self, user_query, schema_desc, examples):
        formatted_examples = "\n\n".join(
            [f"Q: {ex['question']}\nThought: {ex.get('cot', 'Think step-by-step.')}\nSQL: {ex['sql']}" for ex in examples]
        )
        return f"""
You are an expert in converting natural language questions into accurate SQL queries.
First, think step-by-step about how to interpret the user's question using the schema.
Then, generate a valid SQL query using only the provided schema.

Database Schema:
{schema_desc}

Examples:
{formatted_examples}

Q: {user_query}
Thought:""".strip()

class NLPtoSQLConverter:
    def __init__(self, schema_file, examples_file='examples.json', ollama_url="http://localhost:11434/api/generate"):
        self.schema = self._load_schema(schema_file)
        self.ollama_url = ollama_url
        self.model = "llama3"
        self.rag = RAGRetriever(examples_file)
        self.prompt_builder = PromptBuilder()

    def _load_schema(self, schema_file):
        try:
            with open(schema_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading schema: {e}")
            return None

    def _schema_to_text(self):
        if not self.schema or "tables" not in self.schema:
            return "No schema available."
        desc = []
        for table, info in self.schema["tables"].items():
            desc.append(f"\nTable: {table}\nColumns:")
            for col in info.get("columns", []):
                desc.append(f"- {col}")
            if info.get("primary_keys"):
                desc.append("Primary Key: " + ", ".join(info["primary_keys"]))
            if info.get("foreign_keys"):
                desc.append("Foreign Keys:")
                for fk in info["foreign_keys"]:
                    desc.append(f"- {fk['column']} → {fk['references']['table']}.{fk['references']['column']}")
        return "\n".join(desc)

    def generate_sql(self, nl_query, model=None):
        if model:
            self.model = model
        schema_text = self._schema_to_text()
        retrieved_examples = self.rag.retrieve(nl_query)
        prompt = self.prompt_builder.build_prompt(nl_query, schema_text, retrieved_examples)

        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.model, "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                raw_output = response.json().get("response", "")
                match = re.search(r"SQL:(.*)", raw_output, re.DOTALL)
                return re.sub(r'```sql\s*|\s*```|`', '', match.group(1).strip()) if match else raw_output.strip()
            else:
                print(f"❌ API Error: {response.status_code}\n{response.text}")
        except Exception as e:
            print(f"❌ Request failed: {e}")
        return None

    def test_query(self, sql, csv_file):
        try:
            df = pd.read_csv(csv_file)
            print("CSV Preview:")
            print(df.head())
            print("Generated SQL Query:")
            print(sql)
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema', default='schema_json.json')
    parser.add_argument('--examples', default='examples.json')
    parser.add_argument('--model', default='llama3')
    parser.add_argument('--csv', default='data_test.csv')
    parser.add_argument('--ollama_url', default='http://localhost:11434/api/generate')
    args = parser.parse_args()

    converter = NLPtoSQLConverter(args.schema, args.examples, args.ollama_url)
    print("🧠 CoT + Few-Shot + RAG SQL Generator")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter a natural language query: ")
        if user_query.lower() == "exit":
            break
        sql = converter.generate_sql(user_query, args.model)
        if sql:
            print("\nGenerated SQL:")
            print(sql)
            if input("\nTest this on CSV? (y/n): ").lower() == 'y':
                converter.test_query(sql, args.csv)

if __name__ == "__main__":
    main()
