import json
import requests
import argparse
import os
import re
import pandas as pd

class PromptBuilder:
    def __init__(self):
        self.static_examples = [
            {
                "question": "Get all transport passes where the gross weight is more than 5000.",
                "cot": "We are asked to filter transport_pass rows where the gross_weight exceeds 5000.",
                "sql": "SELECT id, vehicle_number, gross_weight FROM transport_pass WHERE gross_weight > 5000;"
            },
            {
                "question": "List the names of entities located in the 'North' district.",
                "cot": "We need to search the entity_master_aud table for rows where district_name is 'North' and select entity_name.",
                "sql": "SELECT entity_name FROM entity_master_aud WHERE district_name = 'North';"
            },
            {
                "question": "Get the total number of transport passes issued by each entity.",
                "cot": "Join transport_pass with entity_master_aud using from_entity_code and from_entity_audit_id. Group by entity_name and count the transport_pass ids.",
                "sql": "SELECT em.entity_name, COUNT(tp.id) AS total_passes FROM transport_pass tp JOIN entity_master_aud em ON tp.from_entity_audit_id = em.audit_id AND tp.from_entity_code = em.entity_code GROUP BY em.entity_name;"
            },
            {
                "question": "Find transport passes with products from at least 3 different brands.",
                "cot": "Join transport_pass with transport_pass_details and then product_spec_details. Group by transport_pass.id and count distinct brands. Filter where count is >= 3.",
                "sql": "SELECT tp.id, tp.vehicle_number FROM transport_pass tp JOIN transport_pass_details tpd ON tp.id = tpd.transport_pass_id JOIN product_spec_details psd ON tpd.product_spec_details_id = psd.id GROUP BY tp.id, tp.vehicle_number HAVING COUNT(DISTINCT psd.brand_master_name) >= 3;"
            },
            {
                "question": "List vehicles used by more than one entity as the sender.",
                "cot": "Group transport_pass by vehicle_number. Count distinct from_entity_code values and filter where count > 1.",
                "sql": "SELECT tp.vehicle_number FROM transport_pass tp JOIN entity_master_aud em ON tp.from_entity_code = em.entity_code GROUP BY tp.vehicle_number HAVING COUNT(DISTINCT em.entity_code) > 1;"
            },
            {
                "question": "List all unique liquor types transported in 2023.",
                "cot": "Join transport_pass → transport_pass_details → product_spec_details. Filter transport_pass by year 2023 and select distinct liquor_type_name.",
                "sql": "SELECT DISTINCT psd.liquor_type_name FROM transport_pass tp JOIN transport_pass_details tpd ON tp.id = tpd.transport_pass_id JOIN product_spec_details psd ON tpd.product_spec_details_id = psd.id WHERE YEAR(tp.created_date) = 2023;"
            },
            {
                "question": "Get the average retail price per liquor type for 2023.",
                "cot": "Join product_spec_details with transport_pass_details. Filter by YEAR(created_date) = 2023. Group by liquor_type_name and take AVG of retail_price.",
                "sql": "SELECT psd.liquor_type_name, AVG(psd.retail_price) AS avg_price FROM product_spec_details psd JOIN transport_pass_details tpd ON psd.id = tpd.product_spec_details_id WHERE YEAR(tpd.created_date) = 2023 GROUP BY psd.liquor_type_name;"
            },
            {
                "question": "Find vehicles reused after a gap of more than 30 days.",
                "cot": "Self-join transport_pass on vehicle_number. Look for alias2.created_date > alias1.created_date and DATEDIFF > 30. Ensure no intermediate transport_pass exists in that range.",
                "sql": "SELECT alias1.vehicle_number FROM transport_pass alias1 JOIN transport_pass alias2 ON alias1.vehicle_number = alias2.vehicle_number AND alias1.created_date < alias2.created_date AND DATEDIFF(alias2.created_date, alias1.created_date) > 30 WHERE NOT EXISTS (SELECT 1 FROM transport_pass p3 WHERE p3.vehicle_number = alias1.vehicle_number AND p3.created_date BETWEEN alias1.created_date AND alias2.created_date) ORDER BY alias1.vehicle_number;"
            },
            {
                "question": "List all vehicle numbers and their total transported weight.",
                "cot": "Group by vehicle_number and use SUM(gross_weight) to get total transported weight per vehicle.",
                "sql": "SELECT vehicle_number, SUM(gross_weight) AS total_weight FROM transport_pass GROUP BY vehicle_number;"
            },
            {
                "question": "Find all transport passes carrying products from more than 3 brands.",
                "cot": "Join transport_pass → transport_pass_details → product_spec_details. Group by transport_pass.id. Count distinct brand_master_name and filter > 3.",
                "sql": "SELECT tp.id, COUNT(DISTINCT psd.brand_master_name) AS brand_count FROM transport_pass tp JOIN transport_pass_details tpd ON tp.id = tpd.transport_pass_id JOIN product_spec_details psd ON tpd.product_spec_details_id = psd.id GROUP BY tp.id HAVING COUNT(DISTINCT psd.brand_master_name) > 3;"
            }
        ]

    def build_prompt(self, user_query, schema_desc):
        examples = "\n\n".join(
            [f"Q: {ex['question']}\nThought: {ex['cot']}\nSQL: {ex['sql']}" for ex in self.static_examples]
        )

        return f"""
You are an expert in converting natural language questions into accurate SQL queries.
First, think step-by-step about how to interpret the user's question using the schema.
Then, generate a valid SQL query using only the provided schema.

Database Schema:
{schema_desc}

Examples:
{examples}

Q: {user_query}
Thought:""".strip()


class NLPtoSQLConverter:
    def __init__(self, schema_file, ollama_url="http://localhost:11434/api/generate"):
        self.schema = self._load_schema(schema_file)
        self.ollama_url = ollama_url
        self.model = "llama3"
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
        prompt = self.prompt_builder.build_prompt(nl_query, schema_text)

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
    parser.add_argument('--model', default='llama3')
    parser.add_argument('--csv', default='data_test.csv')
    parser.add_argument('--ollama_url', default='http://localhost:11434/api/generate')
    args = parser.parse_args()

    converter = NLPtoSQLConverter(args.schema, args.ollama_url)
    print("🧠 Chain-of-Thought + Few-Shot SQL Generator")
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
