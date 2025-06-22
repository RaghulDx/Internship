import json
import requests
import argparse
import os
import re
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


class RAGPromptBuilder:
    def __init__(self, examples_file="examples.json", top_k=3):
        self.top_k = top_k
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.examples = []
        self.questions = []

        # Static few-shot examples (always included)
        self.static_examples = [
  {
    "question": "Get all transport passes where the gross weight is more than 5000.",
    "sql": "SELECT id, vehicle_number, gross_weight FROM transport_pass WHERE gross_weight > 5000;"
  },
  {
    "question": "Get the vehicle number and the name of the receiving entity for each transport pass.",
    "sql": "SELECT tp.vehicle_number, em.entity_name FROM transport_pass tp JOIN entity_master_aud em ON tp.to_entity_audit_id = em.audit_id AND tp.to_entity_code = em.entity_code;"
  },
  {
    "question": "Get the total number of transport passes issued by each entity.",
    "sql": "SELECT em.entity_name, COUNT(tp.id) AS total_passes FROM transport_pass tp JOIN entity_master_aud em ON tp.from_entity_audit_id = em.audit_id AND tp.from_entity_code = em.entity_code GROUP BY em.entity_name;"
  },
  {
    "question": "List transport passes that transported products from at least 3 different brands.",
    "sql": "SELECT tp.id, tp.vehicle_number FROM transport_pass tp JOIN transport_pass_details tpd ON tp.id = tpd.transport_pass_id JOIN product_spec_details psd ON tpd.product_spec_details_id = psd.id GROUP BY tp.id, tp.vehicle_number HAVING COUNT(DISTINCT psd.brand_master_name) >= 3;"
  },
  {
    "question": "Show all vehicle numbers in the system.",
    "sql": "SELECT vehicle_number FROM transport_pass;"
  },
  {
    "question": "Get the average retail price for each liquor type.",
    "sql": "SELECT psd.liquor_type_name, AVG(psd.retail_price) AS avg_retail_price FROM product_spec_details psd GROUP BY psd.liquor_type_name;"
  },
  {
    "question": "Get all transport passes sorted by gross weight in descending order.",
    "sql": "SELECT id, vehicle_number, gross_weight FROM transport_pass ORDER BY gross_weight DESC;"
  },
  {
    "question": "List all entities that have never received a transport pass.",
    "sql": "SELECT em.entity_name FROM entity_master_aud em WHERE NOT EXISTS (SELECT 1 FROM transport_pass tp WHERE tp.to_entity_audit_id = em.audit_id AND tp.to_entity_code = em.entity_code);"
  },
  {
    "question": "Get the total gross weight transported per vehicle.",
    "sql": "SELECT vehicle_number, SUM(gross_weight) AS total_weight FROM transport_pass GROUP BY vehicle_number;"
  },
  {
    "question": "Get all transport passes created in the year 2023.",
    "sql": "SELECT * FROM transport_pass WHERE YEAR(created_date) = 2023;"
  },
            {
                "question": "Find all transport passes carrying products from more than 3 brands.",
                "cot": "Join transport_pass with transport_pass_details and product_spec_details to access brand information. Group by transport_pass.id and vehicle_number, count distinct brands, and filter where count > 3.",
                "sql": "SELECT tp.id, tp.vehicle_number FROM transport_pass tp JOIN transport_pass_details tpd ON tp.id = tpd.transport_pass_id JOIN product_spec_details psd ON tpd.product_spec_details_id = psd.id GROUP BY tp.id, tp.vehicle_number HAVING COUNT(DISTINCT psd.brand_master_name) > 3;"
            },
            {
                "question": "Which vehicles have transported liquor from more than 3 different entities in a single week?",
                "cot": "Join transport_pass with entity_master_aud using from_entity_code and from_entity_audit_id. Group by vehicle_number and week of created_date. Count distinct entity_name, filter where count > 3.",
                "sql": "SELECT tp.vehicle_number, COUNT(DISTINCT em.entity_name) AS num_entities FROM transport_pass tp INNER JOIN transport_pass_details tpd ON tp.id = tpd.transport_pass_id INNER JOIN product_spec_details psd ON tpd.product_spec_details_id = psd.id INNER JOIN entity_master_aud em ON tp.from_entity_audit_id = em.audit_id AND tp.from_entity_code = em.entity_code WHERE DATE_PART('week', tp.created_date) = DATE_PART('week', CURRENT_DATE) GROUP BY tp.vehicle_number HAVING COUNT(DISTINCT em.entity_name) > 3;"
            },
            {
                "question": "Get the total weight transported by each vehicle for each week.",
                "cot": "Group transport_pass by vehicle_number and the week of created_date. Use SUM on gross_weight to get total weight transported.",
                "sql": "SELECT tp.vehicle_number, DATE_PART('week', tp.created_date) AS week_number, SUM(tp.gross_weight) AS total_weight FROM transport_pass tp GROUP BY tp.vehicle_number, DATE_PART('week', tp.created_date);"
            },
            {
                "question": "List all entities that only sent but never received transport passes.",
                "cot": "Join transport_pass with entity_master_aud using from_entity_code and audit_id. Then filter those entities where no record exists in transport_pass where they appear as to_entity_code.",
                "sql": "SELECT DISTINCT em.entity_name FROM entity_master_aud em JOIN transport_pass tp ON em.audit_id = tp.from_entity_audit_id AND em.entity_code = tp.from_entity_code WHERE NOT EXISTS (SELECT 1 FROM transport_pass tp2 WHERE tp2.to_entity_audit_id = em.audit_id AND tp2.to_entity_code = em.entity_code);"
            },
            {
                "question": "Find vehicles reused after a gap of more than 30 days.",
                "cot": "Self-join transport_pass on vehicle_number. Look for rows where the same vehicle appears again after 30 days. Filter cases where the difference in created_date is more than 30.",
                "sql": "SELECT t1.vehicle_number FROM transport_pass t1 JOIN transport_pass t2 ON t1.vehicle_number = t2.vehicle_number AND t2.created_date > t1.created_date WHERE DATEDIFF(t2.created_date, t1.created_date) > 30;"
            },
            {
                "question": "Which products have the highest average retail price per liquor type?",
                "cot": "Join product_spec_details with transport_pass_details. Group by liquor_type_name and calculate AVG(retail_price). Use ORDER BY to get the highest values.",
                "sql": "SELECT psd.liquor_type_name, AVG(psd.retail_price) AS avg_price FROM product_spec_details psd JOIN transport_pass_details tpd ON psd.id = tpd.product_spec_details_id GROUP BY psd.liquor_type_name ORDER BY avg_price DESC;"
            },
            {
                "question": "Find all transport passes issued in 2023 with total gross weight above 10000.",
                "cot": "Filter transport_pass records where year of created_date is 2023. Group by transport_pass.id and sum gross_weight. Filter where total weight > 10000.",
                "sql": "SELECT tp.id, SUM(tp.gross_weight) AS total_weight FROM transport_pass tp WHERE EXTRACT(YEAR FROM tp.created_date) = 2023 GROUP BY tp.id HAVING SUM(tp.gross_weight) > 10000;"
            },
            {
                "question": "Get the number of unique liquor types transported each month.",
                "cot": "Join transport_pass with details and spec. Extract month from created_date. Group by month and count distinct liquor_type_code.",
                "sql": "SELECT EXTRACT(MONTH FROM tp.created_date) AS month, COUNT(DISTINCT psd.liquor_type_code) AS liquor_types FROM transport_pass tp JOIN transport_pass_details tpd ON tp.id = tpd.transport_pass_id JOIN product_spec_details psd ON psd.id = tpd.product_spec_details_id GROUP BY EXTRACT(MONTH FROM tp.created_date);"
            },
            {
                "question": "Which entities received the most transport passes last quarter?",
                "cot": "Filter transport_pass by created_date in last quarter. Group by to_entity_code. Count transport passes and order by count descending.",
                "sql": "SELECT tp.to_entity_code, COUNT(*) AS pass_count FROM transport_pass tp WHERE tp.created_date >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months') GROUP BY tp.to_entity_code ORDER BY pass_count DESC;"
            },
            {
                "question": "Find all transport passes that included both 'BEER' and 'WHISKY' in the same pass.",
                "cot": "Use EXISTS subqueries on product_spec_details via transport_pass_details to check if a single transport_pass_id has both liquor types.",
                "sql": "SELECT tp.id, tp.vehicle_number FROM transport_pass tp WHERE EXISTS (SELECT 1 FROM transport_pass_details tpd JOIN product_spec_details psd ON tpd.product_spec_details_id = psd.id WHERE tpd.transport_pass_id = tp.id AND psd.liquor_type_code = 'BEER') AND EXISTS (SELECT 1 FROM transport_pass_details tpd JOIN product_spec_details psd ON tpd.product_spec_details_id = psd.id WHERE tpd.transport_pass_id = tp.id AND psd.liquor_type_code = 'WHISKY');"
            },
            {
                "question": "Which vehicles have been used by more than 2 different entities in the last 30 days?",
                "cot": "Join transport_pass with entity_master_aud to get entity info. Filter for created_date in the last 30 days. Group by vehicle_number and count distinct entity_code, filter > 2.",
                "sql": "SELECT tp.vehicle_number FROM transport_pass tp JOIN entity_master_aud em ON tp.from_entity_code = em.entity_code AND tp.from_entity_audit_id = em.audit_id WHERE tp.created_date >= DATE('now', '-30 days') GROUP BY tp.vehicle_number HAVING COUNT(DISTINCT em.entity_code) > 2;"
            },
            {
                "question": "How many transport passes were issued every month in the current year?",
                "cot": "Filter transport_pass by YEAR(created_date) = current year. Extract month, group by it and count rows.",
                "sql": "SELECT EXTRACT(MONTH FROM created_date) AS month, COUNT(*) AS total_passes FROM transport_pass WHERE EXTRACT(YEAR FROM created_date) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY EXTRACT(MONTH FROM created_date);"
            },
            {
                "question": "List transport passes where the same vehicle was used more than once in a week.",
                "cot": "Group by vehicle_number and week number of created_date. Count how many times each appears. Use HAVING > 1.",
                "sql": "SELECT vehicle_number, EXTRACT(WEEK FROM created_date) AS week_num, COUNT(*) AS usage_count FROM transport_pass GROUP BY vehicle_number, EXTRACT(WEEK FROM created_date) HAVING COUNT(*) > 1;"
            },
            {
                "question": "Find products that were transported by both Entity A and Entity B.",
                "cot": "Find product_spec_details_id from transport_pass_details where transport_pass came from Entity A and from Entity B. Use INTERSECT or HAVING COUNT(DISTINCT entity) = 2.",
                "sql": "SELECT tpd.product_spec_details_id FROM transport_pass tp JOIN transport_pass_details tpd ON tp.id = tpd.transport_pass_id WHERE tp.from_entity_code IN ('A', 'B') GROUP BY tpd.product_spec_details_id HAVING COUNT(DISTINCT tp.from_entity_code) = 2;"
            },
            {
                "question": "Which transport passes carried both 'BEER' and 'WHISKY'?",
                "cot": "Use EXISTS subqueries to check if same transport_pass_id has both liquor_type_code = 'BEER' and 'WHISKY'.",
                "sql": "SELECT tp.id FROM transport_pass tp WHERE EXISTS (SELECT 1 FROM transport_pass_details tpd JOIN product_spec_details psd ON tpd.product_spec_details_id = psd.id WHERE tp.id = tpd.transport_pass_id AND psd.liquor_type_code = 'BEER') AND EXISTS (SELECT 1 FROM transport_pass_details tpd JOIN product_spec_details psd ON tpd.product_spec_details_id = psd.id WHERE tp.id = tpd.transport_pass_id AND psd.liquor_type_code = 'WHISKY');"
            },
            {
                "question": "What is the average retail price for each liquor type transported in 2023?",
                "cot": "Join product_spec_details and transport_pass_details. Filter for YEAR(created_date)=2023. Group by liquor_type_name and use AVG.",
                "sql": "SELECT psd.liquor_type_name, AVG(psd.retail_price) AS avg_price FROM product_spec_details psd JOIN transport_pass_details tpd ON psd.id = tpd.product_spec_details_id WHERE EXTRACT(YEAR FROM tpd.created_date) = 2023 GROUP BY psd.liquor_type_name;"
            },
            {
                "question": "Top 5 entities with highest number of transport passes sent.",
                "cot": "Join transport_pass with entity_master_aud using from_entity_code. Group by entity_name and count. Order by count DESC and limit to 5.",
                "sql": "SELECT em.entity_name, COUNT(tp.id) AS total_sent FROM transport_pass tp JOIN entity_master_aud em ON tp.from_entity_code = em.entity_code AND tp.from_entity_audit_id = em.audit_id GROUP BY em.entity_name ORDER BY total_sent DESC LIMIT 5;"
            }

]

        # Load full example set
        if os.path.exists(examples_file):
            with open(examples_file, "r", encoding="utf-8") as f:
                self.examples = json.load(f)
                self.questions = [ex["question"] for ex in self.examples]
            self._build_index()
        else:
            print(f"⚠️ Examples file {examples_file} not found. Retrieval will be skipped.")

    def _build_index(self):
        embeddings = self.model.encode(self.questions, show_progress_bar=False)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, user_query):
        if not self.index:
            return []
        q_emb = self.model.encode([user_query])
        _, idxs = self.index.search(q_emb, self.top_k)
        return [self.examples[i] for i in idxs[0]]

    def build_prompt(self, user_query, schema_desc):
        retrieved = self.retrieve(user_query)

        static_text = "\n".join(
            [f"Natural Language Query: {ex['question']}\nSQL Query: {ex['sql']}" for ex in self.static_examples]
        )
        dynamic_text = "\n".join(
            [f"Natural Language Query: {ex['question']}\nSQL Query: {ex['sql']}" for ex in retrieved]
        )

        rules = """
    Rules:
    You are an expert system for converting natural language into SQL queries. Follow these rules strictly:

Think Before You SQL

Always provide a step-by-step reasoning ("Thought") before writing the SQL.

Interpret what the user is asking based on database relationships.

Respect Schema Only

Use only tables and columns provided in the schema.

Never invent column or table names.

Use Aliases Clearly

Use table aliases (tp, tpd, psd, em, etc.) for readability if multiple joins are involved.

Be Precise with Joins

Use INNER JOIN, LEFT JOIN, etc., based on intent.

Use proper foreign key relationships from the schema to join tables.

Use Functions Carefully

Use COUNT(), SUM(), AVG() correctly, only on numeric fields.

Use EXTRACT(YEAR FROM date) or DATE_PART('week', column) appropriately based on SQL dialect.

Group and Filter Thoughtfully

When using GROUP BY, always pair with HAVING for filtered aggregates.

Don't use HAVING without a GROUP BY.

Format the Output

Use consistent indentation and column selection order (e.g., always select id or vehicle_number first if relevant).

No DDL or DML

Do not generate INSERT, UPDATE, DELETE, or CREATE TABLE statements — only SELECT queries.

Output Clean SQL Only

Final output must start with SELECT and end with a ;.

No explanations, comments, or markdown formatting in the final SQL.

Ignore Missing Info

If required details (e.g., filter value) are missing, assume a placeholder or add a sensible default like WHERE 1=1.
    """

        return f"""
    You are an expert SQL query generator. Convert the natural language query to a valid SQL query using only the provided database schema.

Database Schema:
{schema_desc}

Static Examples:
{static_text}

Relevant Examples:
{dynamic_text}

Natural Language Query: {user_query}

Generated SQL Query:
""".strip()


class NLPtoSQLConverter:
    def __init__(self, schema_file, examples_file="examples.json", ollama_url="http://localhost:11434/api/generate"):
        self.schema = self._load_schema(schema_file)
        self.ollama_url = ollama_url
        self.model = "llama3"
        self.rag = RAGPromptBuilder(examples_file)

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
        prompt = self.rag.build_prompt(nl_query, schema_text)

        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.model, "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                output = response.json().get("response", "")
                return re.sub(r'```sql\s*|\s*```|`', '', output).strip()
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
    print("🧠 Hybrid Prompting: Static Few-Shot + RAG")
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
