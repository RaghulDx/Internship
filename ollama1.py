
import json
import requests
import argparse
import os
import re
import pandas as pd


class NLPtoSQLConverter:
    def __init__(self, schema_file_path, ollama_url="http://localhost:11434/api/generate"):
        """
        Initialize the NLP to SQL converter.

        Args:
            schema_file_path (str): Path to the JSON schema file
            ollama_url (str): URL for the Ollama API
        """
        self.ollama_url = ollama_url
        self.schema = self._load_schema(schema_file_path)
        self.model = "llama3"  # Default model, can be changed

    def _load_schema(self, schema_file_path):
        """Load and parse the JSON schema file."""
        try:
            with open(schema_file_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            print(f"Successfully loaded schema from {schema_file_path}")
            return schema
        except Exception as e:
            print(f"Error loading schema: {e}")
            return None

    def _schema_to_readable_format(self):
        """Convert the JSON schema to a readable format for the LLM prompt."""
        if not self.schema or "tables" not in self.schema:
            return "No schema available."

        table_descriptions = []

        for table_name, table_info in self.schema["tables"].items():
            table_descriptions.append(f"\nTable: {table_name}")
            table_descriptions.append("Columns:")
            for column in table_info.get("columns", []):
                table_descriptions.append(f"- {column}")
            table_descriptions.append("Primary Key: " + ", ".join(table_info.get("primary_keys", [])))
            foreign_keys = table_info.get("foreign_keys", [])
            if foreign_keys:
                table_descriptions.append("Foreign Keys:")
                for fk in foreign_keys:
                    table_descriptions.append(
                        f"- {fk['column']} references {fk['references']['table']}.{fk['references']['column']}")

        return "\n".join(table_descriptions)

    def _create_prompt(self, nl_query):
        """Create a prompt for the Ollama model."""
        schema_description = self._schema_to_readable_format()

        prompt = f"""
You are an expert SQL query generator. Convert the following natural language query to a valid SQL query based on the provided database schema. Use only the tables, columns, and relationships defined in the schema. Ensure the query is syntactically correct, efficient, and includes only the conditions explicitly mentioned in the query. When the query mentions 'id' in the context of transport passes, interpret it as the 'id' column from the 'transport_pass' table unless explicitly stated otherwise. Do not add conditions or filters unless they are clearly specified in the natural language query. Do not include explanations, only the SQL query.

Database Schema:
{schema_description}

Example:
Natural Language Query: "Show the id of transport passes"
SQL Query: SELECT id FROM transport_pass;

Natural Language Query: {nl_query}

Generated SQL Query:
"""
        return prompt

    def generate_sql(self, nl_query, model=None):
        """
        Generate SQL from a natural language query using Ollama.

        Args:
            nl_query (str): Natural language query
            model (str, optional): Ollama model to use

        Returns:
            str: Generated SQL query
        """
        if model:
            self.model = model

        prompt = self._create_prompt(nl_query)

        try:
            # Make a request to the Ollama API
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")

                # Remove markdown code blocks, triple/single backticks, and extra whitespace
                sql_query = re.sub(r'```sql\s*|\s*```|`', '', generated_text, flags=re.IGNORECASE)
                sql_query = sql_query.strip()

                return sql_query
            else:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return None

    def test_query(self, sql_query, csv_file_path):
        """
        Test the generated SQL query against a CSV file.

        Args:
            sql_query (str): SQL query to test
            csv_file_path (str): Path to the CSV file

        Returns:
            pd.DataFrame: Result of the query
        """
        try:
            # Load the CSV file as a pandas DataFrame
            df = pd.read_csv(csv_file_path)

            # For demonstration purposes, we'll just print the DataFrame
            print("CSV data:")
            print(df.head())

            print("\nThis is where you would execute the SQL query against your database.")
            print(f"SQL Query: {sql_query}")

            # Note: To actually execute SQL on a DataFrame, you would need to use a package like pandasql
            # For example:
            # from pandasql import sqldf
            # result = sqldf(sql_query, locals())
            # return result

            return df
        except Exception as e:
            print(f"Error testing query: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Convert natural language to SQL using Ollama')
    parser.add_argument('--schema', default='schema_json.json', help='Path to the schema JSON file')
    parser.add_argument('--model', default='llama3', help='Ollama model to use')
    parser.add_argument('--csv', default='data_test.csv', help='Path to CSV file for testing')
    parser.add_argument('--ollama_url', default='http://localhost:11434/api/generate', help='Ollama API URL')

    args = parser.parse_args()

    converter = NLPtoSQLConverter(args.schema, args.ollama_url)

    print("NLP to SQL Converter using Ollama")
    print("Type 'exit' to quit")
    print(f"Using schema: {args.schema}")
    print(f"Using model: {args.model}")

    while True:
        nl_query = input("\nEnter your query in natural language: ")

        if nl_query.lower() == 'exit':
            print("Goodbye!")
            break

        sql_query = converter.generate_sql(nl_query, args.model)

        if sql_query:
            print("\nGenerated SQL query:")
            print(sql_query)

            test_option = input("\nDo you want to test this query against your CSV file? (y/n): ")
            if test_option.lower() == 'y':
                converter.test_query(sql_query, args.csv)


if __name__ == "__main__":
    main()
