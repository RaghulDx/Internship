import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer

def txt_headers_to_json_schema_with_embeddings(txt_file_path, output_file_path="schema_json.json", table_name="default_table"):
    try:
        if not os.path.isfile(txt_file_path):
            raise FileNotFoundError(f"Text file not found: {txt_file_path}")
        if os.path.getsize(txt_file_path) == 0:
            raise ValueError("Text file is empty.")

        df = pd.read_csv(txt_file_path, delimiter='\t')
        headers = df.columns.tolist()
        headers = [header if header != 'Unnamed: 0' else f"Column_{i}" for i, header in enumerate(headers)]

        model = SentenceTransformer('all-MiniLM-L6-v2')

        schema = {
            "table_name": table_name,
            "column_names": [],
            "column_types": [],
            "column_embeddings": []
        }

        for column in headers:
            dtype = str(df[column].dtype)
            column_type = "integer" if dtype == 'int64' else "number" if dtype == 'float64' else "boolean" if dtype == 'bool' else "string"
            embedding = model.encode(column).tolist()

            schema["column_names"].append(column)
            schema["column_types"].append(column_type)
            schema["column_embeddings"].append(embedding)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2)

        print(f"✅ JSON schema with embeddings created and saved at: {output_file_path}")
        return schema

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    txt_file_path = os.path.join(os.getcwd(), "data_test.txt")
    output_file_path = os.path.join(os.getcwd(), "schema_json.json")
    table_name = "my_table"

    schema = txt_headers_to_json_schema_with_embeddings(txt_file_path, output_file_path, table_name)

    if schema:
        print("\nGenerated JSON Schema with Embeddings:")
        print(json.dumps(schema, indent=2))
