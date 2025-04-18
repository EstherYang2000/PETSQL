import json

json_file = "bird/bird/dev.json"
output_file = "bird/bird/dev_new.sql"
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
# Write each SQL query to the output file
with open(output_file, 'w', encoding='utf-8') as f_out:
    for entry in data:
        idx = entry["query"]
        f_out.write(idx + ";\n")  # Add a semicolon and newline for each query

print(f"SQL queries have been written to {output_file}")
 
    