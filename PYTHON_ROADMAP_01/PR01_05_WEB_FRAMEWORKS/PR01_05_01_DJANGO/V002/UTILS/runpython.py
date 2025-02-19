import csv

# Define file paths
input_file = r"C:\Users\angel\Downloads\SECOP_II_-_Contratos_Electr_nicos_20250218.tsv"
output_file = r"C:\Users\angel\Downloads\first_500_records.tsv"

# Read the first 500 records efficiently
with open(input_file, mode="r", encoding="utf-8") as infile:
    reader = csv.reader(infile, delimiter="\t")  # Read TSV format
    header = next(reader)  # Read header row

    # Read first 500 rows
    first_500_records = [header] + [next(reader) for _ in range(500)]

# Write the extracted 500 records to a new TSV file
with open(output_file, mode="w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile, delimiter="\t")
    writer.writerows(first_500_records)

print(f"✅ Extracted 500 records and saved to: {output_file}")
