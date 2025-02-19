import csv
import os
import unicodedata
import mysql.connector
from datetime import datetime

# 🔥 Database connection settings
DB_CONFIG = {
    "host": "186.83.138.162",
    "user": "root",
    "password": "root",
    "database": "secop_ii_contratos_electronicos"
}

TABLE_NAME = "secop_ii_contratos_electronicos"

# Define column mappings
COLUMN_MAPPING = {
    "Localización": "Localizacion",
    "Recursos_Propios_(Alcaldias,_Gobernaciones_y_Resguardos_Indigenas)": "Recursos_Propios_Adicionales",
}

# Define which columns contain dates
DATE_COLUMNS = {
    "Fecha_de_Firma",
    "Fecha_Contrato",
    "Fecha_de_Inicio_del_Contrato",
    "Fecha_de_Fin_del_Contrato",
    "Fecha_de_Inicio_de_Ejecucion",
    "Fecha_de_Fin_de_Ejecucion"
}

# Normalize column names
def normalize_column(name):
    if not name:  # Ensure name is not None before processing
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    name = name.replace(" ", "_")
    return COLUMN_MAPPING.get(name, name)

# Convert date format from MM/DD/YYYY → YYYY-MM-DD (for MySQL)
def convert_date_format(date_str):
    try:
        if not date_str or not isinstance(date_str, str):  # Ensure it's a valid string
            return None
        
        date_str = date_str.strip().lower()
        if date_str in ["null", "none", ""]:
            return None

        if "/" in date_str and len(date_str.split("/")) == 3:
            return datetime.strptime(date_str, "%m/%d/%Y").strftime("%Y-%m-%d")
        elif "-" in date_str and len(date_str.split("-")) == 3:
            return date_str  

        return None  # Invalid format
    except ValueError:
        return None  # Return None if the format is incorrect

# Connect to MySQL
def connect_db():
    return mysql.connector.connect(**DB_CONFIG)

# Fetch column size limits from MySQL schema
def get_column_sizes():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute(f"""
        SELECT COLUMN_NAME, CHARACTER_MAXIMUM_LENGTH 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = '{TABLE_NAME}' 
        AND TABLE_SCHEMA = '{DB_CONFIG["database"]}'
    """)
    
    column_sizes = {row["COLUMN_NAME"]: row["CHARACTER_MAXIMUM_LENGTH"] for row in cursor.fetchall()}
    
    cursor.close()
    conn.close()
    return column_sizes

# Insert data from TSV file
def insert_data_from_tsv(tsv_file):
    if not os.path.exists(tsv_file):
        print(f"❌ ERROR: File not found at {tsv_file}")
        return

    conn = connect_db()
    cursor = conn.cursor()

    column_sizes = get_column_sizes()  # Fetch max column sizes

    try:
        with open(tsv_file, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="\t")
            headers = next(reader)
            headers = [normalize_column(col) for col in headers]

            insert_query = f"INSERT INTO {TABLE_NAME} ({', '.join(headers)}) VALUES ({', '.join(['%s'] * len(headers))})"

            count = 0

            for row_num, row in enumerate(reader, start=1):
                try:
                    values = []
                    oversized_fields = []  # Store oversized fields to print later

                    for i, val in enumerate(row):
                        col_name = headers[i]

                        # Ensure val is a string before stripping
                        if val is None:
                            val = ""
                        elif isinstance(val, str):
                            val = val.strip()

                        # Convert date format if necessary
                        if col_name in DATE_COLUMNS:
                            val = convert_date_format(val)

                        # Convert empty values to None
                        if isinstance(val, str) and val.lower() in ["null", "none", ""]:
                            val = None

                        # Check for exceeding max column size
                        if col_name in column_sizes and column_sizes[col_name] is not None:
                            max_size = column_sizes[col_name]
                            if val and isinstance(val, str) and len(val) > max_size:
                                oversized_fields.append((col_name, len(val), max_size, val[:100]))

                        values.append(val)

                    # Print only oversized values
                    if oversized_fields:
                        print(f"\n❌ ERROR at row {row_num}:")
                        for col_name, length, max_size, preview in oversized_fields:
                            print(f"   🔹 Column: {col_name} | Value Length: {length} | Allowed: {max_size}")
                            print(f"   🔹 Value Preview: {preview}...\n")  # Print only first 100 chars

                    cursor.execute(insert_query, values)
                    conn.commit()
                    count += 1

                except mysql.connector.Error as e:
                    print(f"\n❌ MySQL ERROR at row {row_num}: {e}")
                    continue  # Skip this row and continue

    except Exception as e:
        print(f"❌ General ERROR: {e}")

    finally:
        cursor.close()
        conn.close()
        print(f"\n🎉 Insertion Complete! Total Records Inserted: {count}")

# ✅ File path
tsv_file_path = r"C:\Users\angel\Downloads\SECOP.tsv"

# Run
insert_data_from_tsv(tsv_file_path)
