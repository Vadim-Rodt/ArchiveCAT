import os

def count_code_lines_in_functions_folder():
    folder = 'functions'
    total_lines = 0

    if not os.path.isdir(folder):
        print(f"Ordner '{folder}' nicht gefunden.")
        return

    for filename in os.listdir(folder):
        if filename.endswith('.py'):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = [line for line in file if line.strip()]  # nicht-leere Zeilen
                print(f"{filename}: {len(lines)} Zeilen")
                total_lines += len(lines)

    print(f"\nGesamtzahl der Codezeilen in allen .py-Dateien im Ordner '{folder}': {total_lines}")

if __name__ == "__main__":
    count_code_lines_in_functions_folder()
