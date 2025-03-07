import sqlite3
import argparse

def check_table_exists(cursor, table_name):
    """Check if a table exists in the database."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    return cursor.fetchone() is not None

def find_encoding_errors(cursor):
    """Find all rows in 'players' table where last_name contains an encoding error (�)."""
    cursor.execute("SELECT player_id, first_name, last_name FROM players WHERE last_name LIKE '%�%';")
    return cursor.fetchall()

def update_last_name(cursor, conn, player_id, new_last_name):
    """Update last_name for a specific player_id."""
    cursor.execute("UPDATE players SET last_name = ? WHERE player_id = ?;", (new_last_name, player_id))
    conn.commit()

def main(db_path, fix_all=False):
    """Main function to fix encoding errors in SQLite."""
    try:
        # Connect to SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if 'players' table exists
        if not check_table_exists(cursor, 'players'):
            print("[ERROR] Table 'players' does not exist in the database.")
            return

        # Find problematic rows
        errors = find_encoding_errors(cursor)
        if not errors:
            print("[INFO] No encoding errors found in 'players' table.")
            return
        
        print(f"[INFO] Found {len(errors)} encoding issues.")
        
        for player_id, first_name, last_name in errors:
            print(f"[ERROR] Player ID: {player_id}, Name: {first_name} {last_name}")

            # Suggest replacement
            new_last_name = last_name.replace("�", "n")  # Replace � with 'n' (or modify as needed)
            
            if fix_all:
                update_last_name(cursor, conn, player_id, new_last_name)
                print(f"[FIXED] Updated '{last_name}' → '{new_last_name}'")
            else:
                # Ask user before updating
                user_input = input(f"Replace '{last_name}' with '{new_last_name}'? (y/n/custom): ").strip().lower()
                if user_input == "y":
                    update_last_name(cursor, conn, player_id, new_last_name)
                    print(f"[UPDATED] Player ID {player_id}: '{last_name}' → '{new_last_name}'")
                elif user_input != "n" and user_input != "":
                    update_last_name(cursor, conn, player_id, user_input)
                    print(f"[CUSTOM UPDATED] Player ID {player_id}: '{last_name}' → '{user_input}'")

        conn.close()
        print("[INFO] Database update complete.")

    except sqlite3.Error as e:
        print(f"[ERROR] SQLite Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix encoding issues in SQLite database.")
    parser.add_argument("--db", required=True, help="Path to SQLite database file.")
    parser.add_argument("--fix_all", action="store_true", help="Automatically fix all encoding errors.")

    args = parser.parse_args()
    db_path = args.db
    auto_fix=True
            # Connect to SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor() 
    cursor.execute("SELECT first_name || ' ' || last_name AS full_name FROM players ORDER BY birth_date;")
    rows = cursor.fetchall()
    print(rows)
    #         # Check if 'players' table exists
    # if not check_table_exists(cursor, 'players'):
    #     print("[ERROR] Table 'players' does not exist in the database.")    
    # # cursor.execute("SELECT player_id, first_name, last_name, hex(last_name) FROM players WHERE hex(last_name) LIKE hex('Treyes%');")
    # # 直接用BLOB讀first_name和last_name，繞過UTF-8 decode
    # cursor.execute("SELECT player_id, hex(first_name), hex(last_name) FROM players;")
    # rows = cursor.fetchall()

    # problematic = []

    # for player_id, hex_first, hex_last in rows:
    #     try:
    #         first_name = bytes.fromhex(hex_first).decode('utf-8')
    #         last_name = bytes.fromhex(hex_last).decode('utf-8')
    #     except UnicodeDecodeError:
    #         problematic.append((player_id, hex_first, hex_last))

    # print(f"Found {len(problematic)} problematic records.")

    # for player_id, hex_first, hex_last in problematic:
    #     print(f"\n[!] Problematic player_id={player_id}")
    #     try:
    #         first_name = bytes.fromhex(hex_first).decode('utf-8', errors='replace')
    #         last_name = bytes.fromhex(hex_last).decode('utf-8', errors='replace')
    #     except Exception as e:
    #         print(f"[ERROR] Cannot decode hex data for player_id={player_id}: {e}")
    #         continue

    #     print(f"Original First Name (with errors): {first_name}")
    #     print(f"Original Last Name (with errors): {last_name}")

    #     if auto_fix:
    #         # 自動替換所有�成'n' (你可以自訂替代策略)
    #         fixed_first = first_name.replace('�', 'n')
    #         fixed_last = last_name.replace('�', 'n')

    #         cursor.execute("UPDATE players SET first_name = ?, last_name = ? WHERE player_id = ?;",
    #                        (fixed_first, fixed_last, player_id))

    #         print(f"[FIXED] player_id={player_id} first_name={fixed_first}, last_name={fixed_last}")

    #     else:
    #         new_first = input(f"Manually input corrected first_name for player_id={player_id} [{first_name}]: ").strip() or first_name
    #         new_last = input(f"Manually input corrected last_name for player_id={player_id} [{last_name}]: ").strip() or last_name

    #         cursor.execute("UPDATE players SET first_name = ?, last_name = ? WHERE player_id = ?;",
    #                        (new_first, new_last, player_id))
    #         print(f"[UPDATED] player_id={player_id} → first_name={new_first}, last_name={new_last}")

    conn.commit()
    conn.close()
    print("[INFO] Encoding issues fixed.")
    
