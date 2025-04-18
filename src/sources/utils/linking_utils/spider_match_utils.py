import re
import string
import collections
import numpy as np
import torch
import nltk.corpus

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNKS = set(a for a in string.punctuation)

CELL_EXACT_MATCH_FLAG = "EXACTMATCH"
CELL_PARTIAL_MATCH_FLAG = "PARTIALMATCH"
COL_PARTIAL_MATCH_FLAG = "CPM"
COL_EXACT_MATCH_FLAG = "CEM"
TAB_PARTIAL_MATCH_FLAG = "TPM"
TAB_EXACT_MATCH_FLAG = "TEM"

# schema linking, similar to IRNet
# Function to compute schema linking between question tokens and column/table names
def compute_schema_linking(question, column, table):
    # Helper function to check for a partial match between two token lists
    def partial_match(x_list, y_list):
        x_str = " ".join(x_list) # Join tokens in x_list into a single string
        y_str = " ".join(y_list) # Join tokens in y_list into a single string
        if x_str in STOPWORDS or x_str in PUNKS: # Ignore stopwords or punctuation
            return False
        if re.match(rf"\b{re.escape(x_str)}\b", y_str): # Regex to check if x_str is a word in y_str
            assert x_str in y_str
            return True
        else:
            return False
    # Helper function to check for an exact match between two token lists
    def exact_match(x_list, y_list):
        x_str = " ".join(x_list) # Join tokens in x_list into a single string
        y_str = " ".join(y_list) # Join tokens in y_list into a single string
        if x_str == y_str: # Return True if both strings are exactly the same
            return True
        else:
            return False
    # Initialize dictionaries to store question-to-column and question-to-table matches
    q_col_match = dict()
    q_tab_match = dict()
    # Create a mapping of column IDs to column names
    col_id2list = dict()
    for col_id, col_item in enumerate(column):
        if col_id == 0: # Skip the first column (usually reserved for the row ID)
            continue
        col_id2list[col_id] = col_item  # Map each column ID to its column name
    # print("-----------------col_id2list-----------------")
    # print(col_id2list)
    """
    {1: ['stadium', 'id'], 2: ['location'], 3: ['name'], 4: ['capacity'], 5: ['high'], 6: ['low'], 7: ['average'], 8: ['singer', 'id'], 9: ['name'], 10: ['country'], 11: ['song', 'name'], 12: ['song', 'release', 'year'], 13: ['age'], 14: ['be', 'male'], 15: ['concert', 'id'], 16: ['concert', 'name'], 17: ['theme'], 18: ['stadium', 'id'], 19: ['year'], 20: ['concert', 'id'], 21: ['singer', 'id']}
    """
    # Create a mapping of table IDs to table names
    tab_id2list = dict()
    for tab_id, tab_item in enumerate(table):
        tab_id2list[tab_id] = tab_item # Map each table ID to its table name
    # print("-----------------tab_id2list-----------------")
    # print(tab_id2list)
    """
    {0: ['stadium'], 1: ['singer'], 2: ['concert'], 3: ['singer', 'in', 'concert']}
    """
    # 5-gram
    # Use 5-gram matching to find matches between question tokens and column/table names

    n = 5
    while n > 0: 
        for i in range(len(question) - n + 1): # Loop through n-grams in the question
            n_gram_list = question[i:i + n]  # Get n tokens from the question as a list # ['how', 'many', 'singer', 'do', 'we']
            n_gram = " ".join(n_gram_list) # Join the n-gram into a single string # how many singer do we
            if len(n_gram.strip()) == 0: # Ignore empty n-grams
                continue
            # exact match case
            # Exact match case: Check if the n-gram exactly matches any column or table name
            for col_id in col_id2list:
                if exact_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        q_col_match[f"{q_id},{col_id}"] = COL_EXACT_MATCH_FLAG # Mark exact column match
            for tab_id in tab_id2list:
                if exact_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        q_tab_match[f"{q_id},{tab_id}"] = TAB_EXACT_MATCH_FLAG # Mark exact table match
            # partial match case
            # Partial match case: Check if the n-gram partially matches any column or table name
            for col_id in col_id2list:
                if partial_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{col_id}" not in q_col_match:
                            q_col_match[f"{q_id},{col_id}"] = COL_PARTIAL_MATCH_FLAG # Mark partial column match
            for tab_id in tab_id2list:
                if partial_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{tab_id}" not in q_tab_match:
                            q_tab_match[f"{q_id},{tab_id}"] = TAB_PARTIAL_MATCH_FLAG # Mark partial table match
            # print("-----------------q_col_match-----------------")
            # print(q_col_match) # {'2,8': 'CPM', '2,21': 'CPM'}
            # print("-----------------q_tab_match-----------------")
            # print(q_tab_match) # {'2,1': 'TEM', '2,3': 'TPM'}
        n -= 1 # Reduce n-gram size and repeat
    # Return dictionaries containing matches between question tokens and columns/tables
    return {"q_col_match": q_col_match, "q_tab_match": q_tab_match}


def compute_cell_value_linking(tokens, schema):
    # Helper function to check if a word is a number
    def isnumber(word):
        try:
            float(word) # Attempt to convert the word to a float
            return True
        except:
            return False
    # Helper function for partial match with a database column
    def db_word_partial_match(word, column, table, db_conn):
        cursor = db_conn.cursor()
        # SQL query to check if `word` partially matches any values in the specified column
        p_str = f"select {column} from {table} where {column} like '{word} %' or {column} like '% {word}' or " \
                f"{column} like '% {word} %' or {column} like '{word}'"
        # print("-----------------p_str-----------------")
        # print(p_str)
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            # print("-----------------p_res-----------------")
            # print(p_res)
            if len(p_res) == 0: # Return results if any match, else False
                return False
            else:
                return p_res
        except Exception as e:
            return False
    # Helper function for exact match with a database column
    def db_word_exact_match(word, column, table, db_conn):
        cursor = db_conn.cursor()
        # SQL query to check if `word` exactly matches any values in the specified column
        p_str = f"select {column} from {table} where {column} like '{word}' or {column} like ' {word}' or " \
                f"{column} like '{word} ' or {column} like ' {word} '"
        # print("-----------------p_str-----------------")
        # print(p_str)
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            # print("-----------------p_res-----------------")
            # print(p_res)
            if len(p_res) == 0: # Return results if any match, else False
                return False
            else:
                return p_res
        except Exception as e:
            return False
    # Initialize dictionaries to store matches
    num_date_match = {} # For number/date matches
    cell_match = {} # For cell value matches
    # Iterate over columns in the schema
    # print("-----------------schema.columns-----------------")
    # print(schema.columns)
    # print("-----------------tokens-----------------")
    # print(tokens) #['how', 'many', 'singer', 'do', 'we', 'have', '?']
    for col_id, column in enumerate(schema.columns):
        if col_id == 0: # Skip the first column (usually a general ID column)
            assert column.orig_name == "*"
            continue
        match_q_ids = [] # Store question token indices that match the column
        # Iterate over tokens in the question
        for q_id, word in enumerate(tokens):
            if len(word.strip()) == 0: 
                continue # Skip empty, stopwords, or punctuation tokens
            if word in STOPWORDS or word in PUNKS:
                continue 
            # Check if the word is a number
            num_flag = isnumber(word)
            if num_flag:    # TODO refine the date and time match
                # If the column is a number or time type, add it to num_date_match
                if column.type in ["number", "time"]:
                    num_date_match[f"{q_id},{col_id}"] = column.type.upper()
            else:
                # For non-numeric tokens, check for a partial match in the database
                # print("-----------------word db_word_partial_match-----------------")
                # print(word, column.orig_name, column.table.orig_name, schema.connection) # many Stadium_ID stadium <sqlite3.Connection object at 0x7e15a3229740>
                ret = db_word_partial_match(word, column.orig_name, column.table.orig_name, schema.connection)
                # print("-----------------ret-----------------")
                # print(ret) #False
                if ret:
                    # print(word, ret)
                    match_q_ids.append(q_id) # Store matched question token indices
        # Group consecutive matching question token indices for exact matching
        # print("-----------------match_q_ids-----------------")
        # print(match_q_ids)
        f = 0
        while f < len(match_q_ids):
            t = f + 1
            while t < len(match_q_ids) and match_q_ids[t] == match_q_ids[t - 1] + 1:
                t += 1
            q_f, q_t = match_q_ids[f], match_q_ids[t - 1] + 1
            # print("-----------------q_f, q_t-----------------")
            # print(q_f, q_t)
            words = [token for token in tokens[q_f: q_t]]
            # print("-----------------words-----------------")
            # print(words)
            # print("-----------------db_word_exact_match-----------------")
            # print(' '.join(words), column.orig_name, column.table.orig_name, schema.connection)
            ret = db_word_exact_match(' '.join(words), column.orig_name, column.table.orig_name, schema.connection)
            # Check for an exact match of the phrase (consecutive tokens) in the database
            if ret:
                for q_id in range(q_f, q_t):
                    cell_match[f"{q_id},{col_id}"] = CELL_EXACT_MATCH_FLAG # Exact match
            else:
                for q_id in range(q_f, q_t):
                    cell_match[f"{q_id},{col_id}"] = CELL_PARTIAL_MATCH_FLAG # Partial match
            f = t

    cv_link = {"num_date_match": num_date_match, "cell_match": cell_match}
    return cv_link

def compute_schema_linking_gpu(question, column, table):
    """
    Compute schema linking using GPU-accelerated similarity calculations.
    Matches question n-grams with column and table names using embeddings.
    
    Args:
        question (list): List of question tokens.
        column (list): List of column names (each a list of tokens).
        table (list): List of table names (each a list of tokens).
        word_emb (GloVe): GloVe object for accessing word embeddings.
        device (str): Device to use for computations ('cuda' or 'cpu').
    
    Returns:
        dict: Dictionary with 'q_col_match' and 'q_tab_match' mappings.
    """
    question_tensor = torch.tensor([hash(token) for token in question], device='cuda')
    q_col_match = {}
    q_tab_match = {}

    col_id2list = {col_id: " ".join(col_item) for col_id, col_item in enumerate(column) if col_id != 0}
    tab_id2list = {tab_id: " ".join(tab_item) for tab_id, tab_item in enumerate(table)}

    def match_tensor(n_gram_tensor, schema_dict):
        matches = []
        for schema_id, schema_str in schema_dict.items():
            schema_tensor = torch.tensor([hash(token) for token in schema_str.split()], device='cuda')
            if torch.equal(n_gram_tensor, schema_tensor):
                matches.append((schema_id, True))
            elif torch.any(torch.isin(n_gram_tensor, schema_tensor)):
                matches.append((schema_id, False))
        return matches

    n = 5
    while n > 0:
        for i in range(len(question) - n + 1):
            n_gram_list = question[i:i + n]
            n_gram_str = " ".join(n_gram_list).strip()
            if len(n_gram_str) == 0 or n_gram_str in STOPWORDS or n_gram_str in PUNKS:
                continue

            n_gram_tensor = question_tensor[i:i + n]

            col_matches = match_tensor(n_gram_tensor, col_id2list)
            for col_id, is_exact in col_matches:
                for q_id in range(i, i + n):
                    key = f"{q_id},{col_id}"
                    if is_exact:
                        q_col_match[key] = COL_EXACT_MATCH_FLAG
                    elif key not in q_col_match:
                        q_col_match[key] = COL_PARTIAL_MATCH_FLAG

            tab_matches = match_tensor(n_gram_tensor, tab_id2list)
            for tab_id, is_exact in tab_matches:
                for q_id in range(i, i + n):
                    key = f"{q_id},{tab_id}"
                    if is_exact:
                        q_tab_match[key] = TAB_EXACT_MATCH_FLAG
                    elif key not in q_tab_match:
                        q_tab_match[key] = TAB_PARTIAL_MATCH_FLAG

        n -= 1

    return {"q_col_match": q_col_match, "q_tab_match": q_tab_match}

def compute_cell_value_linking_gpu(tokens, schema):
    tokens_tensor = torch.tensor([hash(token) for token in tokens], device='cuda')
    num_date_match = {}
    cell_match = {}

    def isnumber(word):
        try:
            float(word)
            return True
        except:
            return False

    def db_word_match(word, column, table, db_conn, exact=True):
        cursor = db_conn.cursor()
        word = word.replace("'", "''")  # Escape single quotes
        like_str = f"{word}" if exact else f"%{word}%"
        query = f"SELECT {column} FROM {table} WHERE {column} LIKE '{like_str}'"
        try:
            cursor.execute(query)
            return cursor.fetchall()
        except:
            return False

    for col_id, column in enumerate(schema.columns):
        if col_id == 0 or column.orig_name == "*":
            continue

        match_q_ids = []

        for q_id, word in enumerate(tokens):
            if word.strip() in STOPWORDS or word.strip() in PUNKS:
                continue

            if isnumber(word):
                if column.type in ["number", "time"]:
                    num_date_match[f"{q_id},{col_id}"] = column.type.upper()
            else:
                partial_res = db_word_match(word, column.orig_name, column.table.orig_name, schema.connection, exact=False)
                if partial_res:
                    match_q_ids.append(q_id)

        f = 0
        while f < len(match_q_ids):
            t = f + 1
            while t < len(match_q_ids) and match_q_ids[t] == match_q_ids[t - 1] + 1:
                t += 1
            q_f, q_t = match_q_ids[f], match_q_ids[t - 1] + 1
            phrase = ' '.join(tokens[q_f:q_t])
            exact_res = db_word_match(phrase, column.orig_name, column.table.orig_name, schema.connection, exact=True)

            for q_id in range(q_f, q_t):
                key = f"{q_id},{col_id}"
                cell_match[key] = CELL_EXACT_MATCH_FLAG if exact_res else CELL_PARTIAL_MATCH_FLAG

            f = t

    return {"num_date_match": num_date_match, "cell_match": cell_match}


def match_shift(q_col_match, q_tab_match, cell_match):

    q_id_to_match = collections.defaultdict(list)
    for match_key in q_col_match.keys():
        q_id = int(match_key.split(',')[0])
        c_id = int(match_key.split(',')[1])
        type = q_col_match[match_key]
        q_id_to_match[q_id].append((type, c_id))
    for match_key in q_tab_match.keys():
        q_id = int(match_key.split(',')[0])
        t_id = int(match_key.split(',')[1])
        type = q_tab_match[match_key]
        q_id_to_match[q_id].append((type, t_id))
    relevant_q_ids = list(q_id_to_match.keys())

    priority = []
    for q_id in q_id_to_match.keys():
        q_id_to_match[q_id] = list(set(q_id_to_match[q_id]))
        priority.append((len(q_id_to_match[q_id]), q_id))
    priority.sort()
    matches = []
    new_q_col_match, new_q_tab_match = dict(), dict()
    for _, q_id in priority:
        if not list(set(matches) & set(q_id_to_match[q_id])):
            exact_matches = []
            for match in q_id_to_match[q_id]:
                if match[0] in [COL_EXACT_MATCH_FLAG, TAB_EXACT_MATCH_FLAG]:
                    exact_matches.append(match)
            if exact_matches:
                res = exact_matches
            else:
                res = q_id_to_match[q_id]
            matches.extend(res)
        else:
            res = list(set(matches) & set(q_id_to_match[q_id]))
        for match in res:
            type, c_t_id = match
            if type in [COL_PARTIAL_MATCH_FLAG, COL_EXACT_MATCH_FLAG]:
                new_q_col_match[f'{q_id},{c_t_id}'] = type
            if type in [TAB_PARTIAL_MATCH_FLAG, TAB_EXACT_MATCH_FLAG]:
                new_q_tab_match[f'{q_id},{c_t_id}'] = type

    new_cell_match = dict()
    for match_key in cell_match.keys():
        q_id = int(match_key.split(',')[0])
        if q_id in relevant_q_ids:
            continue
        # if cell_match[match_key] == CELL_EXACT_MATCH_FLAG:
        new_cell_match[match_key] = cell_match[match_key]

    return new_q_col_match, new_q_tab_match, new_cell_match