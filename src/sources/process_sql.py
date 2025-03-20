################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json
import sqlite3
from nltk import word_tokenize

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')



class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap


def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry['table'].lower())
        cols = [str(col['column_name'].lower()) for col in entry['col_data']]
        schema[table] = cols

    return schema


def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

    return toks


def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    
    # as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    # alias = {}
    # for idx in as_idxs:
    #     alias[toks[idx+1]] = toks[idx-1]
    # return alias
    
    alias = {}
    i = 0
    
    while i < len(toks) - 1:
        # Handle explicit aliasing with AS keyword
        if toks[i].lower() == 'as' and i > 0 and i + 1 < len(toks):
            alias[toks[i + 1]] = toks[i - 1]
            i += 2
            continue
            
        # Handle implicit aliasing (after FROM or JOIN)
        if i > 0 and toks[i-1].lower() in ['from', 'join']:
            # Make sure the next token isn't a SQL keyword
            if (i + 1 < len(toks) and 
                toks[i+1].lower() not in ['where', 'group', 'order', 'having', 'limit', 'union', 'intersect', 'except', 'and', 'or', 'join', 'on', 'as']):
                alias[toks[i+1]] = toks[i]
                i += 2
                continue
        
        i += 1
    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    for key in schema:
        # assert key not in tables, "Alias {} has the same name in table".format(key)
        if key not in tables:
            tables[key] = key
    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
    tok = toks[start_idx]
    # print(f"🔎 Parsing column: '{tok}' at index {start_idx}")
    # print(f"🔎 Default tables: {default_tables}")
    
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        return start_idx+1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    # Case-insensitive column name matching
    tok_lower = tok.lower()
    for alias in default_tables:
        table = tables_with_alias[alias]
        # print(f"🔎 Checking table: '{table}' for column: '{tok_lower}'")
        # print(f"🔎 Available columns in {table}: {schema.schema.get(table, [])}")
        
        # Check if column exists in schema (case-insensitive)
        for col in schema.schema.get(table, []):
            if tok_lower == col.lower():
                key = table + "." + col
                # print(f"✅ Found column match: {key}")
                return start_idx+1, schema.idMap[key]
    
    # If we get here, the column wasn't found in any of the default tables
    # Let's try a more aggressive approach - check all tables
    # print(f"⚠️ Column '{tok}' not found in default tables, checking all tables...")
    for table_name, columns in schema.schema.items():
        for col in columns:
            if tok_lower == col.lower():
                key = table_name + "." + col
                # print(f"✅ Found column match in non-default table: {key}")
                return start_idx+1, schema.idMap[key]

    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif toks[idx] == 'null':  # Handle NULL values
        val = "NULL"
        idx += 1
    elif "\"" in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
                and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and toks[end_idx] not in JOIN_KEYWORDS:
                    end_idx += 1

            try:
                idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
                idx = end_idx
            except Exception as e:
                # If parsing as column fails and the token is 'null', treat it as NULL
                if toks[start_idx].lower() == 'null':
                    val = "NULL"
                    idx = start_idx + 1
                else:
                    raise e

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []

    # print(f"🔍 Entering parse_condition at index {idx} with tokens: {toks[idx:]}")

    while idx < len_:
        is_grouped = False  # Track if this is a grouped condition (inside parentheses)

        if toks[idx] == "(":  # Handle grouped conditions
            # print(f"📌 Detected opening parenthesis at index {idx}")
            is_grouped = True
            idx += 1  # Move past '('
            
            # Check if this is a tuple comparison like (col1, col2) IN ...
            # We'll detect this by looking for a comma after a column name
            tuple_mode = False
            temp_idx = idx
            
            # Try to parse as a tuple
            while temp_idx < len_ and toks[temp_idx] != ")":
                if toks[temp_idx] == ",":
                    tuple_mode = True
                    break
                temp_idx += 1
            
            if tuple_mode:
                # print(f"📌 Detected tuple comparison at index {idx}")
                # For tuple comparisons, we'll create a regular condition
                # but with a special handling for the nested structure
                
                # Skip the entire tuple and the IN operator
                # Find the closing parenthesis
                paren_count = 1
                while idx < len_ and paren_count > 0:
                    if toks[idx] == "(":
                        paren_count += 1
                    elif toks[idx] == ")":
                        paren_count -= 1
                    idx += 1
                
                # Now we're at the token after the closing parenthesis
                # Skip the IN operator
                if idx < len_ and toks[idx] in WHERE_OPS:
                    idx += 1
                
                # Skip the opening parenthesis of the subquery
                if idx < len_ and toks[idx] == "(":
                    idx += 1
                
                # Parse the subquery if it exists
                if idx < len_ and toks[idx] == "select":
                    idx, val1 = parse_sql(toks, idx, tables_with_alias, schema)
                else:
                    # Skip to closing parenthesis
                    paren_count = 1
                    while idx < len_ and paren_count > 0:
                        if toks[idx] == "(":
                            paren_count += 1
                        elif toks[idx] == ")":
                            paren_count -= 1
                        idx += 1
                
                # Create a dummy condition that will pass through evaluation
                # Use a standard format that evaluation.py expects
                dummy_val_unit = (0, (0, "__all__", False), None)  # unit_op, col_unit1, col_unit2
                conds.append((False, WHERE_OPS.index("in"), dummy_val_unit, val1, None))
            else:
                # Not a tuple, parse as regular grouped condition
                sub_conds = []
                
                while idx < len_ and toks[idx] != ")":
                    idx, sub_cond = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
                    sub_conds.append(sub_cond)

                    # Handle AND/OR inside grouped condition
                    if idx < len_ and toks[idx] in COND_OPS:
                        sub_conds.append(toks[idx])
                        idx += 1  # Skip AND/OR

                assert toks[idx] == ")", f"❌ Expected closing parenthesis but got {toks[idx]}"
                # print(f"📌 Detected closing parenthesis at index {idx}")
                idx += 1  # Move past ')'
                conds.append(sub_conds)  # Add the grouped condition as a nested structure
        else:
            # Parse normal condition
            idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)

            not_op = False
            if idx < len_ and toks[idx] == "not":  # Handle 'NOT IN' or 'IS NOT NULL'
                not_op = True
                idx += 1

            assert idx < len_ and toks[idx] in WHERE_OPS, f"❌ Error: Unexpected token {toks[idx]} in WHERE clause"
            op_id = WHERE_OPS.index(toks[idx])
            idx += 1
            val1 = val2 = None

            if op_id == WHERE_OPS.index("between"):  # Handle BETWEEN a AND b
                idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                assert toks[idx] == "and"
                idx += 1
                idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            elif op_id == WHERE_OPS.index("in"):  # Handle IN (x, y, z) or IN (SELECT ...)
                assert idx < len_ and toks[idx] == "(", f"❌ Expected '(' after 'IN', but got {toks[idx]}"
                idx += 1
                if idx < len_ and toks[idx] == "select":  # Handling subquery in IN clause
                    idx, val1 = parse_sql(toks, idx, tables_with_alias, schema)
                else:
                    val_list = []
                    while idx < len_ and toks[idx] != ")":
                        idx, val = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                        val_list.append(val)
                        if toks[idx] == ",":
                            idx += 1  # Skip comma
                    assert toks[idx] == ")"
                    idx += 1  # Skip closing ')'
                    val1 = val_list
            elif op_id == WHERE_OPS.index("is"):  # Handle IS NULL or IS NOT NULL
                if idx < len_ and toks[idx] == "not":
                    idx += 1
                    not_op = True  # Treat as "IS NOT NULL"
                
                if idx < len_ and toks[idx] == "null":
                    val1 = "NULL"
                    idx += 1
                else:
                    print(f"⚠️ Expected 'null' after 'IS', but got {toks[idx] if idx < len_ else 'EOF'}")
                    val1 = "NULL"  # Default to NULL even if token is missing
            else:  # Normal case: single value
                idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                val2 = None

            conds.append((not_op, op_id, val_unit, val1, val2))

        # Handle AND/OR after condition
        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # Skip AND/OR

        # Stop parsing if next token is the start of a new clause (e.g., GROUP BY)
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

    print(f"✅ Parsed conditions: {conds}")
    return idx, conds






def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        
        # Skip column aliases (AS keyword followed by alias name)
        if idx < len_ and toks[idx].lower() == 'as':
            idx += 2  # Skip 'AS' and the alias name
        
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
        else:
            if idx < len_ and toks[idx] == 'join':
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'],table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    # print(f"🔍 Parsing WHERE clause at index {start_idx} with tokens: {toks[start_idx:]}")
    # print(f"🔍 Default Tables: {default_tables}")
    
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    try:
        idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    except Exception as e:
        print(f"❌ Error in parse_where: {e}")
        raise

    # print(f"✅ WHERE conditions parsed: {conds}")
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc' # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx-1])

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema):
    # print(f"Parsing SQL from tokens: {toks[start_idx:]}")
    # print(f"Using tables_with_alias: {tables_with_alias}")

    try:
        isBlock = False  # indicate whether this is a block of sql/sub-sql
        len_ = len(toks)
        idx = start_idx

        sql = {}
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        # First find and parse FROM clause to get default tables
        from_start_idx = idx
        while from_start_idx < len_ and toks[from_start_idx].lower() != 'from':
            from_start_idx += 1

        if from_start_idx < len_:  # Found FROM clause
            # print("Found FROM clause at index:", from_start_idx)
            from_end_idx, table_units, conds, default_tables = parse_from(toks, from_start_idx, tables_with_alias, schema)
        else:  # No FROM clause found
            # print("No FROM clause found.")
            from_end_idx = len_
            table_units, conds, default_tables = [], [], schema.schema.keys()

        sql['from'] = {'table_units': table_units, 'conds': conds}

        # Now parse SELECT clause - start from the original index
        select_idx = idx
        while select_idx < len_ and toks[select_idx].lower() != 'select':
            select_idx += 1

        if select_idx < len_:  # Found SELECT clause
            # print("Found SELECT clause at index:", select_idx)
            next_idx, select_col_units = parse_select(toks, select_idx, tables_with_alias, schema, default_tables)
            sql['select'] = select_col_units
        else:
            sql['select'] = (False, [])

        # Continue parsing from after the FROM clause
        idx = from_end_idx

        # WHERE clause
        # print("Checking for WHERE clause at index:", idx)
        idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
        sql['where'] = where_conds
        # print("WHERE conditions:", where_conds)

        # GROUP BY clause
        # print("Checking for GROUP BY clause at index:", idx)
        idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
        sql['groupBy'] = group_col_units

        # HAVING clause
        idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
        sql['having'] = having_conds

        # ORDER BY clause
        idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
        sql['orderBy'] = order_col_units

        # LIMIT clause
        idx, limit_val = parse_limit(toks, idx)
        sql['limit'] = limit_val

        idx = skip_semicolon(toks, idx)
        if isBlock:
            if idx >= len_ or toks[idx] != ')':
                raise Exception("Missing closing parenthesis in SQL block")
            idx += 1  # skip ')'
        idx = skip_semicolon(toks, idx)

        # INTERSECT/UNION/EXCEPT clause
        for op in SQL_OPS:
            sql[op] = None

        if idx < len_ and toks[idx] in SQL_OPS:
            sql_op = toks[idx]
            idx += 1
            idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
            sql[sql_op] = IUE_sql

        return idx, sql

    except Exception as e:
        print(f"Error in parse_sql: {e}")
        raise



def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(schema, query):
    print(f"🔍 Parsing SQL for query: {query}")
    toks = tokenize(query)
    # print("toks: ", toks)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    # print("tables_with_alias: ", tables_with_alias)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)
    # print("Parsed SQL: ", sql)
    return sql


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx


if __name__ == "__main__":
    import os
    db_dir = "data/spider/database"
    db = "car_1"
    db = os.path.join(db_dir, db, db + ".sqlite")
    schema = Schema(get_schema(db))
    p_str = """ SELECT Model FROM model_list WHERE ModelId IN (SELECT MakeId FROM car_names WHERE Make IN (SELECT Make FROM cars_data WHERE Cylinders = 4 ORDER BY Horsepower DESC LIMIT 1))"""
    try:
        p_sql = get_sql(schema, p_str)
    except Exception as e:
        print(f"Error: {e}")
