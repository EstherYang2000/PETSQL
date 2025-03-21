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
    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]
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
    string = string.replace("\'", "\"")
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1:qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = []
    current_tok = ""
    for char in string:
        if char.isspace():
            if current_tok:
                toks.append(current_tok.lower())
                current_tok = ""
        elif char in "=><!()*,;":
            if current_tok:
                toks.append(current_tok.lower())
                current_tok = ""
            toks.append(char)
        else:
            current_tok += char
    if current_tok:
        toks.append(current_tok.lower())

    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        if eq_idx > 0 and toks[eq_idx-1] in prefix:
            toks = toks[:eq_idx-1] + [toks[eq_idx-1] + "="] + toks[eq_idx+1:]

    print(f"Tokenized: {toks}")
    return toks

def scan_alias(toks):
    alias = {}
    i = 0
    while i < len(toks) - 1:
        if toks[i].lower() == 'as' and i > 0 and i + 1 < len(toks):
            alias[toks[i + 1]] = toks[i - 1]
            i += 2
            continue
        if i > 0 and toks[i-1].lower() in ['from', 'join']:
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
        if key not in tables:
            tables[key] = key
    return tables

def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    tok = toks[start_idx]
    # print(f"Parsing column: '{tok}' at {start_idx}")

    if tok == "*":
        return start_idx + 1, schema.idMap[tok]
    if '.' in tok:
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        return start_idx + 1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables required"
    tok_lower = tok.lower()
    for alias in default_tables:
        table = tables_with_alias[alias]
        for col in schema.schema.get(table, []):
            if tok_lower == col.lower():
                key = table + "." + col
                # print(f"Resolved column: {key}")
                return start_idx + 1, schema.idMap[key]
    assert False, f"Error col: {tok}"

def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
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
        idx += 1
    return idx, (agg_id, col_id, isDistinct)

def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    idx, val = parse_value(toks, idx, tables_with_alias, schema, default_tables)
    if isinstance(val, tuple) and len(val) == 3:
        col_unit1 = val
    else:
        col_unit1 = (AGG_OPS.index("none"), val, False)

    unit_op = UNIT_OPS.index('none')
    col_unit2 = None
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1
    return idx, (unit_op, col_unit1, col_unit2)

def parse_table_unit(toks, start_idx, tables_with_alias, schema):
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
    # print(f"Parsing value at {idx}: {toks[idx:]}")

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif toks[idx] == 'null':
        val = "NULL"
        idx += 1
    elif "\"" in toks[idx]:
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except ValueError:
            idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
            val = col_unit

    if isBlock:
        assert toks[idx] == ')', f"Expected ')' but got {toks[idx]}"
        idx += 1
    # print(f"Value parsed: {val}")
    return idx, val

def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []
    # print(f"Parsing condition at {idx}: {toks[idx:]}")

    while idx < len_:
        if toks[idx] == "(":
            idx += 1
            sub_conds = []
            while idx < len_ and toks[idx] != ")":
                idx, sub_cond = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
                sub_conds.append(sub_cond)
                if idx < len_ and toks[idx] in COND_OPS:
                    sub_conds.append(toks[idx])
                    idx += 1
            assert toks[idx] == ")", f"Expected ')' but got {toks[idx]}"
            idx += 1
            conds.append(sub_conds)
        else:
            idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
            # print(f"Val unit: {val_unit}")

            not_op = False
            if idx < len_ and toks[idx] == "not":
                not_op = True
                idx += 1

            assert idx < len_ and toks[idx] in WHERE_OPS, f"Unexpected token {toks[idx]} in WHERE clause"
            op_id = WHERE_OPS.index(toks[idx])
            idx += 1
            val1 = val2 = None

            if op_id == WHERE_OPS.index("between"):
                idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                assert toks[idx] == "and"
                idx += 1
                idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            elif op_id == WHERE_OPS.index("in"):
                assert toks[idx] == "(", f"Expected '(' but got {toks[idx]}"
                idx += 1
                if toks[idx] == "select":
                    idx, val1 = parse_sql(toks, idx, tables_with_alias, schema)
                else:
                    val_list = []
                    while idx < len_ and toks[idx] != ")":
                        idx, val = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                        val_list.append(val)
                        if toks[idx] == ",":
                            idx += 1
                    assert toks[idx] == ")"
                    idx += 1
                    val1 = val_list
            elif op_id == WHERE_OPS.index("like"):
                idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            elif op_id == WHERE_OPS.index("is"):
                if idx < len_ and toks[idx] == "not":
                    not_op = True
                    idx += 1
                assert idx < len_ and toks[idx] == "null", f"Expected 'null' but got {toks[idx]}"
                val1 = "NULL"
                idx += 1
            else:
                idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)

            conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

    print(f"Conditions: {conds}")
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
        if idx < len_ and toks[idx].lower() == 'as':
            idx += 2
        if idx < len_ and toks[idx] == ',':
            idx += 1
    return idx, (isDistinct, val_units)

def parse_from(toks, start_idx, tables_with_alias, schema):
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
                idx += 1
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'], table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1
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
            idx += 1
        else:
            break
    return idx, col_units

def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc'
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
            idx += 1
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
    try:
        isBlock = False
        len_ = len(toks)
        idx = start_idx
        sql = {}
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        from_start_idx = idx
        while from_start_idx < len_ and toks[from_start_idx].lower() != 'from':
            from_start_idx += 1
        if from_start_idx < len_:
            from_end_idx, table_units, conds, default_tables = parse_from(toks, from_start_idx, tables_with_alias, schema)
        else:
            from_end_idx = len_
            table_units, conds, default_tables = [], [], schema.schema.keys()
        sql['from'] = {'table_units': table_units, 'conds': conds}

        select_idx = idx
        while select_idx < len_ and toks[select_idx].lower() != 'select':
            select_idx += 1
        if select_idx < len_:
            next_idx, select_col_units = parse_select(toks, select_idx, tables_with_alias, schema, default_tables)
            sql['select'] = select_col_units
        else:
            sql['select'] = (False, [])

        idx = from_end_idx
        idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
        sql['where'] = where_conds
        idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
        sql['groupBy'] = group_col_units
        idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
        sql['having'] = having_conds
        idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
        sql['orderBy'] = order_col_units
        idx, limit_val = parse_limit(toks, idx)
        sql['limit'] = limit_val

        idx = skip_semicolon(toks, idx)
        if isBlock:
            if idx >= len_ or toks[idx] != ')':
                raise Exception("Missing closing parenthesis in SQL block")
            idx += 1
        idx = skip_semicolon(toks, idx)

        for op in SQL_OPS:
            sql[op] = None
        if idx < len_ and toks[idx] in SQL_OPS:
            sql_op = toks[idx]
            idx += 1
            idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
            sql[sql_op] = IUE_sql

        return idx, sql
    except Exception as e:
        print(f"❌ Error in parse_sql: {e}")
        raise

def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data

def get_sql(schema, query):
    print(f"🔍 Parsing SQL for query: {query}")
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    print(f"Tables with alias: {tables_with_alias}")
    try:
        _, sql = parse_sql(toks, 0, tables_with_alias, schema)
        return sql
    except Exception as e:
        print(f"❌ Error in get_sql: {e}")
        raise

def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx

if __name__ == "__main__":
    import os
    db_dir = "data/spider/database"
    db = "car_1"
    db_path = os.path.join(db_dir, db, db + ".sqlite")
    schema = Schema(get_schema(db_path))
    p_str = """SELECT T1.Model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id ORDER BY T2.mpg DESC LIMIT 1;"""
    try:
        p_sql = get_sql(schema, p_str)
        # print("Parsed SQL:", json.dumps(p_sql, indent=2))
    except Exception as e:
        print(f"❌ Error: {e}")