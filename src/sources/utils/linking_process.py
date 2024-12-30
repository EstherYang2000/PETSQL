import collections
import itertools
import json
import os

import attr
import numpy as np
import torch

from utils.linking_utils import abstract_preproc, corenlp, serialization
from utils.linking_utils.spider_match_utils import (
    compute_schema_linking,
    compute_cell_value_linking,
    match_shift
)

@attr.s
class PreprocessedSchema:
    column_names = attr.ib(factory=list)
    table_names = attr.ib(factory=list)
    table_bounds = attr.ib(factory=list)
    column_to_table = attr.ib(factory=dict)
    table_to_columns = attr.ib(factory=dict)
    foreign_keys = attr.ib(factory=dict)
    foreign_keys_tables = attr.ib(factory=lambda: collections.defaultdict(set))
    primary_keys = attr.ib(factory=list)

    # only for bert version
    normalized_column_names = attr.ib(factory=list)
    normalized_table_names = attr.ib(factory=list)


def preprocess_schema_uncached(schema,
                               tokenize_func,
                               include_table_name_in_column,
                               fix_issue_16_primary_keys,
                               bert=False):
    """If it's bert, we also cache the normalized version of
    question/column/table for schema linking"""
    r = PreprocessedSchema()

    if bert: assert not include_table_name_in_column

    last_table_id = None
    for i, column in enumerate(schema.columns):
        col_toks = tokenize_func(
            column.name, column.unsplit_name)

        # assert column.type in ["text", "number", "time", "boolean", "others"]
        type_tok = f'<type: {column.type}>'
        if bert:
            # for bert, we take the representation of the first word
            column_name = col_toks + [type_tok]
            r.normalized_column_names.append(Bertokens(col_toks))
        else:
            column_name = [type_tok] + col_toks

        if include_table_name_in_column:
            if column.table is None:
                table_name = ['<any-table>']
            else:
                table_name = tokenize_func(
                    column.table.name, column.table.unsplit_name)
            column_name += ['<table-sep>'] + table_name
        r.column_names.append(column_name)

        table_id = None if column.table is None else column.table.id
        r.column_to_table[str(i)] = table_id
        if table_id is not None:
            columns = r.table_to_columns.setdefault(str(table_id), [])
            columns.append(i)
        if last_table_id != table_id:
            r.table_bounds.append(i)
            last_table_id = table_id

        if column.foreign_key_for is not None:
            r.foreign_keys[str(column.id)] = column.foreign_key_for.id
            r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)

    r.table_bounds.append(len(schema.columns))
    assert len(r.table_bounds) == len(schema.tables) + 1

    for i, table in enumerate(schema.tables):
        table_toks = tokenize_func(
            table.name, table.unsplit_name)
        r.table_names.append(table_toks)
        if bert:
            r.normalized_table_names.append(Bertokens(table_toks))
    last_table = schema.tables[-1]

    r.foreign_keys_tables = serialization.to_dict_with_sorted_values(r.foreign_keys_tables)
    r.primary_keys = [
        column.id
        for table in schema.tables
        for column in table.primary_keys
    ] if fix_issue_16_primary_keys else [
        column.id
        for column in last_table.primary_keys
        for table in schema.tables
    ]
    """
    PreprocessedSchema(
    column_names=[['<type: text>', '*'], ['<type: number>', 'stadium', 'id'], ['<type: text>', 'location'], ['<type: text>', 'name'], 
    ['<type: number>', 'capacity'], ['<type: number>', 'high'], ['<type: number>', 'low'], ['<type: number>', 'average'], ['<type: number>', 'singer', 'id'], 
    ['<type: text>', 'name'], ['<type: text>', 'country'], ['<type: text>', 'song', 'name'], ['<type: text>', 'song', 'release', 'year'], ['<type: number>', 'age'], 
    ['<type: others>', 'be', 'male'], ['<type: number>', 'concert', 'id'], ['<type: text>', 'concert', 'name'], ['<type: text>', 'theme'], ['<type: text>', 'stadium', 'id'], 
    ['<type: text>', 'year'], ['<type: number>', 'concert', 'id'], ['<type: text>', 'singer', 'id']], 
    table_names=[['stadium'], ['singer'], ['concert'], ['singer', 'in', 'concert']], 
    table_bounds=[1, 8, 15, 20, 22], column_to_table={'0': None, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 1, '9': 1, '10': 1, '11': 1, '12': 1, '13': 1, '14': 1, '15': 2, '16': 2, '17': 2, '18': 2, '19': 2, '20': 3, '21': 3}, 
    table_to_columns={'0': [1, 2, 3, 4, 5, 6, 7], '1': [8, 9, 10, 11, 12, 13, 14], '2': [15, 16, 17, 18, 19], '3': [20, 21]}, foreign_keys={'18': 1, '20': 15, '21': 8}, 
    foreign_keys_tables={'2': [0], '3': [1, 2]}, primary_keys=[1, 8, 15, 20], normalized_column_names=[], normalized_table_names=[])
    """
    return r


class SpiderEncoderV2Preproc(abstract_preproc.AbstractPreproc):

    def __init__(
            self,
            save_path,
            min_freq=3,
            max_count=5000,
            include_table_name_in_column=True,
            word_emb=None,
            # count_tokens_in_word_emb_for_vocab=False,
            fix_issue_16_primary_keys=False,
            compute_sc_link=False,
            compute_cv_link=False):
        if word_emb is None:
            self.word_emb = None
        else:
            self.word_emb = word_emb

        self.data_dir = os.path.join(save_path, 'enc')
        self.include_table_name_in_column = include_table_name_in_column
        # self.count_tokens_in_word_emb_for_vocab = count_tokens_in_word_emb_for_vocab
        self.fix_issue_16_primary_keys = fix_issue_16_primary_keys
        self.compute_sc_link = compute_sc_link
        self.compute_cv_link = compute_cv_link
        self.texts = collections.defaultdict(list)
        # self.db_path = db_path

        # self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        # self.vocab_path = os.path.join(save_path, 'enc_vocab.json')
        # self.vocab_word_freq_path = os.path.join(save_path, 'enc_word_freq.json')
        # self.vocab = None
        # self.counted_db_ids = set()
        self.preprocessed_schemas = {}

    def validate_item(self, item, schema, section):
        return True, None

    def add_item(self, item, schema, section, validation_info):
        preprocessed = self.preprocess_item(item, schema, validation_info)
        print("----------------------preprocessed----------------------")
        print(preprocessed)
        self.texts[section].append(preprocessed)

    def clear_items(self):
        self.texts = collections.defaultdict(list)

    def preprocess_item(self, item, schema, validation_info):
        # Tokenize the question for processing and copying
        question, question_for_copying = self._tokenize_for_copying(item['question_toks'], item['question'])
        # print("----------------------question----------------------")
        # print(question) #['how', 'many', 'singer', 'do', 'we', 'have', '?']
        # print("----------------------question_for_copying----------------------")
        # print(question_for_copying) # ['how', 'many', 'singers', 'do', 'we', 'have', '?']
        # Preprocess the schema to extract table and column names, foreign keys, and other metadata
        preproc_schema = self._preprocess_schema(schema)
        # print("----------------------preproc_schema----------------------")
        # print(preproc_schema)
        # Compute schema linking (sc_link) if specified
        # Schema linking matches the question tokens with table and column names in the schema
        if self.compute_sc_link:
            # Ensure the column names have type information (starts with "<type:")
            assert preproc_schema.column_names[0][0].startswith("<type:")
            # Remove type information from column names to simplify matching
            column_names_without_types = [col[1:] for col in preproc_schema.column_names]
            # print("----------------------column_names_without_types----------------------")
            # print(column_names_without_types)
            # Compute schema linking between the question and column/table names
            print("----------------------compute_schema_linking----------------------")
            sc_link = compute_schema_linking(question, column_names_without_types, preproc_schema.table_names)
            # print("----------------------sc_link----------------------")
            # print(sc_link) # {'q_col_match': {'2,8': 'CPM', '2,21': 'CPM'}, 'q_tab_match': {'2,1': 'TEM', '2,3': 'TPM'}}
            
        else:
            # If schema linking computation is disabled, use empty placeholders
            sc_link = {"q_col_match": {}, "q_tab_match": {}}
        # Compute cell value linking (cv_link) if specified
        # Cell value linking matches the question tokens with specific cell values in the database
        if self.compute_cv_link:
            print("----------------------compute_cell_value_linking----------------------")
            # print(question) # ['how', 'many', 'singer', 'do', 'we', 'have', '?']
            cv_link = compute_cell_value_linking(question, schema)
        else:
            # If cell value linking computation is disabled, use empty placeholders
            cv_link = {"num_date_match": {}, "cell_match": {}}
        # # Return a dictionary containing the preprocessed question and schema information
        return {
            'raw_question': item['question'],                # Original question text
            'db_id': schema.db_id,                           # Database ID
            'question': question,                            # Tokenized question for processing
            'question_for_copying': question_for_copying,    # Tokenized question for copying

            # Schema linking (sc_link) results showing matches between question tokens and schema elements
            'sc_link': sc_link,

            # Cell value linking (cv_link) results showing matches between question tokens and cell values
            'cv_link': cv_link,

            # Schema metadata processed and simplified for downstream tasks
            'columns': preproc_schema.column_names,          # Column names in the schema
            'tables': preproc_schema.table_names,            # Table names in the schema
            'table_bounds': preproc_schema.table_bounds,     # Boundaries of tables within columns
            'column_to_table': preproc_schema.column_to_table, # Mapping from columns to their respective tables
            'table_to_columns': preproc_schema.table_to_columns, # Mapping from tables to their respective columns
            'foreign_keys': preproc_schema.foreign_keys,     # Foreign key relationships
            'foreign_keys_tables': preproc_schema.foreign_keys_tables, # Tables involved in foreign keys
            'primary_keys': preproc_schema.primary_keys,     # Primary key columns
        }
        
        return {}

    def _preprocess_schema(self, schema):
        if schema.db_id in self.preprocessed_schemas:
            return self.preprocessed_schemas[schema.db_id]
        print("----------------------preprocess_schema_uncached----------------------")
        print(self.include_table_name_in_column) # False
        print(self.fix_issue_16_primary_keys) # True
        result = preprocess_schema_uncached(schema, self._tokenize,
                                            self.include_table_name_in_column, self.fix_issue_16_primary_keys)
        self.preprocessed_schemas[schema.db_id] = result
        return result

    def _tokenize(self, presplit, unsplit):
        if self.word_emb:
            return self.word_emb.tokenize(unsplit)
        return presplit

    def _tokenize_for_copying(self, presplit, unsplit):
        if self.word_emb:
            return self.word_emb.tokenize_for_copying(unsplit)
        return presplit, presplit

    def save(self):
        # os.makedirs(self.data_dir, exist_ok=True)
        # self.vocab = self.vocab_builder.finish()
        # print(f"{len(self.vocab)} words in vocab")
        # self.vocab.save(self.vocab_path)
        # self.vocab_builder.save(self.vocab_word_freq_path)

        for section, texts in self.texts.items():
            with open(section + '_schema-linking.jsonl', 'w') as f:
                for text in texts:
                    f.write(json.dumps(text) + '\n')

    def load(self, sections):
        # self.vocab = vocab.Vocab.load(self.vocab_path)
        # self.vocab_builder.load(self.vocab_word_freq_path)
        for section in sections:
            self.texts[section] = []
            with open(os.path.join(self.data_dir, section + '_schema-linking.jsonl'), 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        self.texts[section].append(json.loads(line))

    def dataset(self, section):
        return [
            json.loads(line)
            for line in open(os.path.join(self.data_dir, section + '.jsonl'))]

