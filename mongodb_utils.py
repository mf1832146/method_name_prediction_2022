import pickle

from pymongo import MongoClient
import gzip
from tqdm import tqdm
import json

from utils import Example


def connect_db():
    client = MongoClient('172.29.7.221', 27017, username='admin', password='123456')
    db = client.code_search_net
    return db


# read data from db
def find_all_by_tag(collection, tag):
    return collection.find({'partition': tag})


# idx, ast, dfg, target, index2code, lang
def read_fuc_name_pre_examples_from_db(collection, split_tag, lang, data_num):
    """Read examples from mongodb with conditions."""
    print('load data from db.')
    return_items = {'code_index': 1, 'ast': 1, 'func_name': 1, 'dfg': 1, 'index_to_code': 1}
    conditions = {'partition': split_tag, 'lang': lang}

    examples = []
    results = collection.find(conditions, return_items)
    for result in tqdm(results):
        idx = result['code_index']
        ast = pickle.loads(result['ast'])
        dfg = pickle.loads(result['dfg'])
        index2code = pickle.loads(result['index_to_code'])
        func_name = result['func_name']
        examples.append(
            Example(
                idx=idx,
                ast=ast,
                dfg=dfg,
                index2code=index2code,
                target=func_name,
                lang=lang
            )
        )
        if idx + 1 == data_num:
            break
    return examples
