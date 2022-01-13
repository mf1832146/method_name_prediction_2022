from parser import DFG_python, DFG_java,DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from pymongo.errors import DocumentTooLarge
from tqdm import tqdm
from pymongo import MongoClient
from tree_sitter import Language, Parser
import pickle
import sys

sys.setrecursionlimit(1000000)

"""
数据库处理数据步骤
  1. 从mongodb读取数据
  2. 从每个数据中读取出original_string字段, 生成AST
     2.1 如果生成失败，添加字段build_ast : false
  3. 生成数据流图
  4. 生成最短路径以及兄弟关系矩阵
  5. 写回
"""

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def process():
    """ 1. read data from db """
    # 1.1 connect db
    client = MongoClient('172.29.7.221', 27017, username='admin', password='123456')
    db = client.code_search_net.codes
    # 1.2 read all data
    return_items = {'_id': 1,  'original_string': 1, 'code_tokens': 1, 'lang': 1}
    total_num = db.find({'lang': 'ruby'}).count()
    results = db.find({'lang': 'ruby'}, return_items)
    too_large_errors = {}
    for item in tqdm(results, desc='build ast and dfg.', total=total_num):
        try:
            code = remove_comments_and_docstrings(item['original_string'], item['lang'])
            if lang == "php":
                code = "<?php" + code + "?>"
            bytecode = bytes(code, 'utf8')
            tree = parser[0].parse(bytecode)
            root_node = tree.root_node
            ast_json = ast2json(root_node)

            # obtain dfg
            tokens_index = tree_to_token_index(root_node)
            code = code.split('\n')
            code_tokens = [index_to_code_token(x, code) for x in tokens_index]
            index_to_code = {}
            for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
                index_to_code[index] = (idx, code)
            try:
                dfg, _ = parser[1](root_node, index_to_code, {})
            except:
                dfg = []
            dfg = sorted(dfg, key=lambda x: x[1])
            indexs = set()
            for d in dfg:
                if len(d[-1]) != 0:
                    indexs.add(d[1])
                for x in d[-1]:
                    indexs.add(x)
            new_dfg = []
            for d in dfg:
                if d[1] in indexs:
                    new_dfg.append(d)
            dfg = new_dfg

            """先暂时保存index_to_code"""
            add_values = {
                'code_tokens': code_tokens,
                'dfg': pickle.dumps(dfg),
                'index_to_code': pickle.dumps(index_to_code),
                'ast': pickle.dumps(ast_json),
                'build_ast': 1
            }
        except Exception as e:
            print(e)
            add_values = {
                'build_ast': -1
            }

        try:
            db.update_one({'_id': item['_id']}, {'$set': add_values})
        except DocumentTooLarge as e:
            print('skip large documents.')
            if item['lang'] in too_large_errors:
                too_large_errors[item['lang']] += 1
            else:
                too_large_errors[item['lang']] = 1
            db.update_one({'_id': item['_id']}, {'$set': {'build_ast': -1}})

    print('Too large errors: ', too_large_errors)


def get_span(node, bytecode) -> str:
    return str(bytecode[node.start_byte:node.end_byte], "utf8")


def ast2json(node):
    return {
        'type': node.type,
        'position_id': (node.start_point, node.end_point),
        'children': [ast2json(child) for child in node.children]
    }


def build_shortest_path():
    return


def build_sibling_path():
    return


if __name__ == '__main__':
    process()