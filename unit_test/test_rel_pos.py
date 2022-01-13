"""args.max_ast_size 512"""
import argparse

from pymongo import MongoClient
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tree_sitter import Parser, Language

from _utils import get_ud2pos
from data_process import ast2json
from parser import tree_to_token_index, index_to_code_token, DFG_python, remove_comments_and_docstrings
from utils import convert_example_to_func_naming_feature, Example

"""args.max_dfg_len 64"""
"""args.max_code_len 256"""
"""args.use_ast"""
"""args.use_code"""
"""args.use_dfg"""
"""args.max_rel_pos = 64 如果code 与 ast 同时使用， 扩为128"""
"""args.max_target_len = 7"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_ast_size", type=int, default=32)
    parser.add_argument("--max_dfg_len", type=int, default=8)
    parser.add_argument("--max_code_len", type=int, default=16)
    parser.add_argument("--use_ast", action='store_true')
    parser.add_argument("--use_code", action='store_true')
    parser.add_argument("--use_dfg", action='store_true')
    parser.add_argument("--max_rel_pos", type=int, default=64)
    parser.add_argument("--max_target_len", type=int, default=7)

    args = parser.parse_args()

    tokenizer = Tokenizer.from_file('./tokenizer/roberta_tokenizer.json')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                        unk_token='<unk>',
                                        bos_token="<s>",
                                        eos_token="</s>",
                                        cls_token='<s>',
                                        sep_token='</s>',
                                        pad_token='<pad>',
                                        mask_token='<mask>')

    # example, example_index, tokenizer, args, stage = item
    client = MongoClient('172.29.7.221', 27017, username='admin', password='123456')
    codes = client.code_search_net.codes
    query = codes.find({'partition': 'train', 'lang': 'python'})[17]

    code = "def get(my, he):\n" \
           "  my = my + 1\n" \
           "  he = myName + he\n" \
           "  print(my+he)\n" \
           ""

    code = remove_comments_and_docstrings(code, 'python')

    bytecode = bytes(code, 'utf8')

    LANGUAGE = Language('parser/my-languages.so', 'python')
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, DFG_python]

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

    print(dfg)

    # idx, ast, dfg, target, index2code, lang
    example = Example(0, ast_json, dfg,
                      'get_and_set_use_name', index_to_code, 'java')
    item = (example, 0, tokenizer, args, 'train')
    feature = convert_example_to_func_naming_feature(item)
    #print(feature.source_ids)
    # print(feature.rel_pos)
    ud2pos = get_ud2pos(args.max_rel_pos)
    pos2ud = {v:k for k,v in ud2pos.items()}
    for k in feature.rel_pos.keys():
        if k[0] == 14:
            v = feature.rel_pos[k]
            if v in pos2ud:
                print('(', k[0], ',', k[1], ')->', pos2ud[v])
            else:
                print('(', k[0], ',', k[1], ')->', v-32-65)