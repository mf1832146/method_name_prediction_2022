import multiprocessing

from torch.utils.data import Dataset
import numpy as np
import logging
import os
import torch
from tqdm import tqdm
import time
import sys
import pickle
from pymongo import MongoClient
from _utils import ast2seq, get_ud2pos, build_relative_position, tokenize_with_camel_case

sys.setrecursionlimit(1000000)
logger = logging.getLogger(__name__)


def connect_db():
    client = MongoClient('172.29.7.221', 27017, username='admin', password='123456')
    return client.code_search_net


db = connect_db()


# idx, ast, dfg, target, index2code, lang
def read_fuc_name_pre_examples_from_db(collection, split_tag, lang, data_num):
    """Read examples from mongodb with conditions."""
    print('load data from db.')
    return_items = {'code_index': 1, 'ast': 1, 'func_name': 1, 'dfg': 1, 'index_to_code': 1}
    conditions = {'partition': split_tag, 'lang': lang, 'build_ast': 1}

    examples = []
    results = collection.find(conditions, return_items)
    for result in tqdm(results, total=results.count()):
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


def load_and_cache_gen_data_from_db(args, pool, tokenizer, split_tag):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)

    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + data_tag)

    db_name = 'tmp_' + split_tag + '_' + str(time.time())

    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        logger.info("Create cache data into %s", cache_fn)

        codes = connect_db().codes
        # collection, split_tag, lang, data_num
        examples = read_fuc_name_pre_examples_from_db(codes, split_tag, args.sub_task, args.data_num)
        tuple_examples = [(example, idx, tokenizer, args, split_tag, db_name) for idx, example in enumerate(examples)]
        features = []
        for tuple_example in tqdm(tuple_examples, total=len(tuple_examples)):
            features.append(convert_example_to_func_naming_feature(tuple_example))
        data = FuncNamingDataset(features, db_name, args, tokenizer)
        if args.local_rank in [-1, 0]:
            torch.save(data, cache_fn)
    return data


class Example(object):
    def __init__(self, idx, ast, dfg, target, index2code, lang):
        self.idx = idx
        self.target = target
        self.ast = ast
        self.dfg = dfg
        self.index2code = index2code
        self.lang = lang


class FuncNamingFeature(object):
    def __init__(self, example_id, source_ids, position_idx, rel_pos, source_mask, target_ids, target_mask, gold_ids):
        self.example_id = example_id
        self.source_ids = source_ids
        self.position_idx = position_idx
        self.rel_pos = rel_pos
        self.source_mask = source_mask
        self.target_ids = target_ids
        self.gold_ids = gold_ids
        self.target_mask = target_mask
        self.gold_ids = gold_ids


class FuncNamingDataset(Dataset):
    def __init__(self, examples, db_name, args, tokenizer):
        self.examples = examples
        self.args = args
        self.db_name = db_name
        self.tokenizer = tokenizer
        self.db = None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if self.db is None:
            self.db = connect_db()[self.db_name]
        result = self.db.find({'example_index': item})[0]
        example = pickle.loads(result['example'])

        max_source_len = self.args.max_source_len
        rel_pos = np.zeros((max_source_len, max_source_len), dtype=np.long)
        for k, v in example.rel_pos.items():
            if k[0] < max_source_len and k[1] < max_source_len:
                rel_pos[k[0]][k[1]] = v
        attn_mask = rel_pos > 0
        return (torch.tensor(example.source_ids),
                torch.tensor(example.source_mask),
                torch.tensor(example.position_idx),
                torch.tensor(attn_mask),
                torch.tensor(rel_pos),
                torch.tensor(example.target_ids),
                torch.tensor(example.target_mask),
                torch.tensor(example.gold_ids))


def convert_example_to_func_naming_feature(item):
    example, example_index, tokenizer, args, stage, db_name = item

    ast = example.ast
    dfg = example.dfg
    func_name = example.target
    index2code = example.index2code

    """mask func name"""
    for k, v in index2code:
        if '.' in func_name:
            func_name = func_name.split('.')[-1]
        if v[1] == func_name:
            index2code[k] = (v[0], '<mask>')

    ud2pos = get_ud2pos(args.max_rel_pos)

    non_leaf_tokens, leaf_tokens, ud_mask = ast2seq(ast, index2code, ud2pos, args)

    # leaf tokens 分词
    split_leaf_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                         enumerate(leaf_tokens)]

    # code token 对应的位置
    ori2cur_pos = {-1: (0, 0)}
    for i in range(len(split_leaf_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(split_leaf_tokens[i]))
    split_leaf_tokens = [y for x in split_leaf_tokens for y in x]

    """truncating"""
    if args.use_ast:
        max_source_len = args.max_ast_len - 2
        if len(split_leaf_tokens) > max_source_len:
            split_leaf_tokens = split_leaf_tokens[:max_source_len]
            origin_non_leaf_len = len(non_leaf_tokens)
            non_leaf_tokens = []
        else:
            origin_non_leaf_len = len(non_leaf_tokens)
            non_leaf_tokens = non_leaf_tokens[:max_source_len-len(split_leaf_tokens)]
    else:
        max_source_len = args.max_code_len - 2
        split_leaf_tokens = split_leaf_tokens[:max_source_len]

    input_tokens = non_leaf_tokens + split_leaf_tokens if args.use_ast else split_leaf_tokens

    source_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

    rel_pos = {}

    if args.use_ast:
        # 将leaf node的位置映射成切分后的位置
        non_leaf_len = len(non_leaf_tokens)
        length = len([tokenizer.cls_token])
        for k, v in ud_mask.items():
            if k[0] >= non_leaf_len or k[1] >= non_leaf_len:
                # 叶节点已拆分，需要映射
                if k[0] >= non_leaf_len:
                    if k[0] < origin_non_leaf_len:  # 已被截断 忽略
                        continue
                    region = ori2cur_pos[k[0]-origin_non_leaf_len]
                    start_pos = range(region[0]+non_leaf_len+length, region[1]+non_leaf_len+length)
                else:
                    start_pos = [k[0]+length]
                if k[1] >= non_leaf_len:
                    if k[1] < origin_non_leaf_len:  # 已被截断 忽略
                        continue
                    region = ori2cur_pos[k[1]-origin_non_leaf_len]
                    end_pos = range(region[0]+non_leaf_len+length, region[1]+non_leaf_len+length)
                else:
                    end_pos = [k[1]+length]
                for s in start_pos:
                    for e in end_pos:
                        if s < max_source_len and e < max_source_len:  # 应对截断
                            rel_pos[(s, e)] = v
            else:
                rel_pos[(k[0]+length, k[1]+length)] = v

    if args.use_code:
        rel_leaf_pos = build_relative_position(len(split_leaf_tokens), len(split_leaf_tokens), args.max_rel_pos)
        if args.use_ast:
            # 偏移到序列独特编码
            length = len([tokenizer.cls_token]) + len(non_leaf_tokens)
            for i in range(len(split_leaf_tokens)):
                for j in range(len(split_leaf_tokens)):
                    rel_pos[(i+length, j+length)] = rel_leaf_pos[0][i][j] + args.max_rel_pos
        else:
            length = len([tokenizer.cls_token])
            for i in range(len(split_leaf_tokens)):
                for j in range(len(split_leaf_tokens)):
                    rel_pos[(i + length, j + length)] = rel_leaf_pos[0][i][j]

    if args.use_dfg:
        dfg = [d for d in dfg if d[1] in ori2cur_pos]
        dfg = dfg[:args.max_dfg_len]
        source_ids += [tokenizer.unk_token_id for x in dfg]
        length = len([tokenizer.cls_token]) + len(non_leaf_tokens) if args.use_ast else len([tokenizer.cls_token])

        # reindex 记录code token idx 到 dfg index
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        # 将dfg的指向节点位置映射为dfg的位置
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
        dfg_to_dfg = [x[-1] for x in dfg]
        # 记录dfg对应的code第几个到第几个index
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]

        # dfg to code
        source_token_len = len(source_tokens)
        for i, item in enumerate(dfg_to_code):
            for j in range(item[0], item[1]):
                if j + length < source_token_len-1:  # 剔除最后一个</s>
                    rel_pos[(i+source_token_len, j+length)] = 1  # 1 means <self>, or no need position
                    rel_pos[(j+length, i+source_token_len)] = 1

        # dfg to dfg
        for i, item in enumerate(dfg_to_dfg):
            for j in item:
                rel_pos[(i + source_token_len, j + source_token_len)] = 1

        source_tokens += [x[0] for x in dfg]

    # special tokens attend to all tokens
    for idx, i in enumerate(source_ids):
        if i in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
            for j in range(len(source_ids)):
                rel_pos[(i, j)] = 1

    max_source_len = 0

    position_idx = [-3]  # start pos

    if args.use_ast:
        max_source_len += args.max_ast_len
    else:
        max_source_len += args.max_code_len
    if args.use_dfg:
        max_source_len += args.max_dfg_len

    padding_length = max_source_len - len(source_ids)

    if args.use_ast:
        position_idx += [-3] * len(non_leaf_tokens)
    if args.use_code:
        position_idx += [i + tokenizer.pad_token_id + 1 for i in range(len(split_leaf_tokens))]
    else:
        position_idx += [-2] * len(split_leaf_tokens)
    position_idx += [-3]
    if args.use_dfg:
        position_idx += [-1] * len(dfg)

    position_idx += [tokenizer.pad_token_id] * (max_source_len-len(position_idx))
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_mask = [1] * (len(source_tokens))
    source_mask += [0] * padding_length

    # # func name 分词
    func_token = tokenize_with_camel_case(func_name)
    target_tokens = tokenizer.tokenize(' '.join(func_token))[:args.max_target_length-2]
    target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

    target_mask = [1] * len(target_ids)
    padding_length = 7 - len(target_ids)
    target_ids += [tokenizer.pad_token_id] * padding_length
    target_mask += [0] * padding_length

    gold_ids = target_ids

    # 保存回数据库
    example = FuncNamingFeature(
        example_id=example_index,
        source_ids=source_ids,
        position_idx=position_idx,
        rel_pos=rel_pos,
        source_mask=source_mask,
        target_ids=target_ids,
        target_mask=target_mask,
        gold_ids=gold_ids
    )
    # db[db_name].insert_one({"example_index": example_index, "source_ids": source_ids,
    #                         "position_idx": position_idx, "rel_pos": rel_pos,
    #                         "source_mask": source_mask, "target_ids": target_ids,
    #                         "target_mask": target_mask, "gold_ids": gold_ids})
    example = pickle.dumps(example)
    db[db_name].insert_one({"example_index": example_index, "example": example})

    return example_index


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)







