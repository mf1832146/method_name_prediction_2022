import torch
import numpy as np
from torch.utils.data import Dataset
from _utils import ast2seq, get_ud2pos, build_relative_position, tokenize_with_camel_case


class Example(object):
    def __init__(self, idx, ast, dfg, target, index2code, lang):
        self.idx = idx
        self.target = target
        self.ast = ast
        self.dfg = dfg
        self.index2code = index2code
        self.lang = lang


class FuncNamingFeature(object):
    def __init__(self, example_id, source_ids, position_idx, rel_pos, source_mask, target_ids, target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.position_idx = position_idx
        self.rel_pos = rel_pos
        self.source_mask = source_mask
        self.target_ids = target_ids
        self.target_mask = target_mask


class FuncNamingDataset(Dataset):
    def __init__(self, examples, args, tokenizer):
        self.examples = examples
        self.args = args
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.max_source_length, self.args.max_source_length), dtype=np.bool)


def convert_example_to_func_naming_feature(item):
    example, example_index, tokenizer, args, stage = item

    ast = example.ast
    dfg = example.dfg
    func_name = example.target
    index2code = example.index2code

    """mask func name"""
    for k, v in index2code:
        if v == func_name:
            index2code[k] = '<mask>'

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
        max_source_len = args.max_ast_size - 2
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
        dfg = dfg[:args.max_dfg_len]
        source_ids += [tokenizer.unk_token_id for x in dfg]
        # padding_length = max_source_len + 2 - len(source_ids)
        # source_ids += [tokenizer.pad_token_id] * padding_length
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
        print(source_tokens)
        print(source_tokens[16])

    # special tokens attend to all tokens
    for idx, i in enumerate(source_ids):
        if i in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
            for j in range(len(source_ids)):
                rel_pos[(i, j)] = 1

    max_source_len = 0

    position_idx = [tokenizer.pad_token_id]  # start pos

    if args.use_ast:
        max_source_len += args.max_ast_size
    else:
        max_source_len += args.max_code_len
    if args.use_dfg:
        max_source_len += args.max_dfg_len

    padding_length = max_source_len - len(source_ids)

    if args.use_ast:
        position_idx += [tokenizer.pad_token_id] * len(non_leaf_tokens)
    if args.use_code:
        position_idx += [i + tokenizer.pad_token_id + 1 for i in range(len(split_leaf_tokens))]
    else:
        position_idx += [tokenizer.pad_token_id] * len(split_leaf_tokens)

    position_idx += [tokenizer.pad_token_id] * (max_source_len-len(position_idx))
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_mask = [1] * (len(source_tokens))
    source_mask += [0] * padding_length

    # # func name 分词
    func_token = tokenize_with_camel_case(func_name)

    if stage == 'test':
        target_tokens = tokenizer.tokenize('None')
    else:
        target_tokens = tokenizer.tokenize(' '.join(func_token))[:7-2]

    target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    target_mask = [1] * len(target_ids)
    padding_length = 7 - len(target_ids)
    target_ids += [tokenizer.pad_token_id] * padding_length
    target_mask += [0] * padding_length

    return FuncNamingFeature(example_index,
                             source_ids,
                             position_idx,
                             rel_pos,
                             source_mask,
                             target_ids,
                             target_mask)





