import torch
from pymongo import MongoClient
import pickle
from tqdm import tqdm


client = MongoClient('172.29.7.221', 27017, username='admin', password='123456')
codes = client.code_search_net.codes
result = codes.find({'partition': 'train', 'lang': 'java'})[0]

word_tokens = []


def get_leaf_node(root_node, ll):
    if len(root_node['children'])==0 and root_node['type']!='comment':
        ll.append((root_node['position_id'], root_node['type']))
    else:
        for child in root_node['children']:
            get_leaf_node(child, ll)


def get_par_position(root_node, par_node, tar_pos_id, right_pos_id):
    if root_node['position_id'] == tar_pos_id:
        right_pos_id.append(par_node)
    else:
        for child in root_node['children']:
            get_par_position(child, root_node, tar_pos_id, right_pos_id)


print()
word_tokens.extend(result['code_tokens'])
ast = pickle.loads(result['ast'])
index_to_code = pickle.loads(result['index_to_code'])

print(result['code_tokens'])
print('+'*10)
leaf_nodes = []
get_leaf_node(ast, leaf_nodes)
lead_values = []
for position_id, node_type in leaf_nodes:
    if position_id in index_to_code:
        lead_values.append(index_to_code[position_id][1])
    else:
        print('error', position_id)

print(lead_values)


position_idx = torch.IntTensor([[2,3,4,5,6,7,8,0,0,0]])

node_mask = position_idx.eq(0)
token_mask = position_idx.ge(2)

print(node_mask[:, :, None])

print(token_mask[:, None, :])

node_to_token_mask = node_mask[:, :, None] & token_mask[:, None, :]

print((node_to_token_mask.sum(-1) + 1e-10)[:, :, None])

nodes_to_token_mask = node_to_token_mask / (node_to_token_mask.sum(-1) + 1e-10)[:, :, None]

print(nodes_to_token_mask)

a = torch.randn((2,4)).cuda()
print(torch.sum(a).cpu().numpy())

