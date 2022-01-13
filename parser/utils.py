import re
from io import StringIO
import tokenize
from copy import deepcopy


def get_root(ast) -> int:
    """get root node index"""
    for idx, node in ast.items():
        if node['parent'] is None:
            return idx


def reset_indices(ast):
    '''rename ast tree's node indices with consecutive indices'''
    if sorted(list(ast.keys())) == list(range(len(ast))):
        return ast

    # firstly, resort node index with a prefix "_", e.g. 0 => "_0"
    _idx = 0

    def _dfs(idx, _parent_idx):
        nonlocal _idx
        _new_idx, _idx = f'_{_idx}', _idx + 1  # update for next node
        node = ast.pop(idx)
        ast[_new_idx] = node

        # update its parent's children
        if node['parent'] is None:
            pass  # current node is root node, no need for update its children
        else:
            parent_node = ast[_parent_idx]
            # update its index in its parent node
            parent_node['children'][parent_node['children'].index(idx)] = _new_idx
            # update parent index
            node['parent'] = _parent_idx

        if 'children' in node:  # non-leaf nodes, traverse its children nodes
            # update its children nodes' parent
            for child_idx in node['children']:
                _dfs(child_idx, _parent_idx=_new_idx)
        else:
            return

    root_idx = get_root(ast)
    _dfs(root_idx, _parent_idx=None)

    # recover name: from _* => *
    node_ids = deepcopy(list(ast.keys()))
    for idx in node_ids:
        node = ast.pop(idx)
        # update children index
        if 'children' in node:
            node['children'] = [int(child_idx[1:]) for child_idx in node['children']]
        # update parent index
        if node['parent'] == None:
            pass
        else:
            node['parent'] = int(node['parent'][1:])
        ast[int(idx[1:])] = node  # _idx => idx
    return ast


def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)

def tree_to_token_index(root_node):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        return [(root_node.start_point,root_node.end_point)]
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_token_index(child)
        return code_tokens
    
def tree_to_variable_index(root_node,index_to_code):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        index=(root_node.start_point,root_node.end_point)
        _,code=index_to_code[index]
        if root_node.type!=code:
            return [(root_node.start_point,root_node.end_point)]
        else:
            return []
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_variable_index(child,index_to_code)
        return code_tokens    


def index_to_code_token(index,code):
    start_point=index[0]
    end_point=index[1]
    if start_point[0]==end_point[0]:
        s=code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s=""
        s+=code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1,end_point[0]):
            s+=code[i]
        s+=code[end_point[0]][:end_point[1]]   
    return s
   