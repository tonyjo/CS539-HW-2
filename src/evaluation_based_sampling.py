import json
import math
import torch
import operator
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth

#--------------------------Useful functions and OPS ---------------------------#
# Functions
def _hashmap(x):
    # List (key, value, key, value, ....)
    return_x = {}
    if len(x)%2 == 0:
        for i in range(0, len(x)-1, 2):
            return_x[x[i]] = torch.tensor(x[i+1])
    else:
        raise IndexError('Un-even key-value pairs')

    return return_x

def _vector(x):
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.float32)
    else:
        # Maybe single value
        return torch.tensor([x], dtype=torch.float32)

def _put(x, idx_or_key, value):
    if isinstance(x, dict):
        try:
            if not torch.is_tensor(value):
                value = torch.tensor(value)
            x[idx_or_key] = value
        except:
            raise IndexError('Key {} cannot put in the dict'.format(idx_or_key))
        return x
    elif isinstance(x, list):
        try:
            x[idx_or_key] = value
        except:
            raise IndexError('Index {} is not present in the list'.format(idx_or_key))

        return x
    elif torch.is_tensor(x):
        try:
            if not torch.is_tensor(value):
                value = torch.tensor(value, dtype=x.dtype)
            x[idx_or_key] = value
        except:
            raise IndexError('Index {} is not present in the list'.format(idx_or_key))

        return x
    else:
         raise AssertionError('Unsupported data structure')

def _remove(x, idx_or_key):
    if isinstance(x, dict):
        try:
            x.pop(idx_or_key, None)
        except:
            raise IndexError('Key {} is not present in the dict'.format(idx_or_key))
        return x
    elif isinstance(x, list):
        try:
            x.pop(idx_or_key)
        except:
            raise IndexError('Index {} is not present in the list'.format(idx_or_key))
        return x
    elif torch.is_tensor(x):
        try:
            x = torch.cat((x[:idx_or_key], x[(idx_or_key+1):]))
        except:
            raise IndexError('Index {} is not present in the tensor'.format(idx_or_key))

        return x
    else:
         raise AssertionError('Unsupported data structure')

def _append(x, value):
    if isinstance(x, list):
        if isinstance(value, list):
            x.extend(value)
        else:
            # single value
            x.append(value)
    elif torch.is_tensor(x):
        if not torch.is_tensor(value):
            if isinstance(value, list):
                value = torch.tensor(value, dtype=x.dtype)
            else:
                value = torch.tensor([value], dtype=x.dtype)
        try:
            x = torch.cat((x, value))
        except:
            raise AssertionError('Cannot append the torch tensors')
        return x
    else:
        raise AssertionError('Unsupported data structure')

def _squareroot(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sqrt(x)

# OPS
basic_ops = {'+':torch.add,\
             '-':torch.sub,\
             '*':torch.mul,\
             '/':torch.div
}

math_ops = {'sqrt': lambda x: _squareroot(x)
}

data_struct_ops = {'vector': lambda x: _vector(x),\
                   'hash-map': lambda x: _hashmap(x)
}

data_interact_ops = {'first':lambda x: x[0],      # retrieves the first element of a list or vector e
                     'last':lambda x: x[-1],      # retrieves the last element of a list or vector e
                     'get':lambda x, idx: x[idx], # retrieves an element at index e2 from a list or vector e1, or the element at key e2 from a hash map e1.
                     'append': lambda x, y: _append(x, y),           # (append e1 e2) appends e2 to the end of a list or vector e1
                     'remove':lambda x, idx: _remove(x, idx),        # (remove e1 e2) removes the element at index/key e2 with the value e2 in a vector or hash-map e1.
                     'put':lambda x, idx, value: _put(x, idx, value) # (put e1 e2 e3) replaces the element at index/key e2 with the value e3 in a vector or hash-map e1.
}


#----------------------------Evaluation Functions -----------------------------#
def evaluate_program(ast):
    """
    Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    # Empty list
    if not ast:
        return [False, None]

    if len(ast) == 1:
        ast = ast[0]
        try:
            root, *tail = ast
            # Basic primitives
            if root in basic_ops.keys():
                op_func = basic_ops[root]
                return [op_func(float(tail[0]), evaluate_program(tail[1:])[0]), None]
            if root in math_ops.keys():
                op_func = math_ops[root]
                return [op_func(tail), None]
            # Data structures-- list and hash-map
            elif root in data_struct_ops.keys():
                op_func = data_struct_ops[root]
                return [op_func(tail), None]
            # Data structures interaction
            elif root in data_interact_ops.keys():
                op_func = data_interact_ops[root]
                if root == 'put':
                    # ['put', ['vector', 2, 3, 4, 5], 2, 3]
                    e1, e2, e3 = tail
                    get_data_struct, _ = evaluate_program([e1])
                    return [op_func(get_data_struct, e2, e3), None]
                elif root == 'remove' or root == 'append' or root == 'get':
                    # ['remove'/'append'/'get', ['vector', 2, 3, 4, 5], 2]
                    e1, e2 = tail
                    get_data_struct, _ = evaluate_program([e1])
                    return [op_func(get_data_struct, e2), None]
                else:
                    # # ['First'/'last', ['vector', 2, 3, 4, 5]]
                    get_data_struct, _ = evaluate_program(tail)
                    return [op_func(get_data_struct), None]
            else:
                # Most likely a single element list
                if tail == []:
                    return [root, None]
                else:
                    raise AssertionError('Unknown list with unsupported ops.')
        except:
            # Just a single element
            return [ast, None]

    else:
        raise AssertionError('Unsupported!')

    return [None, None]


def get_stream(ast):
    """
    Return a stream of prior samples
    """
    while True:
        yield evaluate_program(ast)


#------------------------------Test Functions --------------------------------#
def run_deterministic_tests():
    for i in range(1,14):
        # Note: this path should be with respect to the daphne path!
        # ast = daphne([f'desugar', '-i', f'{daphne_path}/src/programs/tests/deterministic/test_{i}.daphne'])
        #
        # ast_path = f'./jsons/tests/deterministic/test_{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)

        ast_path = f'./jsons/tests/deterministic/test_{i}.json'
        with open(ast_path) as json_file:
            ast = json.load(json_file)
        # print(ast)

        ret, sig = evaluate_program(ast)

        print('Running evaluation-based-sampling for deterministic test number {}:'.format(str(i)))
        truth = load_truth('./programs/tests/deterministic/test_{}.truth'.format(i))
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))

        print('Test passed')

    print('All deterministic tests passed')



def run_probabilistic_tests():

    num_samples=1e4
    max_p_value = 1e-4

    for i in range(11,13):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(ast)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)

    print('All probabilistic tests passed')


if __name__ == '__main__':
    daphne_path = '/Users/tony/Documents/prog-prob/CS539-HW-2'
    run_deterministic_tests()

    # run_probabilistic_tests()
    #
    #
    # for i in range(1,5):
    #     ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
    #     print('\n\n\nSample of prior of program {}:'.format(i))
    #     print(evaluate_program(ast)[0])
