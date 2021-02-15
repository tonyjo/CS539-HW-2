import json
import torch
import torch.distributions as distributions
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
basic_ops = {'+':torch.add,
             '-':torch.sub,
             '*':torch.mul,
             '/':torch.div
}

math_ops = {'sqrt': lambda x: _squareroot(x)
}

data_struct_ops = {'vector': lambda x: _vector(x),
                   'hash-map': lambda x: _hashmap(x)
}

data_interact_ops = {'first':lambda x: x[0],      # retrieves the first element of a list or vector e
                     'last':lambda x: x[-1],      # retrieves the last element of a list or vector e
                     'get':lambda x, idx: x[idx], # retrieves an element at index e2 from a list or vector e1, or the element at key e2 from a hash map e1.
                     'append': lambda x, y: _append(x, y),           # (append e1 e2) appends e2 to the end of a list or vector e1
                     'remove':lambda x, idx: _remove(x, idx),        # (remove e1 e2) removes the element at index/key e2 with the value e2 in a vector or hash-map e1.
                     'put':lambda x, idx, value: _put(x, idx, value) # (put e1 e2 e3) replaces the element at index/key e2 with the value e3 in a vector or hash-map e1.
}

dist_ops = {"normal":lambda mu, sig: distributions.normal.Normal(mu, sig),
            "beta":lambda a, b: distributions.beta.Beta(a, b),
            "exponential":lambda rate: distributions.exponential.Exponential(rate),
            "uniform": lambda low, high: distributions.uniform.Uniform(low, high)
}

cond_ops={"<":  lambda a, b: a < b,
          ">":  lambda a, b: a > b,
          ">=": lambda a, b: a >= b,
          "<=": lambda a, b: a <= b,
          "|":  lambda a, b: a or b,
}

# Global vars
pho = {}
DEBUG = True # Set to true to see intermediate outputs for debugging purposes
#----------------------------Evaluation Functions -----------------------------#
def evaluate_program(ast, sig=None, l={}):
    """
    Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    # Empty list
    if not ast:
        return [False, sig]

    if len(ast) == 1:
        ast = ast[0]
        import pdb; pdb.set_trace()
        if DEBUG:
            print('Current program: ', ast)
        try:
            root, *tail = ast
            if DEBUG:
                print('Current OP: ', root)
            # Basic primitives
            if root in basic_ops.keys():
                op_func = basic_ops[root]
                return [op_func(float(tail[0]), evaluate_program(tail[1:], sig, l=l)[0]), sig]
            if root in math_ops.keys():
                op_func = math_ops[root]
                return [op_func(tail), sig]
            # Data structures-- list and hash-map
            elif root in data_struct_ops.keys():
                op_func = data_struct_ops[root]
                return [op_func(tail), sig]
            # Data structures interaction
            elif root in data_interact_ops.keys():
                op_func = data_interact_ops[root]
                if root == 'put':
                    # ['put', ['vector', 2, 3, 4, 5], 2, 3]
                    e1, e2, e3 = tail
                    if isinstance(e1, list):
                        get_data_struct, _ = evaluate_program([e1], sig, l=l)
                    else:
                        # Most likely a pre-defined varibale in l
                        get_data_struct = l[e1]
                    return [op_func(get_data_struct, e2, e3), sig]
                elif root == 'remove' or root == 'append' or root == 'get':
                    # ['remove'/'append'/'get', ['vector', 2, 3, 4, 5], 2]
                    e1, e2 = tail
                    if isinstance(e1, list):
                        get_data_struct, _ = evaluate_program([e1], sig, l=l)
                    else:
                        # Most likely a pre-defined varibale in l
                        get_data_struct = l[e1]
                    return [op_func(get_data_struct, e2), sig]
                else:
                    # ['First'/'last', ['vector', 2, 3, 4, 5]]
                    if isinstance(e1, list):
                        get_data_struct, _ = evaluate_program([e1], sig, l=l)
                    else:
                        # Most likely a pre-defined varibale in l
                        get_data_struct = l[e1]
                    get_data_struct, _ = evaluate_program(tail, sig, l=l)
                    return [op_func(get_data_struct), sig]
            # Conditionals
            elif root in cond_ops.keys():
                # (< a b)
                op_func = cond_ops[root]
                if DEBUG:
                    print('Conditional param-1: ', tail[0])
                    print('Conditional param-2: ', tail[1])
                a = evaluate_program([tail[0]], sig, l=l)
                b = evaluate_program([tail[1]], sig, l=l)
                try:
                    a = a[0]
                except:
                    # In case of functions returning only a single value and not sigma
                    pass
                try:
                    b = b[0]
                except:
                    pass
                if DEBUG:
                    print('Eval Conditional param-1: ', a)
                    print('Eval Conditional param-2: ', b)
                return [op_func(a, b), sig]
            # Assign
            elif root == 'let':
                # (let [params] body)
                # tail-0: params
                if DEBUG:
                    print('Total params in let: ', len(tail[0]))
                for e1 in range(0, len(tail[0]), 2):
                    if DEBUG:
                        print('Vars: ', tail[0][e1])
                        print('Expr: ', tail[0][e1+1])
                    v1 = tail[0][e1]
                    e1 = evaluate_program([tail[0][e1+1]], sig, l=l)
                    try:
                        # In case of sampler only a single return
                        e1 = e1[0]
                    except:
                        pass
                    l[v1] = e1
                if DEBUG:
                    print('Local Params :  ', l)
                    print('Recursive Body: ', tail[1])
                # tail-1 body
                return evaluate_program([tail[1]], sig, l=l)
            # Assign
            elif root == "if":
                # (if e1 e2 e3)
                if DEBUG:
                    print('Conditonal Expr1 :  ', tail[0])
                    print('Conditonal Expr2 :  ', tail[1])
                    print('Conditonal Expr3 :  ', tail[2])
                e1_, sig = evaluate_program([tail[0]], sig, l=l)
                if DEBUG:
                    print('Conditonal eval :  ', e1_)
                if e1_:
                    return evaluate_program([tail[1]], sig, l=l)
                else:
                    return evaluate_program([tail[2]], sig, l=l)
            # Get distribution
            elif root in dist_ops.keys():
                op_func = dist_ops[root]
                if len(tail) == 2:
                    para1 = evaluate_program([tail[0]], sig, l=l)[0]
                    para2 = evaluate_program([tail[1]], sig, l=l)[0]
                    if DEBUG:
                        print('Sampler Parameter-1: ', para1)
                        print('Sampler Parameter-2: ', para2)
                    return [op_func(para1, para2), sig]
                else:
                    # Exponential has only one parameter
                    para1 = evaluate_program([tail[0]], sig, l=l)[0]
                    if DEBUG:
                        print('Sampler Parameter-1: ', para1)
                    return [op_func(para1), sig]
            # Sample
            elif root == 'sample':
                if DEBUG:
                    print('Sampler program: ', tail)
                sampler = evaluate_program(tail, sig, l=l)[0]
                if DEBUG:
                    print('Sampler: ', sampler)
                return sampler.sample()
            else:
                # Most likely a single element list
                if DEBUG:
                    print('Root Value: ', )
                    print('Tail Value: '. )
                if tail == []:
                    # Check in local vars
                    if root in l.keys():
                        return [l[root], sig]
                    else:
                        return [root, sig]
                else:
                    raise AssertionError('Unknown list with unsupported ops.')
        except:
            # Just a single element
            return [ast, sig]

    else:
        raise AssertionError('Unsupported!')

    return [None, sig]


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
    #num_samples=10
    max_p_value =1e-4

    # for i in range(1,7):
    for i in range(5,6):
        # Note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar', '-i', f'{daphne_path}/src/programs/tests/probabilistic/test_{i}.daphne'])
        # ast_path = f'./jsons/tests/probabilistic/test_{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)

        ast_path = f'./jsons/tests/probabilistic/test_{i}.json'
        with open(ast_path) as json_file:
            ast = json.load(json_file)
        print(ast)

        eval = evaluate_program(ast)
        print(eval)

        stream = get_stream(ast)

        # print(stream)
        #
        # samples = []
        # for k in range(4):
        #     # print(next(stream))
        #     samples.append(next(stream))
        # print(samples)

        print('Running evaluation-based-sampling for probabilistic test number {}:'.format(str(i)))
        #truth = load_truth('./programs/tests/probabilistic/test_{}.truth'.format(i))
        #p_val = run_prob_test(stream, truth, num_samples)

        #assert(p_val > max_p_value)

    print('All probabilistic tests passed')


if __name__ == '__main__':
    daphne_path = '/Users/tony/Documents/prog-prob/CS539-HW-2'
    #run_deterministic_tests()

    run_probabilistic_tests()
    #
    #
    # for i in range(1,5):
    #     ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
    #     print('\n\n\nSample of prior of program {}:'.format(i))
    #     print(evaluate_program(ast)[0])
