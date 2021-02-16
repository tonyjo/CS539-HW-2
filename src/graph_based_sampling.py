import json
import torch
import torch.distributions as dist

from daphne import daphne

#from primitives import funcprimitives #TODO
from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': dist.Normal,
       'sqrt': torch.sqrt}


def deterministic_eval(exp):
    """
    Evaluation function for the deterministic target language of the graph based representation.
    """
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    else:
        raise("Expression type unknown.", exp)


def sample_from_joint(graph):
    """
    This function does ancestral sampling starting from the prior.
    """
    # TODO insert your code here
    D, G, E = graph

    if bool(D) == False:
        V = G['V']
        A = G['A']
        P = G['P']
        Y = G['Y']

        # Check for empty graph
        if not V:
            return E
    else:
        for i in range(D.keys()):
            pass

    return torch.tensor([0.0, 0.0, 0.0])


def get_stream(graph):
    """
    Return a stream of prior samples
    Args:
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
    """
    while True:
        yield sample_from_joint(graph)




#Testing:
def run_deterministic_tests():
    #for i in range(1,13):
    for i in range(1,2):
        # Note: this path should be with respect to the daphne path!
        # ast = daphne(['graph', '-i', f'{daphne_path}/src/programs/tests/deterministic/test_{i}.daphne'])
        # ast_path = f'./jsons/graphs/deterministic/test_{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)

        ast_path = f'./jsons/graphs/deterministic/test_{i}.json'
        with open(ast_path) as json_file:
            graph = json.load(json_file)
        print(graph)

        ret = deterministic_eval(graph[-1])

        print('Running evaluation-based-sampling for deterministic test number {}:'.format(str(i)))
        truth = load_truth('./programs/tests/deterministic/test_{}.truth'.format(i))
        print(truth)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))

        print('Test passed')

    print('All deterministic tests passed')



def run_probabilistic_tests():
    #TODO:
    num_samples=1e4
    max_p_value = 1e-4

    for i in range(1,7):
        # Note: this path should be with respect to the daphne path!
        ast = daphne(['graph', '-i', f'{daphne_path}/src/programs/tests/probabilistic/test_{i}.daphne'])
        ast_path = f'./jsons/graphs/probabilistic/test_{i}.json'
        with open(ast_path, 'w') as fout:
            json.dump(ast, fout, indent=2)

        # stream = get_stream(graph)
        #
        # p_val = run_prob_test(stream, truth, num_samples)
        #
        # print('p value', p_val)
        # assert(p_val > max_p_value)

    print('All probabilistic tests passed')


if __name__ == '__main__':
    daphne_path = '/Users/tony/Documents/prog-prob/CS539-HW-2'

    run_deterministic_tests()
    #run_probabilistic_tests()
    #
    #
    #
    #
    # for i in range(1,5):
    #     graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
    #     print('\n\n\nSample of prior of program {}:'.format(i))
    #     print(sample_from_joint(graph))
