import json
import torch
from collections import deque
import torch.distributions as dist

from daphne import daphne

# funcprimitives
from primitives import _totensor, _squareroot, _hashmap
from evaluation_based_sampling import evaluate_program
from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': dist.Normal,
       'sqrt': torch.sqrt}

# OPS
all_ops = {
 'sqrt': lambda x: _squareroot(x),
 'vector': lambda x: _totensor(x),
 'hash-map': lambda x: _hashmap(x),
 'first':lambda x: x[0],      # retrieves the first element of a list or vector e
 'last':lambda x: x[-1],      # retrieves the last element of a list or vector e
}


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


# Global vars
global rho
rho = {}
DEBUG = True # Set to true to see intermediate outputs for debugging purposes

def make_link(G, node1, node2):
    """
    DAG
    """
    if node1 not in G:
        G[node1] = {}
    (G[node1])[node2] = 1
    if node2 not in G:
        G[node2] = {}
    (G[node2])[node1] = -1

    return G

def eval_path(path, l={}, Y={}, P={}):
    if DEBUG:
        print('Local Vars: ', l)
        print('***************')
    outputs = []
    # Add Y to local vars
    for y in Y.keys():
        l[y] = Y[y]

    for n in path:
        # Evaluate node
        if DEBUG:
            print('Node: ', n)
        if n in l.keys():
            output_ = l[n]
            outputs.append([output_])

        else:
            p = P[n] # [sample* [n, 5, [sqrt, 5]]]
            if DEBUG:
                print('PMF for Node: ', p)
                root = p[0]
                tail = p[1]
                if DEBUG:
                    print('Empty sample Root: ', root)
                    print('Empty sample Root: ', tail)
                if root == "sample*":
                    if None not in tail:
                        sample_eval = ["sample", tail]
                        if DEBUG:
                            print('Sample AST: ', sample_eval)
                        output_ = evaluate_program(ast=[sample_eval], sig=None, l=l)[0]
                        if DEBUG:
                            print('Evaluated sample: ', output_)
                    else:
                        output_ = torch.tensor([0.00001])
                if DEBUG:
                    print('Node eval sample output: ', output_)
                # Check if not torch tensor
                if not torch.is_tensor(output_):
                    if isinstance(output_, list):
                        output_ = torch.tensor(output_, dtype=torch.float32)
                    else:
                        output_ = torch.tensor([output_], dtype=torch.float32)
                # Add to local var
                l[n] = output_
                # Collect
                outputs.append([output_])

    return outputs, l


def traverse(G, node, visit={}, path=[]):
    visit[node] = True
    neighbors = G[node]
    if DEBUG:
        print('------------')
        print('Node: ', node)
        print('visit: ', visit)
        print('Neigbors: ', neighbors)

    # Path should be empty only at the first node
    if path == []:
        path.append(node)

    for n in neighbors:
        if DEBUG:
            print('Neighbor: ', n)
        if (neighbors[n] == -1) and (n not in visit):
            if n not in path:
                path.append(n)
            traverse(G, node=n, visit=visit, path=path)
            return path

        elif (neighbors[n] == 1):
            return path

        else:
            raise AssertionError('WTF')

    # # Check if not torch tensor
    # if not torch.is_tensor(outpu):
    #     if isinstance(output_, list):
    #         output_ = torch.tensor(output_, dtype=torch.float32)
    #     else:
    #         output_ = torch.tensor([output_], dtype=torch.float32)
    # # Check for 0 dimensional tensor
    # elif output_.shape == torch.Size([]):
    #     output_ = torch.tensor([output_.item()], dtype=torch.float32)
    # try:
    #     output = torch.cat((output, output_))
    # except:
    #     raise AssertionError('Cannot append the torch tensors')

    # return output, l

def sample_from_joint(graph):
    """
    This function does ancestral sampling starting from the prior.
    Args:
        graph: json Graph of FOPPL program
    Returns: sample from the prior of ast
    """
    D, G, E = graph

    #import pdb; pdb.set_trace()
    if bool(D) == False:
        V = G['V']
        A = G['A']
        P = G['P']
        Y = G['Y']

        # Check for empty graph
        if not V:
            #import pdb; pdb.set_trace()
            # Evaluate expressions
            eval_E = evaluate_program(ast=[E])[0]
            try:
                eval_E = _totensor(x=eval_E)
            except:
                pass
            if DEBUG:
                print("Returned Expression: ", eval_E)
            return eval_E
        else:
            output = torch.zeros(0, dtype=torch.float32)
            # If empty edges
            if bool(A) == False:
                for v in V:
                    # Check if it's Prob. dist. defined and in evaluation exp
                    if (v in P.keys()) and (v in E):
                        # Get a sampler
                        p = P[v] # [sample* [n, 5, [sqrt, 5]]]
                        if DEBUG:
                            print('PMF for v: ', p)
                        root = p[0]
                        tail = p[1]
                        if DEBUG:
                            print('Empty sample Root: ', root)
                            print('Empty sample Root: ', tail)
                        if root == "sample*":
                            if None not in tail:
                                sample_eval = ["sample", tail]
                                if DEBUG:
                                    print('Sample AST: ', sample_eval)
                                sample  = evaluate_program(ast=[sample_eval])[0]
                                if DEBUG:
                                    print('Evaluated sample: ', sample)
                            else:
                                sample_eval = ["sample", tail]
                                sample = torch.tensor([0.00001])
                            # Check if not torch tensor
                            if not torch.is_tensor(sample):
                                if isinstance(sample, list):
                                    sample = torch.tensor(sample, dtype=torch.float32)
                                else:
                                    sample = torch.tensor([sample], dtype=torch.float32)
                            # Check for 0 dimensional tensor
                            elif sample.shape == torch.Size([]):
                                sample = torch.tensor([sample.item()], dtype=torch.float32)
                            try:
                                output = torch.cat((output, sample))
                            except:
                                raise AssertionError('Cannot append the torch tensors')

            # If connected V
            else:
                # Find the link nodes aka nodes not in V
                adj_list = []
                for a in A.keys():
                    links = A[a]
                    for link in links:
                        adj_list.append((a, link))
                if DEBUG:
                    print("Created Adjacency list: ", adj_list)

                # Create Graph
                G_ = {}
                for (n1, n2) in adj_list:
                    G_ = make_link(G=G_, node1=n1, node2=n2)
                if DEBUG:
                    print("Constructed Graph: ", G_)
                    print("Evaluation Expression: ", E)

                # import pdb; pdb.set_trace()
                # Eval based on E
                if isinstance(E, str):
                    # output = torch.zeros(0, dtype=torch.float32)
                    path = []
                    path = traverse(G=G_, node=E, visit={}, path=path)
                    if DEBUG:
                        print('Evaluated graph output: ', path)
                    # List Reverse
                    path.reverse()
                    if DEBUG:
                        print('Evaluated reverse graph path: ', path)

                    output, l = eval_path(path, l={}, Y=Y, P=P)

                elif isinstance(E, list):
                    import pdb; pdb.set_trace()
                    def eval_each_exp(exp, output=[], l={}):
                        root, *tail = exp
                        if DEBUG:
                            print('Root: ', root)
                            print('tail: ', tail)
                        if isinstance(root, str):
                            if root in all_ops.keys():
                                op_func = all_ops[root]
                                if tail == []:
                                    output_ = op_func(output)
                                else:
                                    output_ = op_func(eval_each_exp(tail, output=output))
                                return output
                            elif tail == []:
                                path = []
                                path = traverse(G=G_, node=root, visit={}, path=path)
                                if DEBUG:
                                    print('>>> Evaluated graph output: ', path)
                                # List Reverse
                                path.reverse()
                                if DEBUG:
                                    print('Evaluated reverse graph path: ', path)
                                # Evaluate
                                eval_output, l = eval_path(path, l=l, Y=Y, P=P)
                                if DEBUG:
                                    print('Evaluated sample path output: ', eval_output)
                                return eval_output
                            else:
                                # Evaluate root
                                path = []
                                path = traverse(G=G_, node=root, visit={}, path=path)
                                print('>>> Evaluated graph output: ', path)
                                # List Reverse
                                path.reverse()
                                if DEBUG:
                                    print('Evaluated reverse graph path: ', path)
                                # Evaluate
                                eval_output, l = eval_path(path, l=l, Y=Y, P=P)
                                if DEBUG:
                                    print('Evaluated sample path output: ', eval_output)
                                output.append([eval_output])

                                # Recurse
                                path_r = eval_each_exp(tail, output=output, l=l)
                                # # Evaluate
                                # eval_output = eval_path(path_r, l={}, Y=Y, P=P)
                                if DEBUG:
                                    print('Evaluated sample path output: ', path_r)

                            return output.append([path_r])

                        else:
                            root, *tail = root
                            if root in all_ops.keys():
                                op_func = all_ops[root]
                                if tail == []:
                                    output_ = op_func(output)
                                else:
                                    output_ = op_func(eval_each_exp(tail, output=output))

                                    return output

                    output = []
                    output = eval_each_exp(E, output)

                    # List Reverse
                    if DEBUG:
                        print('Evaluated sample graph path: ', output)
                else:
                    raise AssertionError('Invalid input of E!')

            return output
    else:
        # Pass through function defs, save to rho and procede with empty D
        for d in D.keys():
            # (name[param] body, )
            tail = D[d]
            if DEBUG:
                print('Defn d: ', d)
            try:
                fnname   = tail[0]
                fnparams = tail[1]
                fnbody   = tail[2]
            except:
                raise AssertionError('Failed to define function!')
            if DEBUG:
                print('Function Name : ', d)
                print('Function Param: ', fnparams)
                print('Function Body : ', fnbody)

            # Define functions
            rho[d] = [fnparams, fnbody]
            if DEBUG:
                print('Global Funcs : ', rho)

        D_={}
        graph = [D_, G, E]
        return sample_from_joint(graph)


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
    for i in range(1,13):
    #for i in range(2,3):
        # Note: this path should be with respect to the daphne path!
        # ast = daphne(['graph', '-i', f'{daphne_path}/src/programs/tests/deterministic/test_{i}.daphne'])
        # ast_path = f'./jsons/graphs/deterministic/test_{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)

        ast_path = f'./jsons/graphs/deterministic/test_{i}.json'
        with open(ast_path) as json_file:
            graph = json.load(json_file)
        print(graph)

        ret = sample_from_joint(graph)
        #ret = deterministic_eval(graph[-1])
        print(ret)
        print('Running evaluation-based-sampling for deterministic test number {}:'.format(str(i)))
        truth = load_truth('./programs/tests/deterministic/test_{}.truth'.format(i))
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
    #for i in range(6,7):
        # Note: this path should be with respect to the daphne path!
        # ast = daphne(['graph', '-i', f'{daphne_path}/src/programs/tests/probabilistic/test_{i}.daphne'])
        # ast_path = f'./jsons/graphs/probabilistic/test_{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)

        ast_path = f'./jsons/graphs/probabilistic/test_{i}.json'
        with open(ast_path) as json_file:
            graph = json.load(json_file)
        print(graph)

        stream = get_stream(graph)

        samples = []
        for k in range(1):
            samples.append(next(stream))
        print(samples)

        if i != 4:
            print('Running evaluation-based-sampling for probabilistic test number {}:'.format(str(i)))
            truth = load_truth('./programs/tests/probabilistic/test_{}.truth'.format(i))
            print(truth)
            p_val = run_prob_test(stream, truth, num_samples)

            print('p value', p_val)
            assert(p_val > max_p_value)

            print('Test passed')

    print('All probabilistic tests passed')


if __name__ == '__main__':
    daphne_path = '/Users/tony/Documents/prog-prob/CS539-HW-2'

    # run_deterministic_tests()

    # run_probabilistic_tests()

    #for i in range(1,5):
    for i in range(4,5):
        # Note: this path should be with respect to the daphne path!
        # ast = daphne(['graph', '-i', f'{daphne_path}/src/programs/{i}.daphne'])
        # ast_path = f'./jsons/graphs/final/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of prior of program {}:'.format(i))

        ast_path = f'./jsons/graphs/final/{i}.json'
        with open(ast_path) as json_file:
            graph = json.load(json_file)
        print(graph)

        output = sample_from_joint(graph)
        print(output)

        #stream = get_stream(graph)
        # samples = []
        # for k in range(1):
        #     samples.append(next(stream))
        # print(samples)
