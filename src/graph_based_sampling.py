import json
import torch
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
DEBUG = False # Set to true to see intermediate outputs for debugging purposes

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

def traverse_and_eval(G, node, output, visit={}, P={}, l={}):
    visit[node] = True
    neighbors = G[node]
    if DEBUG:
        print('Node: ', node)
        print('visit: ', visit)
        print('Neigbors: ', neighbors)
        print('local_vars: ', l)
        print('Global vars: ', rho)

    for n in neighbors:
        if DEBUG:
            print('Neighbor: ', n)
        if (neighbors[n] == -1) and (n not in visit):
            output_, l = traverse_and_eval(G, node=n, output=output, visit=visit, P=P, l=l)

        elif (neighbors[n] == 1) and (n in visit):
            # Evaluate node
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

            return output_, l

        elif (neighbors[n] == 1) and (n not in visit):
            raise AssertionError('Something wrong')

        else:
            if DEBUG:
                print('Did not find: ', n)
            raise AssertionError('Something wrong')

        # Check if not torch tensor
        if not torch.is_tensor(output_):
            if isinstance(output_, list):
                output_ = torch.tensor(output_, dtype=torch.float32)
            else:
                output_ = torch.tensor([output_], dtype=torch.float32)
        # Check for 0 dimensional tensor
        elif output_.shape == torch.Size([]):
            output_ = torch.tensor([output_.item()], dtype=torch.float32)
        try:
            output = torch.cat((output, output_))
        except:
            raise AssertionError('Cannot append the torch tensors')

    return output, l

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
                    output = torch.zeros(0, dtype=torch.float32)
                    output, _ = traverse_and_eval(G=G_, node=E, output=output, visit={}, P=P, l={})
                    if DEBUG:
                        print('Evaluated graph output: ', output)
                elif isinstance(E, list):
                    for exp in E:
                        print(exp)
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

        # samples = []
        # for k in range(4):
        #     samples.append(next(stream))
        # print(samples)

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

    run_deterministic_tests()

    run_probabilistic_tests()

    # for i in range(1,5):
    #     graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
    #     print('\n\n\nSample of prior of program {}:'.format(i))
    #     print(sample_from_joint(graph))
