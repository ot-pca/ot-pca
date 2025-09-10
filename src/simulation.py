# calculate number of errors in the first half of the reordered vector, and simulate success rate for isd
import numpy as np
import logging
import math
import sys
import pickle
from util import rm_decoder_result_w_noise,  load_config, load_lib, load_rm_decoder, setup_logging


def generate_full_vector_dropping_excess(n, n1, n2, w):
    vector = np.zeros(n, dtype=int)
    vector[:w] = 1
    np.random.shuffle(vector)
    return vector[:n1 * n2]
    
def generate_probability_blocks(error_patterns, templates,vector,n2,lib,scheme,rm_decoder,rho):
    # get RM decoding result for each block, and retrieve the probability from the templates
    vector_blocks = np.array_split(vector, int(len(vector)/n2))
    vector_probability_blocks = np.empty_like(vector_blocks, dtype=float)

    for i, vector_block in enumerate(vector_blocks):
        #initialize probability for this block
        block_probability = np.ones(n2)

        for j, error_pattern in enumerate(error_patterns):
            # uXORe
            uXORe = np.bitwise_xor(vector_block, error_pattern)

            oracle_output = rm_decoder_result_w_noise(lib=lib, scheme=scheme, vector_sum=uXORe, rm_decoder=rm_decoder,rho=rho) 

            this_template = templates['template_'+str(j)]
            probability = np.array(this_template[oracle_output])
            block_probability *= probability
        
        vector_probability_blocks[i] = block_probability

    return vector_probability_blocks.ravel()
    
def sample_and_sort_according_to_probability(n, n1, n2, w,error_patterns, templates,lib,scheme,rm_decoder,rho):
    x = generate_full_vector_dropping_excess(n=n, n1=n1, n2=n2, w=w)
    y = generate_full_vector_dropping_excess(n=n, n1=n1, n2=n2, w=w)
    xy = np.concatenate((x, y))
    xy_probabilities = generate_probability_blocks(error_patterns=error_patterns, templates=templates,vector=xy,n2=n2,lib=lib,scheme=scheme,rm_decoder=rm_decoder,rho=rho)
    sorted_indices = np.argsort(xy_probabilities)
    sorted_xy = xy[sorted_indices]
    return sorted_xy
    
def calculate_num_errors_half_length(n,array):
    half_sorted_xy = array[:n]
    return sum(half_sorted_xy)

def get_half_sorted_xy_plus_l(n,l, array):
    half_sorted_xy_plus_l = array[:n+l]
    return half_sorted_xy_plus_l

def exclude_random_positions(array, T1, T0):
    
    if T1 >= T0:
        raise ValueError("T0 must be less than T1")

    new_array = array[:T0]
    # Generate all indices from 0 to T0-1
    all_indices = np.arange(T0)
    
    # Randomly choose T1 indices to exclude
    exclude_indices = np.random.choice(all_indices, T1, replace=False)
    
    # Get the mask of indices to include by checking which are not in exclude_indices
    include_mask = np.isin(all_indices, exclude_indices, invert=True)
    
    # Use the mask to select the remaining elements
    remaining_array =new_array[include_mask]

    excluded_array = new_array[exclude_indices]

    return remaining_array, excluded_array

def random_split_array(array, array_length):
    # Check that the number of elements is even
    if len(array) % 2 != 0:
        raise ValueError("The array must contain an even number of elements")
    
    indices = np.arange(array_length)  # Create an array of indices
    np.random.shuffle(indices)  # Shuffle the indices

    # Split indices for preserving order within each split
    first_half_indices = sorted(indices[:array_length//2])
    second_half_indices = sorted(indices[array_length//2:])

    # Use these indices to form the new arrays
    first_half = array[first_half_indices]
    second_half = array[second_half_indices]

    return first_half, second_half
    

def calculate_success_rate(half_sorted_xy_plus_l, maxW, T0, T1, T):

    half_sorted_xy_plus_l_weight = half_sorted_xy_plus_l.sum()

    # if the weight is larger than maxW, the split will end up with more than maxW/2 errors in one list, and cannot be solved. --> failure 
    if half_sorted_xy_plus_l_weight > maxW:
        return [half_sorted_xy_plus_l_weight,float('inf')]

    found = False

    for j in range(T):
        
        T0_minus_T1, excluded_array = exclude_random_positions(array=half_sorted_xy_plus_l, T1=T1, T0=T0)

        if np.sum(excluded_array)>0: # excluded array contains one. draw another round.
            num_draw=j+1
            continue
        
        L=np.concatenate((T0_minus_T1,half_sorted_xy_plus_l[T0:]))

        lenL = len(L)

        L1, L2 = random_split_array(array=L, array_length=lenL)

        # if any list is greater than maxW/2, cannot be solved, draw another round 
        if np.sum(L1)<=maxW/2 and np.sum(L2)<=maxW/2:
            num_draw=j+1
            found=True
            break

    if not found:
        num_draw=float('inf')
    
    return [half_sorted_xy_plus_l_weight, num_draw]


def main(scheme, num_vec, rho):
    # load parameters
    params = load_config(scheme)
    lib_path = params.get("lib path", "lib path_not_specified")
    lib = load_lib(lib_path=lib_path)
    rm_decoder = load_rm_decoder(lib=lib)

    n2 = params.get("n2", "n2_not_specified")
    n = params.get("n", "n_not_specified")
    n1 = params.get("n1", "n1_not_specified")
    w = params.get("w", "w_not_specified")

    setup_logging(script="simulation",scheme=scheme)


    # load error patterns
    error_patterns = np.load(f'data_input/best_vectors_{scheme}_{num_vec}.npy')
    
    # load template
    with open(f'./templates/{scheme}_{num_vec}_rho={rho}.pkl', 'rb') as file:
        templates = pickle.load(file)

    # calculate number of errors at half-length for each reordered key vector, and simulate success rate for isd
    error_stats = []
    num_samples = 10000

    result = []
    for _ in range(num_samples):
        # sample xy and sort according to probability
        sorted_xy = sample_and_sort_according_to_probability(n=n, n1=n1, n2=n2, w=w,error_patterns=error_patterns, templates=templates,lib=lib,scheme=scheme,rm_decoder=rm_decoder,rho=rho)

        # Part 1: calculate number of errors at half-length for each sample
        num_errors_half_length = calculate_num_errors_half_length(n=n,array=sorted_xy)
        error_stats.append(num_errors_half_length)

        # Part 2: evaluate success or not
        # parameters
        maxW = 4
        if scheme == 'hqc128':
            T0=12000
            T1=10000
        elif scheme =='hqc192':
            T0=24000
            T1=18000
        elif scheme == 'hqc256':
            T0=40000
            T1=30000        
        
        T=16 # number of random draws and splits 
        
        if scheme == 'hqc128':
            l=51
        else:
            l=61
        
        half_sorted_xy_plus_l = get_half_sorted_xy_plus_l(n=n,l=l,array=sorted_xy)

        success_trial =calculate_success_rate(half_sorted_xy_plus_l=half_sorted_xy_plus_l, maxW=maxW, T0=T0, T1=T1, T=T)

        result.append(success_trial)

    result_list = [t[1] for t in result] 
    success_rate = sum(1 for x in result_list if math.isfinite(x))/len(result_list)


    # save error stats
    np.save(f'./output/error_stats_{scheme}_{num_vec}_rho={rho}.npy', np.array(error_stats))
    logging.info(np.mean(error_stats))

    # log success rate
    logging.info(f'maxW={maxW} success rate: {success_rate}')
    logging.info(f'result with rho={rho}, T0={T0}, T1={T1}, T={T}, maxW={maxW}: {result}')
                
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <scheme> <number_of_error_patterns_to_use> <rho>")
        sys.exit(1)
    main(scheme=sys.argv[1], num_vec=int(sys.argv[2]), rho=float(sys.argv[3]))
