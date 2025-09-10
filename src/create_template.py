import numpy as np
import sys
import pickle
from util import sample_vector_from_scheme, rm_decoder_result_w_noise, load_config, load_lib, load_rm_decoder


def load_vectors(scheme, num_vec):
    file_path = f'data_input/best_vectors_{scheme}_{num_vec}.npy'
    vectors = np.load(file_path)
    return vectors

def create_template(scheme, error_patterns, n2, rm_decoder, rho, lib):
    probability_template = {}
    for idx, this_e in enumerate(error_patterns):
        
        results_distribution = {0: 0, 1: 0}
        u_vectors_summary = {key: np.zeros(n2, dtype=int) for key in results_distribution.keys()}

        for _ in range(1000000): 
            u_vector = sample_vector_from_scheme(scheme)
            uXORe = np.bitwise_xor(u_vector, this_e)

            rm_decoder_result = rm_decoder_result_w_noise(lib=lib, scheme=scheme, vector_sum=uXORe, rm_decoder=rm_decoder,rho=rho)
            results_distribution[rm_decoder_result] += 1
            u_vectors_summary[rm_decoder_result] += u_vector

        # Calculate proportions of 1s in each position for each result category
        this_template = {}
        for key in u_vectors_summary:
                this_template[key] = u_vectors_summary[key] / results_distribution[key]

        probability_template[f"template_{idx}"] = this_template
    return probability_template

def save_template_to_pickle(probability_template, scheme, rho, num_vec):
    filename = f'./templates/{scheme}_{num_vec}_rho={rho}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(probability_template, file)

def main(scheme, num_vec, rho):
    params = load_config(scheme)
    lib_path = params.get("lib path", "lib path_not_specified")
    lib = load_lib(lib_path=lib_path)
    rm_decoder = load_rm_decoder(lib=lib)
    n2 = params.get("n2", "n2_not_specified")

    error_patterns = load_vectors(scheme, num_vec)
    templates = create_template(scheme, error_patterns, n2, rm_decoder, rho, lib)
    save_template_to_pickle(templates, scheme, rho, num_vec)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <scheme> <number_of_error_patterns_to_use> <oracle_accuracy_rho>")
        sys.exit(1)
    
    main(scheme=sys.argv[1], num_vec=int(sys.argv[2]), rho=float(sys.argv[3]))