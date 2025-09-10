# find a group of vectors that are as different from one another as possible 
import numpy as np
import pandas as pd
import random
import logging
import sys
from util import setup_logging

def hamming_distance(v1, v2):
    # normalized hamming distance
    return np.sum(v1 != v2)/len(v1)

def total_pairwise_hamming(vectors):
    return sum(hamming_distance(vectors[i], vectors[j]) for i in range(len(vectors)) for j in range(i + 1, len(vectors)))

def main(scheme, num_vec):
    # scheme: hqc128, hqc192, hqc256
    # num_vec: number of error patterns to use

    setup_logging(script = "find_group", scheme=scheme)

    num_rand = 600 # Number of random draws
    num_gen = 60
    num_children = 100

    # Load and preprocess the found error patterns
    df = pd.read_csv(f'./output/error_patterns_{scheme}.csv')
    df = df[df['Entropy'] >= 0.999]
    logging.info(f'number of vectors with individual entropy larger than 0.999: {len(df)}')
    df['Vectors'] = df['Vectors'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=int) if isinstance(x, str) else np.array(x, dtype=int))
    list_of_vectors = list(df['Vectors'])
    
    # initialize best distance
    best_distance =0
    for randi in range(num_rand):
        # initialize by randomly selecting num_vec vectors
        previous_rand_best = random.sample(list_of_vectors, num_vec)
        previous_rand_best_distance = total_pairwise_hamming(previous_rand_best)

        previous_gen_best = previous_rand_best[:]
        previous_gen_best_distance = previous_rand_best_distance
        
        current_gen_best = previous_rand_best[:]
        current_gen_best_distance = previous_rand_best_distance

        for _ in range(num_gen): 
            
            for _ in range(num_children): # number of children
                this_child = previous_gen_best[:]
                replace_index = random.randint(0, num_vec-1)
                new_choice = random.choice(list_of_vectors)
                while any(np.array_equal(new_choice, existing_vec) for existing_vec in this_child):
                    new_choice = random.choice(list_of_vectors)

                this_child[replace_index] = new_choice
                this_child_distance = total_pairwise_hamming(this_child)

                if this_child_distance > current_gen_best_distance:
                    current_gen_best_distance = this_child_distance
                    current_gen_best = this_child

            if current_gen_best_distance > previous_gen_best_distance:
                previous_gen_best = current_gen_best[:]
                previous_gen_best_distance = current_gen_best_distance
        
        if previous_gen_best_distance> best_distance:
            best_vectors = previous_gen_best[:]
            best_distance = previous_gen_best_distance
            logging.info(f"random draw {randi} found better vectors with distance {best_distance}")
            #if best_distance>39.909:
            #    logging.info(f"current draw {randi} best distance {best_distance} and vectors {best_vectors}")

    logging.info(f"Best group of {scheme} with {num_vec} vectors: {best_vectors}")
    logging.info("Total pairwise Hamming distance: " + str(best_distance))

    # save the group to npy
    best_vectors_array = np.array(best_vectors)
    output_path = f'data_input/best_vectors_{scheme}_{num_vec}.npy'
    np.save(output_path, best_vectors_array)
    logging.info(f"Saved the best vectors to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <scheme><number_of_error_patterns_to_use>")
        sys.exit(1)
    
    main(scheme=sys.argv[1], num_vec=int(sys.argv[2]))