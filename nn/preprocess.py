# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # check inputs
    if len(seqs) != len(labels):
        raise ValueError("seqs and labels must be the same length")
    if len(seqs) == 0:
        return [], []
    
    # sort pos / neg
    pos_seqs = [s for s, l in zip(seqs, labels) if l]
    neg_seqs = [s for s, l in zip(seqs, labels) if not l]

    n_pos = len(pos_seqs)
    n_neg = len(neg_seqs)

    # make sure at least one of each class
    if n_pos == 0 or n_neg == 0:
        raise ValueError("classes must have at least 1 item to sample")

    # determine which to sample
    if n_pos == n_neg: #balanced
        indices = np.random.permutation(len(seqs))
        return [seqs[i] for i in indices], [labels[i] for i in indices]
    if n_pos < n_neg:
        minor = pos_seqs
        major = neg_seqs
        target_size = n_neg
        min_label = True
        maj_label = False
    else:
        minor = neg_seqs
        major = pos_seqs
        target_size = n_pos
        min_label = False
        maj_label = True
    
    # oversample w replacement
    sampled_indices = np.random.choice(len(minor),size=target_size,replace=True)
    sampled_minority = [minor[i] for i in sampled_indices]

    # combine
    sampled_seqs = major + sampled_minority
    sampled_labels = ([maj_label] * len(major) + [min_label] * len(sampled_minority))

    # shuffle
    shuffle_indices = np.random.permutation(len(sampled_seqs))
    sampled_seqs = [sampled_seqs[i] for i in shuffle_indices]
    sampled_labels = [sampled_labels[i] for i in shuffle_indices]

    return sampled_seqs, sampled_labels


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    dictionary = {'A': [1,0,0,0], 
                  'T': [0,1,0,0], 
                  'C': [0,0,1,0], 
                  'G': [0,0,0,1]}

    # check inputs
    if not seq_arr:
        return np.array([]) #empty array
    for seq in seq_arr:
        if len(seq_arr[0]) != len(seq):
            raise ValueError("sequences must be the same length")     

    output  = []
    for seq in seq_arr:
        encoded = []
        for base in seq:
            base_up = base.upper() #allow lowercase inputs
            if base_up not in dictionary:
                raise ValueError('invalid nucleotide')
            
            encoded.extend(dictionary[base_up])
        output.append(encoded)

    encodings = np.array(output, dtype=float)
    return encodings