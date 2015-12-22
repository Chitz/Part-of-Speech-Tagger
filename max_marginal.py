from __future__ import division
import result_cache

__author__ = 'sagar, chitesh'

'''
Step 4 - Approximate max-marginal inference

In this step, we use the large number of samples generated in Step 3 to find the approximate max-marginal inference

-- Description -

    - As we are efficiently, saving the unique samples from Step 3, along with the count of each sample
    - This steps only calculates the max-marginal for each word, ie., which parts of speech for the word has the highest count
    - It also returns of the probability of picking the word

-- Results -
    - On bc.test
    - We get a percentage of -
        -  Word correctness -  96.09%
        -  Sentence correctness - 60.77%

    This varies in initial range of 96% for word correctness.

-- Assumptions & Design Decisions -
    - We are using the 1000 samples generated from Step 3 in this Step.
    - As we already have the unique sample count, and samples, we are able to calculate with ease

'''

import sys


def get_part_of_speech(max_sample_map):
    pos = []
    pos_prob = []
    for index in range(0, len(max_sample_map)):
        best_prob = -sys.maxint
        total_speech_prob = 0
        best_speech = ""
        for speech in max_sample_map[index]:
            total_speech_prob += max_sample_map[index][speech]
            if best_prob < max_sample_map[index][speech]:
                best_prob = max_sample_map[index][speech]
                best_speech = speech
        pos.append(best_speech)
        pos_prob.append(best_prob / total_speech_prob)

    # store result in result cache for future use
    result_cache.max_marginal = pos
    return [pos, pos_prob]
