__author__ = 'sagar, chitesh'

import result_cache

'''
Voting based part of speech tagger which uses viterbi, naive bayes and max-marginal algorithms.
Viterbi has given weight 3 while other two algorithms have weight 2 each.

The result of viterbi algorithm is returned if the output of all three algorithms matches or output of none matches
If naive bayes and max marginal gives same output then it is considered over the output of viterbi. The algorithm
performs better when viterbi and max marginal performs better for different sentences.
'''

def get_part_of_speech(sentence):
    pos = []
    naive_result = result_cache.naive_result
    mm_result = result_cache.max_marginal
    viterbi_result = result_cache.viterbi_result

    for index in range(0, len(sentence)):
        v_result = viterbi_result[index]
        m_result = mm_result[index]
        n_result = naive_result[index]
        if v_result != n_result and n_result == m_result:
            v_result = m_result

        pos.append(v_result)

    return [[pos], []]
