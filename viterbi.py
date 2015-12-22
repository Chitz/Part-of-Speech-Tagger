__author__ = 'sagar, chitesh'
from probabilities import Probabilities
import result_cache
import sys

'''
Dynamic programming based part of speech tagger using viterbi algorithm.
Formuls:
(s1,....sN) = arg max s1,...,sN P(Si = si|W)
that is,
P(S|W) = max( P(W|S) * max P(S|W-1) * P(Si-i|Si) )

In the formula, we are computing the solution for next word using solution of the previous. That is intuition for
dynamic programming based solution.

'''


def get_part_of_speech(sentense):
    # all the speeches from the train data
    speeches = Probabilities.speech_prob.keys()
    no_words = len(sentense)
    no_speech = len(speeches)

    # Holds backtrack path for trace back
    back_tracks = [[0] * (no_words + 1) for x in range(no_speech)]

    # maximum probabilities for each speech at each level
    max_probabilities = [[0] * (no_words + 1) for x in range(no_speech)]

    # initial probability calculation for first word
    first_word = sentense[0]

    # basic step
    for i in range(0, no_speech):
        max_probabilities[i][0] = Probabilities.get_first_speech_prob(speeches[i]) * Probabilities.get_word_probability(
            first_word, speeches[i])
        back_tracks[i][0] = ""

    # recursive step
    for word_index in range(1, no_words):
        curr_word = sentense[word_index]
        for tag_index in range(0, no_speech):
            arg_max = -sys.maxint
            arg_bt = ""
            max_total_prob = -sys.maxint
            curr_tag = speeches[tag_index]
            word_prob = Probabilities.get_word_probability(curr_word, curr_tag)
            for prev_tag_index in range(0, no_speech):
                prev_tag = speeches[prev_tag_index]

                # calculate transition probability
                transition_prob = Probabilities.get_transition_prob(curr_tag, prev_tag) * max_probabilities[
                    prev_tag_index][word_index - 1]

                if transition_prob > arg_max:
                    arg_max = transition_prob
                    arg_bt = prev_tag

                # total probability
                total_prob = transition_prob * word_prob

                if total_prob > max_total_prob:
                    max_total_prob = total_prob

            back_tracks[tag_index][word_index] = arg_bt
            max_probabilities[tag_index][word_index] = max_total_prob

    # terminal step, calculate speech for last word
    max_probabilities[no_speech - 1][no_words] = -1

    for i in range(0, no_speech):
        tag = speeches[i]
        last_prob = max_probabilities[i][no_words - 1] * Probabilities.get_last_speech_prob(tag)

        if max_probabilities[no_speech - 1][no_words] < last_prob:
            max_probabilities[no_speech - 1][no_words] = last_prob
            back_tracks[no_speech - 1][no_words] = tag

    # backtrack, get best path
    last_tag = back_tracks[no_speech - 1][no_words]

    solution = [last_tag]

    for word_index in range(no_words - 1, 0, -1):
        prev_tag_index = speeches.index(last_tag)
        last_tag = back_tracks[prev_tag_index][word_index]
        solution.append(last_tag)

    # Due to backtrack it will be in reverse, change it
    solution.reverse()

    # store result in result cache for future use
    result_cache.viterbi_result = solution[:]

    return [[solution], []]
