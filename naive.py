__author__ = 'sagar, chitesh'
from probabilities import Probabilities
import result_cache

'''
 POS tagger using Naive-Bayes Inference

    Uses nave bayes formula to calculate best possible tag for given word.

    P(S|W) = P(W|S) * P(S)

    Si = arg max si P(Si = si|W):
'''


def get_part_of_speech(sentence):
    pos = []
    for word in sentence:
        best_prob = 0
        best_speech = ""
        for speech in Probabilities.speech_prob.keys():
            word_prob = Probabilities.get_naive_word_probability(word, speech) * Probabilities.speech_prob[speech]

            # choose best possible tag for speech
            if word_prob > best_prob:
                best_prob = word_prob
                best_speech = speech

        # choose best speech by heuristic if no occurrences found
        if best_prob == 0:
            best_speech = Probabilities.get_best_possible_speech(word)

        pos.append(best_speech)

        # store result in result cache for future use
        result_cache.naive_result = pos

    return [[pos], []]
