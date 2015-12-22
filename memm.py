__author__ = 'ctewani'

'''
Alternate attempt to best algorithm - 

We tried to implement Greedy based Maximum Entrophy Markov Model (MEMM) 

1. We came across this model - http://nlp.stanford.edu/courses/cs224n/2011/reports/highfill.pdf and read more about the model
2. This is inverse of HMM and uses MaxEnt entrophy 
3. Assumption for the model, instead of finding P(S/W), posterior probability as in case of HMM, we calculate the word probability for the speech, next to it and previous to it.
E.g. It is a dog
    NOUN PRN DET ADJ
    
    We calculate the probability, as word which is preceeds a speech and follows teh speech. 
    So all we will know, that "is" follows NOUN and "is" preceeds DET..
    
    We calculate all teh probalities of the words which are next to the parts of speecha nd preceeds.
    
4. we use, for each 
argmax over parts of speech [P(S1/W1) * P(S1/W2) * P(S2/W2) * P(S2/W1) * P(S2/W3)) ....] 

5. Using the directional model where each word influences the tag next to it and preceeding it
6. The results from this is 90.83% for word correctness
'''

from probabilities import Probabilities
import sys
import result_cache


def get_part_of_speech(sentence):
    # print result_cache.results
    result_cache.results = [result_cache.naive_result] + [result_cache.max_marginal] + [result_cache.viterbi_result]
    Probabilities.convert_algo_results(result_cache.results, sentence)
    pos = []
    previous_word = ""
    for index in range(len(sentence)):
        best_prob = - sys.maxint

        for speech in Probabilities.speech_prob.keys():
            word = sentence[index]
            prob = Probabilities.get_word_probability(word, speech)
            if index == 0:
                next_word = sentence[index + 1]
                prob *= Probabilities.get_next_word_speech_probability(next_word,
                                                                       speech) * Probabilities.get_first_speech_prob(
                    speech)
            elif len(sentence) == 1:
                prob *= Probabilities.get_first_speech_prob(speech)
            elif index == len(sentence) - 1:
                prob *= Probabilities.get_transition_prob(speech,
                                                          previous_speech) * Probabilities.get_prev_word_speech_probability(
                    previous_word, speech)
            else:
                next_word = sentence[index + 1]

                prob *= Probabilities.get_transition_prob(speech,
                                                          previous_speech) * Probabilities.get_prev_word_speech_probability(
                    previous_word, speech) * Probabilities.get_next_word_speech_probability(next_word, speech)

            previous_speech = speech
            previous_word = word

            if best_prob < prob:
                best_prob = prob
                best_speech = speech

        pos.insert(index, best_speech)

    return pos
