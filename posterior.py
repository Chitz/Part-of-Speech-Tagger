__author__ = 'sagar, ctewani'
from probabilities import Probabilities
from math import log

'''
We are using the formula -

Log(P(W1/S1) * P(S1) * P(W2/S2) * P(S2/S1) .. P(Wn/Sn) * P(Wn-1/Sn-1))

 -- P(S1) -> Probability of Speech S1 being the first speech in a given sentence, calculated in step 1
 -- P(Wx/Sx) -> Calulcated in step 1, probability of word, given speech
 -- P(Si+1/Si) -> Markov Model, transition probability of next speech given previous speech


'''


def calculate_posterior(sentence, label):
    posterior_prob = 1.0
    for index in range(0, len(sentence)):

        posterior_prob *= Probabilities.get_posterior_word_probability(sentence[index], label[index])

        if index == 0:
            posterior_prob *= Probabilities.get_first_speech_prob(label[index])
        else:
            posterior_prob *= Probabilities.get_transition_prob(label[index], label[index - 1])

    return log(posterior_prob)
