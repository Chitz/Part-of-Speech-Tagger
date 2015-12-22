# !/usr/bin/python
__author__ = 'sagar, chitesh'
###################################
# CS B551 Fall 2015, Assignment #5
#
# sagar bhandare sagabhan
# chitesh tiwani ctiwani
#
# (Based on skeleton code by D. Crandall)
#
#
####
#
'''
Brief report on part of speech:

Code organization :
Code is nicely arranged in different modules based on functionality :)
     Module             Functionality
Probabilities    Handles all operations related to probability calculation including training from data
Naive            Generates most likely tag sequence using naive bayes inference method
MCMC             Generates samples using Gibb's sampling method
Viterbi          Generates most likely tag sequence using viterbi algorithm
Max_Marginal     Gives most likely tag sequence with approximate max-marginal inference
Posterior        Caculates logarithm of posterior values for given sequence of tags
Best             Voting based algorithm to provide most likely tag sequence
Constants        Holds values of string constants used in the program
Result_Cache     Intermediate storage for results of various algorithms for sharing


Results on bc.test file :

==> So far scored 2001 sentences with 29469 words.
                   Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
          1. Naive:       94.06%               49.08%
        2. Sampler:       94.54%               50.62%
   3. Max marginal:       96.07%               60.37%
            4. MAP:       96.08%               60.77%
           5. Best:       96.10%               61.12%
----

Result analysis:
As from the results MAP(Viterbi), Max marginal and Best algorithm gives results with maximum accuracy. The sentence
accuracy is also significantly greater than naive and sampler.

Assumptions/Design decisions made:

1. Handling of new words for a given speech:
This plays significant role in accuracy of any algorithms. We are deciding the speech of new word based on three
factors.
A. If the word is present in training with some other part of speech, that speech gets more probability and current
speech gets default minimum probability which is probability of single occurrence in entire training data.
B. If the word is completely new, we try to heuristically predict part of speech for that word considering its
suffix. Count of each word's suffix (last 3 letters) is stored in map which is used to predict possible part of
speech of word if the occurrence of speech with given suffix is more than 60%.
C. In any other case minimum default probability is given

2. Handling of zero transition probabilities / Smoothing
All possible transitions for all parts of speech from the training data are given single occurrence by default. The
speech probabilities are updated accordingly to ensure overall ratio remains constant.

3. Result cache is used to share results across the algorithms for each sentence. The order of execution of
algorithms should not be changed to ensure best results for 'best' algorithm.

Apart from this individual file has comments and explanations of corresponding code in it.
'''


####
import logging
import naive
import viterbi
import mcmc
import max_marginal
import posterior
import best
from random import sample
import result_cache
from probabilities import Probabilities

# set debug level
logging.basicConfig(level=logging.INFO)


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        return posterior.calculate_posterior(sentence, label)

    # Do the training!
    #
    def train(self, data):
        # create all probabilities required
        Probabilities.train_from_data(data)

    # Functions for each algorithm.
    #
    def naive(self, sentence):
        return naive.get_part_of_speech(sentence)

    def mcmc(self, sentence, sample_count):
        samples, sampleMap = mcmc.get_samples(sentence, sample_count)
        result_cache.max_sample_map = mcmc.pick_samples(sampleMap, sample_count, sentence)
        return [sample(samples, sample_count), []]

    def best(self, sentence):
        return best.get_part_of_speech(sentence)

    def max_marginal(self, sentence):
        pos, pos_prob = max_marginal.get_part_of_speech(result_cache.max_sample_map)
        return [[pos], [pos_prob]]

    def viterbi(self, sentence):
        return viterbi.get_part_of_speech(sentence)

        # This solve() method is called by label.py, so you should keep the interface the
        #  same, but you can change the code itself.
        # It's supposed to return a list with two elements:
        #
        #  - The first element is a list of part-of-speech labelings of the sentence.
        #    Each of these is a list, one part of speech per word of the sentence.
        #    Most algorithms only return a single labeling per sentence, except for the
        #    mcmc sampler which is supposed to return 5.
        #
        #  - The second element is a list of probabilities, one per word. This is
        #    only needed for max_marginal() and is the marginal probabilities for each word.
        #

    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 5)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algo!"
