__author__ = 'ctewani'
from probabilities import Probabilities
from math import ceil
from constants import Constants
from collections import Counter, defaultdict
from random import random
import result_cache

'''
MCMC with Gibbs - POS tagger

Step 3: Sampling -


In this step, we are generating large number of samples for the sentence as sequence of parts of speech
using Gibbs Sampler. We have used burn-in time around half of the total number of samples generated from this step

(1) Description -

We perform Gibbs sampling in following steps -
1. Generate initial samples, i.e assign the sentence random parts of speech - This step is done only once
2. For large number of samples
    2.a. Use the previous sample [First time, use the initial sample generated]
    2.b. Perform Gibbs Sampling
        2.b.1. For random walks over the previous sample [Any order of the sample, regardless]
        2.b.2. Calculate the Marginal distribution for chosen word in the sentence for each POS, having all the other
                parts of speech for the sentence as given.

                [Assumption 1 explained below]

                Formula -
                If first-word in the sentence -
                    P(W1/S1) * P(S1) * P(S2/S1)
                        -- Considering S2 (part of speech for 2nd word), S3, S4.. Sn as the given
                        -- S1, will be part of speech for first word (In our case, 12 parts of speech)
                        -- Marginal distribution calculates the probability of each parts of speech, given all the other
                            parts of speech for the sentence.
                        -- P(W1/S1), calculated in Step 1,
                        -- P(S1), probability of parts of speech occuring at the start of sentnce
                        -- P(S2/S1), transition probability, calculated in step 1

                Else any word in the sentence expect the end -
                    P(Wi/Si) * P(Si/Si-1) * P(Si+1/Si)
                        -- Transition probability from (i-1)th -> ith & ith -> (i+1)th
                        -- Part of speech of previous word and next word is given for this case
                        -- Marginal distribution for the word shows, the probability of the word having that parts of speech
                            [GIVEN]
                            -- the parts of speech for the previous word
                            -- the parts of speech for the next word

                Else word is at the end of the sentence -
                    P(Wn/Sn) * P(Si/Si-1)

        2.b.3. Normalize the marginal distribution for the word for all parts of speech, so their sum equals 1
        2.b.5. Based on the marginal distribution, randomly pick any parts of speech for the word. [Like 12 sided-dice rolled,
                expect a number occuring will be based out of probability]
                2.b.5.1.Find the cummulative sum of the normalized probabilities
                2.b.5.2. Pick a random number between 0 to 1, and use the cummulative sum at each index to get the index
        2.b.6. Assign the picked parts of speech for the word and use this modified value for rest of the calculations
    2.c. Add the sample in the Map, maintaining the count of each unique sample [useful in max_marginal]

3. From the samples generated, perform BURN-IN/WARM-UP, throw initial samples from the list
4. From the list of remaining samples, randomly pick sample_count samples, 5 in our case, and return

-- Results of the evaluation on the bc.test file

    - We get a percentage of
        - 94.38% for Word correctness
        - 50.97% for sentence correctness
    - The above results sometimes generates similar samples, sometimes unique,
        - If we pick only unique top 5 (ranking by their posterior probability of unique samples) from the list
        - We get a percetage of
            - 92.43% for Word correctness
            - 47.24% for Sentence correctness
        - As we have randomly sample, we do not return the unique sample, instead let it pick randomly from the list,
        to perform sampling

(2) Problem faced -

1. Initially, we were using NUMPY library to randomly pick POS based on marginal distribution as weighted probabilities,
we ARE NOT using it anymore, as it was informed to us that it is not allowed

Using cumulative sum, we are picking randomly. Happy to learn.

(3) Assumptions -

1. In Gibbs sampling steps, we are not considering ALL the parts of speech for other words in the sequence, just
 - Parts of Speech of Previous word, as given
 - Parts of Speech of Next word, as given

 As the parts of speech of all other words in the senteence far away from the word, would always have
 constant transition probabilities for them to be multiplied, as is nullified in normalization step.

 This saves commutation of all multiplying all other transition probabilities.

(4) Simplifications -

1. We are using the samples generated in MCMC step 3, for Step 4, as it takes a lot of computation time to genrate 1000 of samples

(5) Design decisions -

1. Data structure to store the unique samples, (key as concat of POS), which helps in Max Marginal Step 4

'''


def generateInitSamples(sentence):
    # use result from naive bayes to start with
    initSample = result_cache.naive_result
    return initSample


def pick_samples(sampleMap, sample_count, sentence):
    max_sample_map = defaultdict(Counter)
    for sampleKey in sampleMap.keys():
        sample = sampleKey.split('_')
        max_sample_map[0][sample[0]] += sampleMap[sampleKey]
        for i in range(1, len(sentence)):
            max_sample_map[i][sample[i]] += sampleMap[sampleKey]

    return max_sample_map


def get_samples(sentence, sample_count):
    ''' Gibbs sampler
    1. Generate initial samples, i.e. Assign the sentence random Parts of Speech [uniform or random or EM, not sure]
    1.1 Optional Burn-in or thinning?, throw away few samples. [TO-DO]
    Repeat sample_count times
        2. Pick the last sample, x[t] <- x[t-1]
        3. Repeat the following steps for all unobserved/non-evidence words
        4. For each word picked, sample it by calculating the posterior probability keeping all other variable as evidence
    5. Add to sample list
    '''
    sampleMap = Counter()
    samples = []
    # step 1 - generate initial samples
    initSample = generateInitSamples(sentence)
    previousSample = initSample

    for i in range(1, Constants.gibbs_max_iteration):
        # step 2 - pick last sample
        modifiedSample = previousSample
        # step 3
        for j in range(0, len(sentence)):
            speechTags = []
            probWeights = []
            sumProbWeights = 0
            # step 4, calculate the posterior probability
            for speech in Probabilities.speech_prob.keys():
                word_prob = Probabilities.get_word_probability(sentence[j], speech)

                if len(sentence) == 1:
                    prob1 = Probabilities.get_first_speech_prob(speech)
                    prob = word_prob * prob1
                elif j == 0:  # first word, nothing prior
                    prob1 = Probabilities.get_first_speech_prob(speech) * Probabilities.get_transition_prob(
                        modifiedSample[j + 1], speech)
                    prob = word_prob * prob1
                elif j == len(sentence) - 1:
                    prob1 = Probabilities.get_transition_prob(speech, modifiedSample[j - 1])
                    prob = word_prob * prob1
                else:
                    prob1 = Probabilities.get_transition_prob(speech, modifiedSample[
                        j - 1]) * Probabilities.get_transition_prob(modifiedSample[j + 1], speech)
                    prob = word_prob * prob1
                sumProbWeights += prob
                speechTags.append(speech)
                probWeights.append(prob)

            probWeights = [x / sumProbWeights for x in probWeights]

            # cummulative sum
            cumsum = 0
            randomWeight = random()
            for i in range(len(probWeights)):
                cumsum += probWeights[i]
                probWeights[i] = cumsum

            randomIndex = -1
            for i in range(1, len(probWeights)):
                if probWeights[i] >= randomWeight and randomWeight >= probWeights[i - 1]:
                    randomIndex = i

            if randomIndex == -1:
                randomIndex = 0

            modifiedSample[j] = speechTags[randomIndex]

        previousSample = modifiedSample[:]
        samples.append(previousSample)
        sampleMap['_'.join(previousSample)] += 1

    samples = samples[Constants.gibbs_burn_in_count:]
    return samples, sampleMap
