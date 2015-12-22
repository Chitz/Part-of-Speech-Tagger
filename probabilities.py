from __future__ import division

__author__ = 'sagar, chitesh'

'''
 Class holds all the probabilities matrices and maps. USed by all other modules for calculations.
 Performs all the operations from the probabilities
'''
from collections import Counter, defaultdict
import logging
import re
from constants import Constants

# set debug level
logging.basicConfig(level=logging.INFO)


class Probabilities:
    word_prob = Counter()  # P(W|S)
    speech_prob = Counter()  # P(S)w
    speech_rel_prob = Counter()  # P(Si+1|Si)
    total_speech_count = 0  # total occurrences of speech in train data
    unknown_words = defaultdict(str)  # stores new words and best possible speech for them
    default_unknown_prob = 1.0
    total_sentences = 0
    suffix_counter = Counter()  # suffixes of words and their counter

    # Best - MEMM
    word_prev_speech_prob = defaultdict(Counter)
    word_next_speech_prob = defaultdict(Counter)
    word_count = Counter()

    @staticmethod
    def train_from_data(data):
        # data is in the form <word, part_of_speech><word, part_of_speech>
        # Let's parse each line one by one and calculate  P(W|S), P(S), and  P(Si+1|Si)
        # positions of words and tags in train data
        logging.info("Started working on training data, let's see what we have got!")
        WORD_POS = 0
        SPEECH_POS = 1
        Probabilities.total_speech_count = 2  # start and end symbols
        for line in data:
            # number of sentences
            Probabilities.total_sentences += 1
            # (PS1) probability of first part of speech
            Probabilities.speech_rel_prob[line[SPEECH_POS][0] + Constants.delimiter + Constants.first_word] += 1
            # (PSn) probability of last part of speech
            Probabilities.speech_rel_prob[Constants.last_word +
                                          Constants.delimiter + line[SPEECH_POS][len(line[0]) - 1]] += 1
            previous_speech = ""
            previous_word = ""
            for j in range(0, len(line[WORD_POS])):
                word = line[WORD_POS][j]
                speech = line[SPEECH_POS][j]

                # transition probabilities
                if previous_speech:
                    tran_key = speech + Constants.delimiter + previous_speech
                    Probabilities.speech_rel_prob[tran_key] += 1

                previous_speech = speech

                # total number of words in the train data
                Probabilities.total_speech_count += 1

                # increase count for corresponding speech
                Probabilities.speech_prob[speech] += 1

                # create unique key in the form word+speech e.g. <the,det> -> thedet to save the counts for it
                # increase count for corresponding word-speech
                key = word + Constants.delimiter + speech
                Probabilities.word_prob[key] += 1

                # separate last 3 letters as suffix from data and maintain counter for it
                if len(word) > 2:
                    suffix = word[-3:]
                    Probabilities.suffix_counter[suffix + Constants.delimiter + speech] += 1

                    # Best - MEMM
                # P(word/prev-speech)
                Probabilities.word_prev_speech_prob[word][previous_speech] += 1

                if previous_word:
                    # P(word/next-speech)
                    Probabilities.word_next_speech_prob[previous_word][speech] += 1

                previous_word = word

                Probabilities.word_count[word] += 1

        Probabilities.apply_smoothing()

        logging.info("Finished learning from data. Total words : " + str(Probabilities.total_speech_count) + ". " +
                     "whoa! nice training set. " if Probabilities.total_speech_count > 50000 else " hmm, I know a better "
                                                                                                  "training set.")

    # Updates the transition probabilities to avoid case of zero probability. Also converts counts to the
    # corresponding probability
    @staticmethod
    def apply_smoothing():
        speeches = Probabilities.speech_prob.keys()
        for current in speeches:
            Probabilities.speech_rel_prob[current + Constants.delimiter + Constants.first_word] += 1
            Probabilities.speech_rel_prob[Constants.last_word + Constants.delimiter + current] += 1
            Probabilities.speech_prob[current] += 2
            Probabilities.total_speech_count += 2
            for previous in speeches:
                Probabilities.speech_rel_prob[current + Constants.delimiter + previous] += 1
                Probabilities.speech_prob[current] += 1
                Probabilities.speech_prob[previous] += 1
                Probabilities.total_speech_count += 2

        Probabilities.convert_word_speech_probabilities()
        Probabilities.convert_transition_probabilities()
        Probabilities.convert_speech_probabilities()
        Probabilities.default_unknown_prob = 0.5 / Probabilities.total_speech_count

    # convert count(words|speech) to P(words|speech)
    @staticmethod
    def convert_word_speech_probabilities():
        for word_speech in Probabilities.word_prob.keys():
            speech = word_speech.split(Constants.delimiter)[1]
            Probabilities.word_prob[word_speech] /= Probabilities.speech_prob[speech]

    # convert count(speech) to P(speech)
    @staticmethod
    def convert_speech_probabilities():
        for speech in Probabilities.speech_prob.keys():
            Probabilities.speech_prob[speech] /= Probabilities.total_speech_count

    # convert count(Si+1|Si) to P(Si+1|Si)
    @staticmethod
    def convert_transition_probabilities():
        for key in Probabilities.speech_rel_prob.keys():
            speech = key.split(Constants.delimiter)
            # if probability of coming first or last in sentence
            if speech[1] == Constants.first_word or speech[0] == Constants.last_word:
                Probabilities.speech_rel_prob[key] /= Probabilities.total_sentences
            else:
                Probabilities.speech_rel_prob[key] /= Probabilities.speech_prob[speech[0]]

    # returns P(Si+1|Si)
    @staticmethod
    def get_transition_prob(current, previous):
        key = current + Constants.delimiter + previous
        return Probabilities.speech_rel_prob[key]

    # P(S1)
    @staticmethod
    def get_first_speech_prob(speech):
        return Probabilities.get_transition_prob(speech, Constants.first_word)

    # P(Sn)
    @staticmethod
    def get_last_speech_prob(speech):
        return Probabilities.get_transition_prob(Constants.last_word, speech)

    # returns P(word|speech)
    @staticmethod
    def get_word_probability(word, speech):
        key = word + Constants.delimiter + speech
        prob = Probabilities.word_prob[key]

        # word is seen first time for this speech
        if prob == 0:
            # check if its totally new word
            best_speech = Probabilities.unknown_words[word]
            if not best_speech:
                # yes, its completely new word
                prob = Probabilities.default_unknown_prob
                # if heuristically calculated speech matches current speech
            elif best_speech == speech:
                prob = 1.0
            else:
                prob = Probabilities.default_unknown_prob

        return prob

    @staticmethod
    def get_naive_word_probability(word, speech):
        key = word + Constants.delimiter + speech
        prob = Probabilities.word_prob[key]
        return prob

    @staticmethod
    def get_posterior_word_probability(word, speech):
        key = word + Constants.delimiter + speech
        prob = Probabilities.word_prob[key]
        if prob == 0:
            prob = Probabilities.default_unknown_prob
        return prob

    @staticmethod
    def get_best_possible_speech(word):
        speech = Constants.default_speech_tag

        if Probabilities.contains_digit(word):
            speech = Constants.number_tag
        elif word.endswith("ly"):
            speech = Constants.adverb_tag
        elif word.endswith("-like"):
            speech = Constants.adjective_tag
        elif word.endswith("ed"):
            speech = Constants.verb_tag
        # Check suffix of word to calculate best possible speech
        elif len(word) > 2:
            max_speech = ""
            max_speech_cnt = 0
            total_cnt = 0
            for s in Probabilities.speech_prob.keys():
                count = Probabilities.suffix_counter[word[-3:] + Constants.delimiter + s]
                total_cnt += count
                if max_speech_cnt < count:
                    max_speech_cnt = count
                    max_speech = s

            # if 60% of time same speech comes for given suffix, use it
            if max_speech and max_speech_cnt / total_cnt > 0.6:
                speech = max_speech

        Probabilities.unknown_words[word] = speech
        return speech

    @staticmethod
    def contains_digit(word):
        return bool(re.compile("\d").search(word))

    @staticmethod
    # Best -  MEMM
    def convert_word_seq_speech_probabilities():
        for word in Probabilities.word_count.keys():
            for prev_speech in Probabilities.speech_prob.keys():
                Probabilities.word_prev_speech_prob[word][prev_speech] = (Probabilities.word_prev_speech_prob[word][
                                                                              prev_speech] + 1) / \
                                                                         Probabilities.word_count[word]

            for next_speech in Probabilities.speech_prob.keys():
                Probabilities.word_next_speech_prob[word][next_speech] = (Probabilities.word_next_speech_prob[word][
                                                                              next_speech] + 1) / \
                                                                         Probabilities.word_count[word]

    @staticmethod
    def get_prev_word_speech_probability(word, speech):
        prob = Probabilities.word_prev_speech_prob[word][speech]
        return prob

    @staticmethod
    def get_next_word_speech_probability(word, speech):
        prob = Probabilities.word_next_speech_prob[word][speech]
        return prob

    @staticmethod
    def convert_algo_results(results, sentence):
        previous_speech = ""
        for wordIndex in range(len(sentence)):
            word = sentence[wordIndex]
            if Probabilities.word_count[word] == 0:
                for resultIndex in range(len(results)):
                    if wordIndex != 0:
                        previous_speech = results[resultIndex][wordIndex - 1]
                    if len(sentence) != 1 and wordIndex != len(sentence) - 1:
                        next_speech = results[resultIndex][wordIndex + 1]
                    if wordIndex == 0:
                        Probabilities.word_prev_speech_prob[word][next_speech] += 1
                    elif wordIndex == len(sentence):
                        Probabilities.word_next_speech_prob[word][previous_speech] += 1
                    else:
                        Probabilities.word_next_speech_prob[word][previous_speech] += 1
                        Probabilities.word_prev_speech_prob[word][next_speech] += 1

                for next in Probabilities.speech_prob.keys():
                    Probabilities.word_next_speech_prob[word][next] = (Probabilities.word_next_speech_prob[word][
                                                                           next] + 1) / (len(results) + 1)

                for prev in Probabilities.speech_prob.keys():
                    Probabilities.word_prev_speech_prob[word][prev] = (Probabilities.word_next_speech_prob[word][
                                                                           prev] + 1) / (len(results) + 1)
