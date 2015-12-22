__author__ = 'sagar'

# Holds results of pos tagging algorithms for later use by best algorithm.

# Results are kept for just last sentence assuming the current architecture where each sentence is being tested for
# all algorithms one by one, best being last.

naive_result = []
viterbi_result = []
max_marginal = []
max_sample_map = {}
results = []