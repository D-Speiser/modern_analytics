# Digit Recognizer
from matplotlib import pylab as plt
import numpy as np
import sys
import csv
import scipy.misc
import matplotlib.image

TRAIN_PATH = './train.csv'
TEST_PATH = './test.csv'

# Parse dataset
def parse_data(file_path):
    try:
        input = open(file_path)
        labels, digits = [], []
        try:
            reader = csv.reader(input)
            for row in reader:
                labels.append(row[0])
                pixels = [row[i:i+28] for i in range(1, len(row), 28)]
                digits.append(np.array(pixels))
        finally:
            input.close()
            return {'labels': map(int, labels[1:]), 'digits': digits[1:]}
    except IOError:
        print "Error - please check filename and try again.\n"
        sys.exit(1)

# B: Save MNIST digits
def save_digits(data_set):
    digit_flag = np.zeros(10) # initialize boolean flag array
    first_instance_idx = {}
    for idx, label in enumerate(data_set['labels']):
        if digit_flag[label] == False:
            matplotlib.image.imsave('digit_' + str(label), data_set['digits'][idx])
            digit_flag[label] = True
            first_instance_idx[label] = idx
    return first_instance_idx # used for part D
# C: return frequency of digits
def get_digit_frequencies(digits):
    frequencies = np.zeros(10)
    for i in range(len(digits)):
        frequencies[digits[i]] += 1

    frequency_dict = dict(zip(range(10), frequencies / sum(frequencies))) # / by sum normalizes
    return frequency_dict
# C: build normalized histogram
def build_normalized_histogram(data_set):
    frequency_dict = get_digit_frequencies(data_set['labels'])
    fig = plt.figure()
    plt.hist(frequency_dict.keys(), weights=frequency_dict.values())
    plt.title('Normalized Histogram'), plt.xlabel('Digit #'), plt.ylabel('Frequency'), plt.gca().set_xlim([0, 9]) # set labels, titles, and x range
    fig.savefig('histogram.png')

# train_set, test_set = parse_data(TRAIN_PATH), parse_data(TEST_PATH) # UNCOMMENT ATFER completion. ALSO ADD condition to parse_data method, as input structure varies
train_set = parse_data(TRAIN_PATH)
digit_instance_idx = save_digits(train_set)
build_normalized_histogram(train_set)
