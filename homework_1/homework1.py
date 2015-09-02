# Digit Recognizer
from matplotlib import pylab as plt
import numpy as np
import sys
import csv
import scipy.misc
import matplotlib.image

# Parse dataset
def parse_data (file_path):
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
            return {'labels': labels[1:], 'digits': digits[1:]}
    except IOError:
        print "Error - please check filename and try again.\n"
        sys.exit(1)

# Save MNIST digits
def save_digits (data_set):
    digit_flag = np.zeros(10) #initialize boolean flag array
    
    index = 0
    for label in data_set['labels']:
        if digit_flag[int(label)] == False:
            matplotlib.image.imsave('digit_' + label, data_set['digits'][index])
            digit_flag[int(label)] = True
            print label, " at index ", index # for debugging
        index += 1
    return

def get_digit_frequencies(digits):
    frequencies = np.zeros(10)

    for i in range(len(digits)):
        frequencies[int(digits[i])] += 1

    frequency_dict = dict(zip(range(10), frequencies / sum(frequencies))) # / by sum normalizes
    return frequency_dict

def build_normalized_histogram(data_set):
    frequency_dict = get_digit_frequencies(data_set['labels'][1:]) # hardcoded removal of first 'label', fix within parser
    plt.hist(frequency_dict.keys(), weights=frequency_dict.values())
    plt.show()

data_set = parse_data('./train.csv')

save_digits(data_set)
build_normalized_histogram(data_set)