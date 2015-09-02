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
            return {'labels': labels, 'digits': digits}
    except IOError:
        print "Error - please check filename and try again.\n"
        sys.exit(1)

# Write a function to display MNIST digit
def save_image (pixel_matrix, index):
    index += 1
    matplotlib.image.imsave('digit_' + str(index), data)
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
build_normalized_histogram(data_set)

# display_digit(data, 1)
