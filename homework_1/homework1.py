# Digit Recognizer
from sklearn.neighbors import NearestNeighbors as kNN
from matplotlib import pylab as plt, image as img
import numpy as np
import scipy as sp
import sys
import csv

TRAIN_PATH = './train.csv'
TEST_PATH = './test.csv'

# Parse dataset
def parse_data(file_path):
    try:
        input = open(file_path)
        labels, raw_pixels = [], []
        try:
            reader = csv.reader(input)
            for row in reader:
                if reader.line_num > 1:
                    labels.append(row[0])
                    raw_pixels.append(np.array(map(int, row[1:])))
                    # Currently not needed. Slows parse time by ~3x
                    # pixels = [map(int, row[i:i+28]) for i in range(1, len(row), 28) if reader.line_num > 1]
                    # digits.append(np.array(pixels))
                
        finally:
            input.close()
            return {'labels': map(int, labels), 'pixels': raw_pixels}
    except IOError:
        print "Error - please check filename and try again.\n"
        sys.exit(1)

# B: Save MNIST digits
def save_digits(data_set):
    digit_flag = np.zeros(10) # initialize boolean flag array
    first_instance_idx = {}
    for idx, label in enumerate(data_set['labels']):
        if digit_flag[label] == False:
            matrix = data_set['pixels'][idx].reshape(28, 28) # reshape 1d array into 2d, size 28x28
            img.imsave('digit_' + str(label), matrix, cmap="gray_r")
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

def find_best_matches(dig_inst_idx, data_set):
    digits, pixels = data_set['labels'], data_set['pixels']
    # plot_nearest_matches(distance_array)
    best_fits = [(sys.maxint, -1)] * 10 # best fit array containing a tuple (distance, idx) for each digit
    for idx, dig_pix in enumerate(pixels): # for each matrix of pixels
        if idx != dig_inst_idx[digits[idx]]: # if not first instance of this digit
            L2 = sp.spatial.distance.euclidean(pixels[dig_inst_idx[digits[idx]]], dig_pix) # Distance = euclidean(first_instance_matrix, pixel_matrix[idx])
            if L2 < best_fits[digits[idx]][0]: # if current distance < current smallest distance
                best_fits[digits[idx]] = (L2, idx) # set new distance as smallest 
    save_best_fits(data_set, best_fits)


def save_best_fits(data_set, best_fits):
    for idx, fit in enumerate(best_fits):
        matrix = data_set['pixels'][fit[1]].reshape(28, 28) # reshape 1d array into 2d, size 28x28
        img.imsave('best_fit_' + str(idx), matrix, cmap="gray_r")


# train_set, test_set = parse_data(TRAIN_PATH), parse_data(TEST_PATH) # UNCOMMENT ATFER completion. ALSO ADD condition to parse_data method, as input structure varies
train_set = parse_data(TRAIN_PATH)
digit_instance_idx = save_digits(train_set)
build_normalized_histogram(train_set)
find_best_matches(digit_instance_idx, train_set)
