# Digit Recognizer

import numpy as np
import scipy as sp
import sys
import csv

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
                digits.append(np.matrix(pixels))
        finally:
            input.close()
            return (labels, digits)
    except IOError:
        print "Error - please check filename and try again.\n"
        sys.exit(1)


# TODO Write a function to display MNIST digit
def display_digit (pixel_matrix):
    # USE SCIPY TOIMAGE

data_set = parse_data('./train.csv')