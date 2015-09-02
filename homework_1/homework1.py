# Digit Recognizer

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
    
data_set = parse_data('./train.csv')

# display_digit(data, 1)




