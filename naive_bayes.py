## import packages
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

# Library to navigate into directory
import os


def gaussian_distribution(mean, sigma):

	if sigma == 0:
		result = [0 for i in range(0, 256)]
		result[int(mean)] = np.inf
		return result
	else:
		return [(1/math.sqrt(2*math.pi*(sigma**2)))*(math.exp((-(X-mean)**2)/(2*(sigma**2)))) for X in range(0, 256)]


# Create memory to store every normal distributions
memory = [[[[] for i in range(28)] for i in range(28)] for i in range(10)]

# Put our cursor in the folder 0 in order to be in the good folder
os.chdir('mnist_png/training/0')

# For each digit execute the following code
for digit in range(10):

	print("Creating the model of the digit {}".format(digit), end='\r')

	# Go to folder of the concerned digit
	os.chdir('../{}'.format(digit))

	image_digit = os.listdir()

	# For each file
	for file in image_digit:

		# Convert file into array
		img = np.array(Image.open(file))

		# for each line
		for idxLine, line in enumerate(img):

			# for each pixel in line
			for idxPixel, pixel in enumerate(line):

				memory[digit][idxLine][idxPixel].append(pixel)


	# Treat memory for the digit
	for idxLine, line in enumerate(memory[digit]):
		for idxPixel, pixel in enumerate(line):

			current_array = np.array(memory[digit][idxLine][idxPixel])
			memory[digit][idxLine][idxPixel] = gaussian_distribution(np.mean(current_array), np.std(current_array))

print('')

# ---- TEST PART -----


os.chdir('../../testing/0')

accuracy = []

correct_digit_counter = [0 for i in range(10)]

for digit_folder in range(10):

	# Go to folder of the concerned digit
	os.chdir('../{}'.format(digit))

	image_test_digit = os.listdir()

	for file in image_test_digit:

		# file = "../../../test.png"

		image_test = np.array(Image.open(file))

		max_score = -np.inf
		correct_digit = None

		for digit in range(10):

			score = np.log(0.1)

			for idxLine, line in enumerate(image_test):
				for idxPixel, pixel in enumerate(line):

					likelihood = memory[digit][idxLine][idxPixel][image_test[idxLine, idxPixel]]
					log_likelihood = np.log(likelihood)
					score += log_likelihood

			if score > max_score:
				max_score = score
				correct_digit = digit

		if correct_digit == digit_folder:
			accuracy.append(1)
			correct_digit_counter[digit_folder] += 1
		else:
			accuracy.append(0)

		print("Testing the folder of the digit {} --> {}/{}".format(digit_folder, correct_digit_counter[digit_folder], len(image_test_digit)), end='\r')

	print()

print(np.sum(accuracy))
print(len(accuracy))
print(np.sum(accuracy)/len(accuracy)*100)

