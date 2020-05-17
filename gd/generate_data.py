# ------------------------------------------------------------------------------
# Example of generating a geometric dataset
# ------------------------------------------------------------------------------

import csv
import os
from gd.distributions.sampling import sample_normal, sample_uniform

dataset_path = r'storage/ellipsis/'
os.makedirs(dataset_path)
img_shape=(64, 64)
nr_observations = 100
nr_dummy_total = 1*nr_observations

#  Sampling and creation of csv files
# With these parameters, I get an 'empty ratio of 0.12'
big_length_long = sample_normal(mean=0.5, std=0.0, min_value=0, max_value=1, nr_samples=nr_dummy_total)
big_length_short = sample_normal(mean=0.4, std=0.0, min_value=0, max_value=1, nr_samples=nr_dummy_total)
big_x_start = sample_normal(mean=0.25, std=0.1, min_value=0, max_value=1, nr_samples=nr_dummy_total)
big_y_start = sample_normal(mean=0.25, std=0.1, min_value=0, max_value=1, nr_samples=nr_dummy_total)
big_angle = sample_uniform(min_value=0, max_value=1, nr_samples=nr_dummy_total)
small_length_long = sample_normal(mean=0.3, std=0.05, min_value=0, max_value=1, nr_samples=nr_observations)
small_length_short = sample_normal(mean=0.2, std=0.05, min_value=0, max_value=1, nr_samples=nr_observations)
small_x_start = sample_normal(mean=0.25, std=0.0, min_value=0, max_value=1, nr_samples=nr_observations)
small_y_start = sample_normal(mean=0.25, std=0.0, min_value=0, max_value=1, nr_samples=nr_observations)
small_angle = sample_uniform(min_value=0, max_value=1, nr_samples=nr_observations)

with open(os.path.join(dataset_path, 'small_ellipsis.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['length_long', 'length_short', 'x_start', 'y_start', 'angle'])
    for ix in range(nr_observations):
        length_long = max(small_length_long[ix], small_length_short[ix])
        length_short = min(small_length_long[ix], small_length_short[ix])
        row = [length_long, length_short, small_x_start[ix], small_y_start[ix], small_angle[ix]]
        writer.writerow(row)

with open(os.path.join(dataset_path, 'big_ellipsis.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['length_long', 'length_short', 'x_start', 'y_start', 'angle'])
    for ix in range(nr_dummy_total):
        length_long = max(big_length_long[ix], big_length_short[ix])
        length_short = min(big_length_long[ix], big_length_short[ix])
        row = [length_long, length_short, big_x_start[ix], big_y_start[ix], big_angle[ix]]
        writer.writerow(row)

# Open csv files
with open(os.path.join(dataset_path, 'small_ellipsis.csv'), 'r') as f:
    small_data = list(csv.reader(f))[1:]
    small_data = [[float(x) for x in row] for row in small_data if row]
with open(os.path.join(dataset_path, 'big_ellipsis.csv'), 'r') as f:
    big_data = list(csv.reader(f))[1:]
    big_data = [[float(x) for x in row] for row in big_data if row]
assert len(small_data) == nr_observations
assert len(big_data) == nr_dummy_total

# Create images
from gd.generate_dataset_imgs import Ellipsis
data = Ellipsis(small_data, big_data)
