import numpy as np
import altair as alt
import pandas as pd
from scipy.stats import expon
import os
import random

import random
import os
from collections import defaultdict

import random
import os
from collections import defaultdict

def load_mnist_images_by_label(dataset_path, max_per_digit=20):
    """
    Load MNIST images from organized folders with an optional cap on the number of images per digit.
    
    Parameters:
    - dataset_path (str): Path to the MNIST dataset.
    - max_per_digit (int, optional): Maximum number of images to keep for each digit.

    Returns:
    - mnist_population (dict): {digit: list of image paths}
    - available_indexes (dict): {digit: set of available indexes}
    """
    mnist_population = {}
    available_indexes = {}

    for split in ["train"]:  # Load from train folder
        for digit in range(10):
            digit_path = os.path.join(dataset_path, split, str(digit))
            if os.path.exists(digit_path):
                images = [os.path.join(digit_path, img) for img in os.listdir(digit_path) if img.endswith(".png")]

                # Limit number of images per digit if specified
                if max_per_digit and len(images) > max_per_digit:
                    images = random.sample(images, max_per_digit)  # Randomly sample max_per_digit images

                mnist_population[digit] = images  # Store paths in a list
                available_indexes[digit] = set(range(len(images)))  # Store available indexes as a set

    return mnist_population, available_indexes



class StochasticProcess:
    def __init__(self, process_type="linear", steps = 15, angular_coef=0.3, rate=0.1, period=None, initial_population=None, combine_with=None, frequency=None):
        self.process_type = process_type
        self.angular_coef = angular_coef
        self.rate = rate
        self.period = period  # Used only for sine process, optional
        self.frequency = frequency  # Frequency for sine process, optional
        self.time = 0
        self.initial_population = initial_population if initial_population is not None else 100  # Default population
        self.steps = steps
        self.combine_with = combine_with  # Processes to combine with the linear process
        self.data = self._compute_process()

    def __str__(self):
        return (f"Stochastic Process (Type: {self.process_type}, "
                f"Initial Population: {self.initial_population}, "
                f"Steps: {self.steps}, "
                f"Rate: {self.rate}, "
                f"Angular Coefficient: {self.angular_coef}, "
                f"Period: {self.period}, "
                f"Frequency: {self.frequency}, "
                f"Combine With: {self.combine_with}, "
                f"Time: {self.time}, "
                f"Data: {self.data})")
        
    def _compute_process(self):
        """
        Return the process data
        """
        if self.time == 0:
            data = [{'time': 0, 'value': self.initial_population}]
            
        for step in range(1, self.steps + 1):
            delta_pop = self.update() 
            data.append({'time': step, 'value': delta_pop if delta_pop > 0 else 0})
        
        # Convert data into a pandas DataFrame
        df = pd.DataFrame(data)

        return df

    
    def update(self):
        self.time += 1
       
        # If it's not the first step, proceed with the normal process calculation
        if self.process_type == "linear":
            return self.linear_process()
        elif self.process_type == "exponential":
            return self.exponential_process()
        elif self.process_type == "sine":
            return self.sine_process()
        elif self.process_type == "uniform":
            return self.uniform_process()
        elif self.process_type == "exp_decay":
            return self.exponential_decay_process()
        elif self.process_type == "combined":
            valid_processes = {"linear", "exponential", "sine", "exp_decay"}
            if self.combine_with is not None:
                invalid_processes = [process for process in self.combine_with if process not in valid_processes]
                if invalid_processes:
                    raise ValueError(f"Invalid process types in combine_with: {', '.join(invalid_processes)}\n Try 'exponential', 'sine' or 'exp_decay'.")
                        
                return self.combined_process()
        else:
            raise ValueError("Unknown process type. Try 'linear', 'exponential', 'sine', 'uniform', 'exp_decay' ou 'combined'.")
    
    def linear_process(self):
        """Linear process increases with a constant rate over time."""
        return int(self.initial_population + self.angular_coef * self.time)
    
    def exponential_process(self):
        """Exponential growth process."""
        return int(self.initial_population * np.exp(self.rate * self.time))
    
    def sine_process(self):
        """Sine wave process with a specified frequency."""
        if self.frequency is None:
            raise ValueError("Frequency must be set for sine process")
        # Scale the sine wave to have an offset based on initial_population
        return int(self.initial_population + self.period * np.sin(self.frequency * self.time))
    
    def uniform_process(self):
        """Uniform process with a range from 0 to the rate."""
        return int(self.initial_population + np.random.uniform(0, self.rate))
    
    def exponential_decay_process(self):
        """Exponential decay process: population decreases over time."""
        return int(self.initial_population * np.exp(-self.rate * self.time))
    
    def combined_process(self):
        """
        Combines linear with one of the other processes.
        For example: linear + exponential, linear + sine, linear + exp_decay
        """
        # Start with the initial population to avoid peak at first step
        result = self.linear_process()
    
        if "exponential" in self.combine_with:
            result += self.exponential_process() - self.initial_population  # Adjust to prevent overlap at step 1
        
        if "sine" in self.combine_with:
            result += self.sine_process() - self.initial_population  # Adjust to prevent overlap at step 1
        
        if "exp_decay" in self.combine_with:
            result += self.exponential_decay_process() - self.initial_population  # Adjust to prevent overlap at step 1
        
        return result

        
    def next_step(self):
        delta_pop = self.update() 
        self.data.loc[len(self.data)] = {'time': self.time, 'value': delta_pop if delta_pop > 0 else 0}
    
    def plot_process(self):
        """
        Plot the stochastic process over a specified number of steps.
        """
        df = self.data
    
        # Create an Altair chart
        chart = alt.Chart(df).mark_line().encode(
            x='time:O',  # ordinal axis for time
            y='value:Q',  # quantitative axis for process value
            tooltip=['time', 'value']
        ).properties(
            title=f"{self.process_type.capitalize()} Process over Time",
            width=300,  # Set the width of the plot
            height=150  # Set the height of the plot
        )
        
        # Display the chart
        return chart



class Sample:
    def __init__(self, label, stochastic_process, mnist_population, available_indexes):
        """
        Initialize the Sample class.
        :param label: The label for the sample (0-9 for MNIST digits).
        :param stochastic_process: A precomputed StochasticProcess instance.
        :param mnist_population: Global MNIST data population (dict with lists of file paths).
        :param available_indexes: Dictionary tracking available indexes for sampling.
        """
        self.label = label
        self.mnist_population = mnist_population  # Dictionary with lists of file paths
        self.available_indexes = available_indexes  # Dictionary with available index sets
        self.process_data = stochastic_process.data["value"].tolist()  # Precomputed sample sizes
        
        self.samples = []  # Store sampled indexes
        self.process = stochastic_process
        self._process_sampling()  # Compute entire process in advance

    def get_process(self):
        return self.process

    def get_process_data(self):
        """Return the full time-series of sample sizes."""
        return self.process_data
        
    def _process_sampling(self):
        """Precompute sampling based on the stochastic process."""
        if len(self.available_indexes[self.label]) < max(self.process_data):
            print(len(self.available_indexes[self.label]), max(self.process_data))
            raise ValueError(f"Not enough samples available for digit {self.label}")

        self.samples = []  # Store sampled indexes over time
        current_samples = set()  # Track current samples in each step

        for sample_size in self.process_data:
            # If increasing, add new samples
            if len(current_samples) < sample_size:
                needed = sample_size - len(current_samples)
                new_samples = random.sample(list(self.available_indexes[self.label]), needed)  # Convert set to list
                current_samples.update(new_samples)
                self.available_indexes[self.label].difference_update(new_samples)  # Remove from available pool
            
            # If decreasing, return some samples
            elif len(current_samples) > sample_size:
                remove_count = len(current_samples) - sample_size
                returning_samples = random.sample(list(current_samples), remove_count)
                current_samples.difference_update(returning_samples)
                self.available_indexes[self.label].update(returning_samples)  # Return to population
            
            # Store the current sample set (as a copy)
            self.samples.append(set(current_samples))

    def next_step(self):

        current_samples = self.get_samples_at(time_step = self.process.time, return_indexes = True)
        self.process.next_step()
        self.process_data = self.process.data["value"].tolist()
        sample_size = self.process_data[-1]
        print(self.process.time, sample_size)
        
        if len(current_samples) < sample_size:
            needed = sample_size - len(current_samples)
            new_samples = random.sample(list(self.available_indexes[self.label]), needed)  # Convert set to list
            current_samples.update(new_samples)
            self.available_indexes[self.label].difference_update(new_samples)  # Remove from available pool
        
        # If decreasing, return some samples
        elif len(current_samples) > sample_size:
            remove_count = len(current_samples) - sample_size
            returning_samples = random.sample(list(current_samples), remove_count)
            current_samples.difference_update(returning_samples)
            self.available_indexes[self.label].update(returning_samples)  # Return to population
        
        # Store the current sample set (as a copy)
        self.samples.append(set(current_samples))
        
    def get_samples_at(self, time_step, return_indexes = True):
        """Retrieve the actual file paths for the sample at a specific time step."""
        if time_step < 0 or time_step >= len(self.samples):
            raise ValueError("Time step out of bounds")
        
        indexes = self.samples[time_step]
        return [self.mnist_population[self.label][i] for i in indexes] if not return_indexes else indexes


