import numpy as np
import altair as alt
import pandas as pd
from scipy.stats import expon
import os
import random
import torch

from collections import defaultdict
import itertools


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




class Participant:
    def __init__(self, labels, stochastic_params, mnist_population, available_indexes):
        """
        Initialize the Participant with labels and their stochastic process parameters.
        
        labels: List of labels to sample from (e.g., [0, 1, 2, 3, ...])
        stochastic_params: A list of dictionaries containing the stochastic process parameters for each label
        mnist_population: The indexed MNIST population
        available_indexes: The available indexes for sampling from the MNIST population
        """
        self.labels = labels
        self.stochastic_params = stochastic_params
        self.mnist_population = mnist_population
        self.available_indexes = available_indexes
        self.Samples = self._generate_Samples()
    
    def _generate_Samples(self):
        """
        Generate samples for each label based on their respective stochastic process parameters.
        """
        samples = {}
        for label, params in zip(self.labels, self.stochastic_params):
            try:
                # Create the stochastic process for this label
                stochastic_process = StochasticProcess(
                    initial_population=params['initial_population'],
                    steps=params['steps'],
                    process_type=params['process_type'],
                    combine_with=params['combine_with'],
                    rate=params['rate'],
                    angular_coef=params['angular_coef'],
                    period=params['period'],
                    frequency=params['frequency']
                )
                # Create a sample instance for this label
                sample = Sample(
                    label=label,
                    stochastic_process=stochastic_process,
                    mnist_population=self.mnist_population,
                    available_indexes=self.available_indexes
                )
                # Store the samples for this label
                samples[label] = sample
            except KeyError as e:
                print(f"Missing parameter {e} for label {label}")
            except Exception as e:
                print(f"An error occurred for label {label}: {e}")
        return samples
    
    def get_sample_step(self, time_step):
        """
        Retrieve a list of all sample indexes across all labels for a given time step.
        
        time_step: The time step at which to fetch the samples.
        
        Returns:
            A list containing all indexes from the sets returned by get_sample_label.
        """
        sample_list = []
        
        for label in self.labels:
            sample_set = self.get_sample_label(label, time_step)  # This returns a set
            
            if sample_set:  # Ensure it's not None or empty
                sample_list.extend(sample_set)  # Convert set to list and merge it
        
        return sample_list

        
    def get_sample_label(self, label, time_step):
        """
        Get the samples for a specific label at a given time step.
        
        label: The label for which samples are needed (e.g., 3)
        time_step: The time step at which to fetch the samples
        """
        try:
            sample = self.Samples[label]
            return sample.get_samples_at(time_step)
        except KeyError:
            print(f"Sample for label {label} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def report_data(self):
        aux_data = {}
        for label in self.labels:
            aux_data[label] = self.Samples[label].get_process_data()
        
        df = pd.DataFrame.from_dict(aux_data, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)
        df.columns = ['id'] + [f'value_{i+1}' for i in range(df.shape[1] - 1)]

        return df







def get_sorted_label_indices(label_dict):
    """
    Given a dictionary where each key is a dataset name and the value is a tensor or list of labels,
    returns a new dictionary mapping each label to its corresponding indices, sorted by label.
    Indices are stored in sets instead of lists for efficient lookups.

    Args:
        label_dict (dict): A dictionary with dataset names as keys and tensors/lists of labels as values.

    Returns:
        dict: A dictionary with dataset names as keys and sorted mappings of labels to index sets.
    """
    index_map = {}

    for dataset_name, y_values in label_dict.items():
        index_map[dataset_name] = {}  # Initialize the dataset key

        # If y_values is a tensor, convert it to a list
        if isinstance(y_values, torch.Tensor):
            y_values = y_values.tolist()

        # Loop through all labels in the list
        for index, label in enumerate(y_values):  
            if label not in index_map[dataset_name]:
                index_map[dataset_name][label] = set()  # Initialize set if not present
            
            index_map[dataset_name][label].add(index)  # Store index
        
        # Sort the dictionary keys for each dataset
        index_map[dataset_name] = dict(sorted(index_map[dataset_name].items()))
    
    return index_map


# Function to randomly select labels based on the length of stochastic_params
def select_random_labels(stochastic_params, label_range=(0, 10)):
    # Generate random labels based on the length of stochastic_params
    num_labels = len(stochastic_params)  # The number of labels to select
    random_labels = random.sample(range(label_range[0], label_range[1]), num_labels)
    
    return random_labels
    
def generate_limited_values(start, stop, step, max_values=None):
    values = list(frange(start, stop, step))
    if max_values and len(values) > max_values:
        values = random.sample(values, max_values)
    return values

def frange(start, stop, step):
    while start < stop:
        yield round(start, 10)  # Avoid floating-point precision errors
        start += step

def generate_stochastic_combinations(init_pop=200, init_steps=20, init_freq=None, init_period=None, init_rate = None, init_angular = None,  max_combinations=None):
    if init_freq is None:
        init_freq = generate_limited_values(0.1 * init_pop, 0.5 * init_pop, 0.05 * init_pop, max_combinations)
    else:
        init_freq = generate_limited_values(init_freq * init_pop, 5 * init_freq * init_pop, 0.1 * init_freq * init_pop, max_combinations) if not isinstance(init_period, list) else generate_limited_values(init_freq[0] * init_pop, init_freq[1] * init_pop, (init_freq[1] - init_freq[0]) * 0.1 * init_pop, max_combinations)

    if init_period is None:
        init_period = generate_limited_values(0, 0.5, 0.07, max_combinations)
    else:
        init_period = generate_limited_values(0, init_period, 0.07*init_period, max_combinations)

    if init_rate is None:
        init_rate = generate_limited_values(-0.1, 0.1, 0.025, max_combinations)
    else:
        init_rate = generate_limited_values(-init_rate, init_rate, 0.05*init_rate, max_combinations)

    if init_angular is None:
        init_angular = generate_limited_values(-4, 4, 0.5, max_combinations)
    else:
        init_angular = generate_limited_values(-init_angular, init_angular, 0.05*init_angular, max_combinations)


    # Define possible values for 'combine_with' and 'process_type'
    combine_with_values = ['exponential', 'sine', 'exp_decay']
    process_type_values = ['combined']

    # Generate valid combinations of 'combine_with'
    combine_with_combinations = [
        comb for r in range(1, 3)  # Generate combinations of size 1 and 2
        for comb in itertools.combinations(combine_with_values, r)
        if not ('exponential' in comb and 'exp_decay' in comb)
    ]

    # Generate all combinations
    param_combinations = list(itertools.product(init_rate, init_angular, combine_with_combinations, process_type_values))

    # If max_combinations is specified, sample from it
    if max_combinations and len(param_combinations) > max_combinations:
        param_combinations = random.sample(param_combinations, max_combinations)

    # Generate the parameter dictionaries
    parameter_dicts = []
    for rate, angular, combine, process in param_combinations:
        param_dict = {
            'initial_population': init_pop,
            'steps': init_steps,
            'process_type': process,
            'combine_with': list(combine),
            'rate': rate,
            'angular_coef': angular,
            'period': random.choice(init_period) if isinstance(init_period, list) else init_period,
            'frequency': random.choice(init_freq) if isinstance(init_freq, list) else init_freq,
        }
        parameter_dicts.append(param_dict)

    return parameter_dicts
    
def add_participant_to_federation(
    federation, available_indexes,
    init_pop, init_steps, init_freq, init_period, init_rate, init_angular, max_comb
):
    """
    Generates stochastic parameter combinations, selects random labels, and adds a new Participant 
    instance to the federation.

    Args:
        federation (list): The list where the new Participant will be appended.
        available_indexes (dict): Dictionary of available indexes per dataset.
        init_pop (list): Initial population parameters.
        init_steps (list): Initial steps parameters.
        init_freq (list): Initial frequency parameters.
        init_period (list): Initial period parameters.
        max_comb (int): Maximum number of stochastic parameter combinations to generate.

    Returns:
        None: The function modifies the `federation` list in place.
    """
    # Generate a subset of combinations
    param_combinations = generate_stochastic_combinations(
        init_pop=init_pop, init_steps=init_steps,
        init_freq=init_freq, init_period=init_period, 
        init_rate = init_rate, init_angular = init_angular,
        max_combinations=max_comb
    )

    # # Print the generated combinations
    # for idx, params in enumerate(param_combinations):
    #     print(f"Combination {idx + 1}: {params}")

    # Generate a list of random labels for the given stochastic parameters
    random_labels = select_random_labels(param_combinations)

    # Create a Participant instance
    participant = Participant(
        labels=random_labels,
        stochastic_params=param_combinations, 
        mnist_population=available_indexes, 
        available_indexes=available_indexes
    )

    # Append the new participant to the federation
    federation.append(participant)

    # print(f"Added new participant with {len(random_labels)} labels to the federation.")



def select_and_remove_samples(data_dict, sample_size, keys):
    """
    Randomly selects and removes indexes from given keys in data_dict.
    
    Parameters:
        data_dict (dict): A dictionary where keys map to sets of indexes.
        sample_size (int): Total number of indexes to sample.
        keys (list): List of keys to sample from.

    Returns:
        selected_samples (list): A list of randomly chosen indexes.
    """
    selected_samples = {}
    num_keys = len(keys)

    if num_keys == 0:
        print("No keys provided for sampling.")
        return []

    # Calculate how many samples per key
    samples_per_key = sample_size // num_keys
    remainder = sample_size % num_keys  # Handle uneven splits

    for i, key in enumerate(keys):
        if key in data_dict and data_dict[key]:  # Ensure key exists and set is not empty
            # Adjust if there's a remainder (give extra to the first few keys)
            num_samples = samples_per_key + (1 if i < remainder else 0)
            
            available_samples = list(data_dict[key])  # Convert set to list for sampling
            selected = random.sample(available_samples, min(num_samples, len(available_samples)))  # Random selection
            
            selected_samples.extend(selected)  # Add to final list
            data_dict[key].difference_update(selected)  # Remove selected samples from original set

    return selected_samples

