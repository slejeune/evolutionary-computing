from typing import Union, Tuple
import string

import numpy as np


def main():
    # set hyper parameters
    target_string = np.array(list("target string"))
    K = 2
    N = 1000
    p_crossover = 1
    mutation_rate = 0.01
    alphabet = np.array(list(string.ascii_lowercase) + [" "])

    # initialize population
    idxs = np.random.randint(low=0, high=alphabet.shape[0], size=(N, target_string.shape[0]))
    population = alphabet[idxs]

    # create hacky stopping condition
    target_found = False
    n_generations = 0

    # create a list for fast indexing
    idx_grid = np.linspace(start=0, stop=N, num=N, endpoint=False)

    print("Fittest individual within the population:")
    while not target_found:
        # determine the fitness of every solution in the population
        fitness = calculate_fitness(population, target_string)

        children = np.empty((N, target_string.shape[0]), dtype=str)
        for i in range(0, children.shape[0], 2):
            # select the parents using tournament selection
            p_one = select_parent(indices=idx_grid, population=population, fitness=fitness, size=K)
            p_two = select_parent(indices=idx_grid, population=population, fitness=fitness, size=K)

            # generate the children
            c_one, c_two = crossover(one=p_one, two=p_two)
            children[i, :] = mutate(c_one, rate=mutation_rate, alphabet=alphabet)
            children[i+1, :] = mutate(c_two, rate=mutation_rate, alphabet=alphabet)

        # update the generation number
        n_generations += 1

        # check if the target is found
        target_found = 1 in fitness

        # paste fittest target on stdout
        print(f"\t{population[np.argmax(fitness)]}, fitness: {np.max(fitness):.2f}")

        # update the population
        population = children

    print(f"Target found in {n_generations} generations.")


def mutate(one: np.ndarray, rate: float, alphabet: np.ndarray) -> np.ndarray:
    """
    Make point mutations in ``one`` with a probability ``rate``.

    Args:
        one (np.ndarray): The individual to mutate.
        rate (float): The mutation rate.
        alphabet (np.ndarray): The possible characters to mutate with.

    Returns:
        np.ndarray: The mutated individual.
    """
    probs = np.random.uniform(low=0, high=1, size=one.shape[0])
    for i in range(one.shape[0]):
        if probs[i] <= rate:
            one[i] = np.random.choice(alphabet)
    return one


def crossover(one: np.ndarray, two: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crossover two arrays at a random location.

    Args:
        one (np.ndarray): the first array.
        two (np.ndarray): the second array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The crossover results from the two arrays.
    """
    crossover_point = np.random.randint(low=0, high=one.shape[0])
    return (
        np.concatenate((one[:crossover_point], two[crossover_point:])),
        np.concatenate((two[:crossover_point], one[crossover_point:])),
    )


def select_parent(indices: np.ndarray, population: np.ndarray, fitness: np.ndarray, size: int) -> np.ndarray:
    """
    Select a parent using tournament selection.

    Args:
        indices (np.ndarray): A convenience list to sample indices from.
        population (np.ndarray): The population.
        fitness (np.ndarray): The fitness that corresponds to the individuals from the population.
        size (int): The tournament selection parameter.

    Returns:
        np.ndarray: The selected parent.
    """
    potential_parent_idxs = np.random.choice(indices, size=size, replace=False).astype(int)
    return population[potential_parent_idxs[np.argmax(fitness[potential_parent_idxs])], :]


def calculate_fitness(this: np.ndarray, target: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate the fitness of a string given the target string.

    Args:
        this   (np.ndarray): The string to calculate the fitness from. This can either be a single string (an array of
                             shape: len(string) x 1), or the entire population of size N (an array of shape:
                             N x len(string)).
        target (np.ndarray): The target string.

    Returns:
        Union[float, np.ndarray]: The fitness.
    """
    if this.shape == target.shape:
        return np.count_nonzero(this == target) / target.shape[0]

    fitness = np.zeros(this.shape[0])
    for i in range(this.shape[0]):
        fitness[i] = np.count_nonzero(this[i, :] == target)
    return fitness / target.shape[0]


if __name__ == "__main__":
    main()
