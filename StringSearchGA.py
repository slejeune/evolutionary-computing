from typing import Union, Tuple
import string

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    # set hyper parameters
    target_string = np.array(list("long target string"))
    K = 2
    N = 200
    p_crossover = 1
    mutation_rate = 0.1
    alphabet = np.array(list(string.ascii_lowercase) + [" "])
    n_runs = 20
    matplotlib.rcParams.update({'font.size': 20})

    # run the algorithm: exercise 4.1 - 4.3
    #exercise_4_1__4_3(K, N, alphabet, mutation_rate, n_runs, target_string)

    # run the algorithm: exercise 4.4 - 4.5
    exercise_4_4__4_5(K, N, alphabet, n_runs, target_string)


def exercise_4_4__4_5(K, N, alphabet, n_runs, target_string):
    mutation_rates = np.arange(start=0, stop=0.2, step=0.02)
    results = np.empty((mutation_rates.shape[0], n_runs))
    for i, mu in enumerate(tqdm(mutation_rates, desc="Running string search GA")):
        for j in range(n_runs):
            results[i, j] = string_search_GA(
                target_string=target_string, K=K, N=N, mutation_rate=mu, alphabet=alphabet, verbose=False
            )
    mean = np.mean(results, axis=1)
    std = np.std(results, axis=1)
    plt.plot(mutation_rates, mean, label="mean")
    plt.fill_between(
        x=mutation_rates,
        y1=mean - std,
        y2=mean + std,
        alpha=0.2,
        label="standard deviation (1σ)",
    )
    plt.legend()
    plt.xlabel("Mutation rate")
    plt.ylabel("Finish time (number of generations)")
    plt.title(f"GA performance using different values of mu\n"
              f"with N={N}, K={K}, and n_runs={n_runs}\n"
              f"length of target string = {target_string.shape[0]} characters")
    plt.show()


def exercise_4_1__4_3(K, N, alphabet, mutation_rate, n_runs, target_string):
    runs = [
        string_search_GA(
            target_string=target_string, K=K, N=N, mutation_rate=mutation_rate, alphabet=alphabet, verbose=False
        )
        for i in tqdm(range(n_runs), desc="Running string search GA")
    ]
    # plot the resulting distribution
    plt.boxplot(runs)
    plt.title(f"Boxplot of the distribution of the finish times over {n_runs} runs\n"
              f"with N={N}, K={K}, and mu={mutation_rate}\n"
              f"length of target string = {target_string.shape[0]} characters")
    plt.ylabel("Finish time (number of generations)")
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.show()


def string_search_GA(
    target_string: np.ndarray, K: int, N: int, mutation_rate: float, alphabet: np.ndarray, verbose: bool = True
) -> int:
    """
    The string search GA as described in the fourth Natural Computing lecture.

    Args:
        target_string (np.ndarray): The target string represented in an array.
        K (int): The number of individuals to select for the parental tournament selection.
        N (int): The population size.
        mutation_rate (float): The pointwise mutation rate of the individuals.
        alphabet (np.ndarray): All possible characters that can occur in the target string.
        verbose (bool): If True, prints the best fittest individual per generation, default = False.

    Returns:
        int: The number of generations the GA needed to find the target string.
    """
    # initialize population
    idxs = np.random.randint(low=0, high=alphabet.shape[0], size=(N, target_string.shape[0]))
    population = alphabet[idxs]

    # create hacky stopping condition
    target_found = False
    n_generations = 0

    # create a list for fast indexing
    idx_grid = np.linspace(start=0, stop=N, num=N, endpoint=False)

    if verbose:
        print("Fittest individual within the population:")
    while not target_found and n_generations < 100:
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
            children[i + 1, :] = mutate(c_two, rate=mutation_rate, alphabet=alphabet)

        # update the generation number
        n_generations += 1

        # check if the target is found
        target_found = 1 in fitness

        # paste fittest target on stdout
        if verbose:
            print(f"\t{population[np.argmax(fitness)]}, fitness: {np.max(fitness):.2f}")

        # update the population
        population = children

    if verbose:
        print(f"Target found in {n_generations} generations.")
    return n_generations


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
