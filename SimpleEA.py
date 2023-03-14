# 4.3 Travelling sales person
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm 

f = open('u-tsp.txt', 'r')
tsp = f.read()

# Number of cities
n = 22

generations = 15

mutation_rate = 0.05
crossover_rate = 0.8

candidate_n = 500

opt_search = True # ~30*time
multi_pass = False # ~5*time


# Fitness = 1/(total distance travelled)


def calculate_fitness(cities, tour):

    total_distance = 0
    sorted_c = [cities[i] for i in tour]

    for i in range(len(sorted_c)-1):
        cord = cities[tour[i]]
        next_cord = cities[tour[i+1]]
        total_distance += ((cord[0] - next_cord[0]) **
                           2 + (cord[1] - next_cord[1])**2)**0.5

    return 1/total_distance

def crossover(first_p, second_p):
    # 1. Choose two cut points.
    cutoff = np.random.randint(n/2)

    # 2. Keep middle piece.
    first_p_cut = first_p[cutoff:-cutoff]
    second_p_cut = second_p[cutoff:-cutoff]

    # 3. Take complement in other parent.
    first_comp = [x for x in first_p if x not in second_p_cut]
    second_comp = [x for x in second_p if x not in first_p_cut]

    # 4. Fill gaps in order.
    first_child = second_comp[cutoff:] + first_p_cut + second_comp[:cutoff]
    second_child = first_comp[cutoff:] + second_p_cut + first_comp[:cutoff]
    return first_child, second_child


def mutate(tour):
    i, j = np.random.choice(len(tour), size=2)
    copy_child = tour[i]
    tour[i] = tour[j]
    tour[j] = copy_child

def dis(c1,c2,cities):
    return np.linalg.norm(cities[c1]-cities[c2])

def local_search(tour,cities):
    changed=True
    while changed:
        changed = False
        for i in range(len(tour)):
            for j in range(i+2,len(tour)):    
                cur=dis(tour[i],tour[i+1],cities)+dis(tour[j],tour[j-1],cities)
                new=dis(tour[i],tour[j-1],cities)+dis(tour[j],tour[i+1],cities)
                if new < cur:
                    tour[i+1:j]=tour[j-1:i:-1]
                    changed=multi_pass 




# Coordinates parsed from the txt file
c = np.array([[float(y) for y in x.split(' ') if y] for x in tsp.split('\n')])

for ite in range(10):

    candidates = []
    fitness = []
    best_gen_fitness = []

    # Tour ordering (random initialization)
    t = [x for x in range(n)]

    for i in range(candidate_n):
        candidates.append(t.copy())
        random.shuffle(candidates[i])
        fitness.append(calculate_fitness(c, candidates[i]))

    # Repeat until stop condition satisfied:
    for _ in tqdm(range(generations)):
        fitness = np.array(fitness)
        new_candidates = []

        for _ in range(candidate_n//2):

            # 1. Select parents for reproduction
            first_p = candidates[np.random.choice(
                len(fitness), p=fitness/sum(fitness))]
            second_p = candidates[np.random.choice(
                len(fitness), p=fitness/sum(fitness))]

            # 2. Recombine and mutate (crossover)
            if random.random() < crossover_rate:
                first_child, second_child = crossover(first_p, second_p)
            else:
                first_child, second_child = first_p,second_p

            # Random mutation
            if random.random() < mutation_rate:
                mutate(first_child)

            if random.random() < mutation_rate:
                mutate(second_child)

            # (3. Apply local search to each individual)
            if opt_search:
                local_search(first_child,c)
                local_search(second_child,c)

            new_candidates.append(first_child)
            new_candidates.append(second_child)
            

        # 4. Evaluate fitness of each candidate
        new_fitness = [calculate_fitness(c, candidate) for candidate in new_candidates]

        # 5. Select next generation
        # candidates = new_candidates # Kill parents
        # fitness = new_fitness

        # Elitism
        total_c = candidates + new_candidates
        total_f = np.concatenate((fitness, np.array(new_fitness)))
        candidates = [x for _, x in sorted(zip(total_f, total_c))]
        candidates = candidates[-candidate_n:]
        fitness = sorted(total_f)[-candidate_n:]
        best_gen_fitness.append(max(fitness))

    # Select Best Child (Chiara)
    best_child = candidates[np.argmax(fitness)]

    # plt.plot([x[0] for x in [c[i] for i in best_child]],
    #         [x[1] for x in [c[i] for i in best_child]],
    #         color='black',
    #         linewidth=1,
    #         alpha=0.75)
    # plt.scatter([x[0] for x in c], [x[1] for x in c], color='firebrick', s=100)
    # plt.show()

    plt.plot(range(generations), best_gen_fitness)
plt.xlabel("generation")
plt.ylabel("fitness")    
plt.title(f"Fitness over {n} generations")
plt.show()
