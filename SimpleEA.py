# 4.3 Travelling sales person
import matplotlib.pyplot as plt
import random
import numpy as np

# Fitness = 1/(total distance travelled)
def calculate_fitness(cities, tour):

    total_distance = 0
    sorted_c = [cities[i] for i in tour]

    for i in range(len(sorted_c)):
        if i == len(sorted_c)-1:
            break
        cord = cities[tour[i]]
        next_cord = cities[tour[i+1]]
        total_distance += ((cord[0] - next_cord[0])**2 + (cord[1] - next_cord[1])**2)**0.5
        
    return 1/total_distance

f = open('file-tsp.txt', 'r')
tsp = f.read()

# Number of cities
n = 50

generations = 1500

mutation_rate = 0.05
crossover_rate = 0.1

candidate_n = 500
candidates = []
fitness = []
best_gen_fitness = []

# Coordinates parsed from the txt file
c = [[float(y) for y in x.split(' ') if y] for x in tsp.split('\n')]

# Tour ordering (random initialization)
t = [x for x in range(n)]

for i in range(candidate_n):
    candidates.append(t.copy())
    random.shuffle(candidates[i])
    fitness.append(calculate_fitness(c,candidates[i]))

# Repeat until stop condition satisfied:
for r in range(generations):

    fitness = np.array(fitness)
    new_candidates = []

    for g in range(candidate_n):

        if random.random() < crossover_rate:
            # 1. Select parents for reproduction
            first_p = candidates[np.random.choice(len(fitness),p=fitness/sum(fitness))]
            second_p = candidates[np.random.choice(len(fitness),p=fitness/sum(fitness))]

            # 2. Recombine and mutate (crossover)
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

        # Random mutation
        # TODO: function
        if random.random() < mutation_rate:
            i_one = np.random.choice(len(first_child))
            i_two = np.random.choice(len(first_child))
            copy_child = first_child[i_one]
            first_child[i_one] = first_child[i_two]
            first_child[i_two] = copy_child
        
        if random.random() < mutation_rate:
            i_one = np.random.choice(len(first_child))
            i_two = np.random.choice(len(first_child))
            copy_child = second_child[i_one]
            second_child[i_one] = second_child[i_two]
            second_child[i_two] = copy_child

        new_candidates.append(first_child)
        new_candidates.append(second_child)
    best_gen_fitness.append(max(fitness))

    # (3. Apply local search to each individual)


    # 4. Evaluate fitness of each candidate
    new_fitness = []
    for i in range(candidate_n):
        new_fitness.append(calculate_fitness(c,new_candidates[i]))

    # 5. Select next generation
    # candidates = new_candidates # Kill parents

    # Elitism
    total_c = candidates + new_candidates
    total_f = np.concatenate((fitness, np.array(new_fitness)))
    candidates = [x for _, x in sorted(zip(total_f, total_c))]
    candidates = candidates[candidate_n:]
    fitness = sorted(total_f)[candidate_n:]

# Select Best Child (Chiara)
best_child = candidates[np.argmax(fitness)]

plt.plot([x[0] for x in [c[i] for i in best_child]], 
         [x[1] for x in [c[i] for i in best_child]], 
         color='black',
         linewidth=1,
         alpha=0.75)
plt.scatter([x[0] for x in c],[x[1] for x in c], color='firebrick', s=100)
plt.show()

plt.plot(range(generations), best_gen_fitness)
plt.title("Fitness over time")
plt.show()