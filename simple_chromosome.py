import pygad
import numpy as np
import utils

hidden_layers = 5
activation_functions = ['sigmoid', 'sigmoid', 'sigmoid','sigmoid','sigmoid']

initial_pop = [[50, 50, 50, 30, 30],
               [200, 100, 200, 40, 40]]


def fitness_func(pygadClass, solution, solution_idx):
    if np.all(solution > 0):
        output = utils.evaluateANN(hidden_layers, solution, activation_functions)
        print(output)
        fitness = 1 / output
        return fitness
    else:
        return 0


ga_instance = pygad.GA(num_generations=3,
                       sol_per_pop=2,
                       initial_population=initial_pop,
                       num_genes=hidden_layers,
                       num_parents_mating=2,
                       gene_type=int,
                       fitness_func=fitness_func,
                       mutation_type="random",
                       mutation_probability=0.7)

ga_instance.run()

print(ga_instance.initial_population)
print("-----------------------------")
print(ga_instance.population)
ga_instance.plot_fitness()
