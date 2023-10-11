import numpy as np
import pygad
import pygad.utils.mutation as mutation
import numpy

inputs = [25,50,25]
desired_output = 32


def fitness_func(pygadClass, solution, solution_idx):
    output = numpy.sum(solution * inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)

    return fitness


ga_instance = pygad.GA(num_generations=10000,
                       sol_per_pop=2,
                       num_genes=len(inputs),
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       mutation_type="random",
                       mutation_probability=0.6)

ga_instance.run()

print(ga_instance.initial_population)
print("-----------------------------")
print(ga_instance.population)
ga_instance.plot_fitness()
