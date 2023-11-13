import pygad
import numpy as np
import utils

genes = 24

def mutation_func(offspring, ga_instance):
    mutation_indices = np.where(np.random.rand(offspring.shape[1]) < ga_instance.mutation_probability)[0]
    offspring[:, mutation_indices] = 1 - offspring[:, mutation_indices]
    return offspring


number_bits = 6
hidden_layers = 5
activation_functions = ['sigmoid','relu','tanh','softmax']
genes = hidden_layers * number_bits + 2 * hidden_layers


def fitness_func(pygadClass, solution, solution_idx):
    neuronas, funciones = utils.desnormalizar(solution)
    print(neuronas, funciones)
    output = utils.evaluateANN(len(neuronas), neuronas, funciones)
    fitness = (1 / output) * 10000
       
    print('fitness: ' + str(fitness))
    return fitness


ga_instance = pygad.GA(num_generations=20,
                       sol_per_pop=2,
                    #    initial_population=initial_pop,
                       num_genes=genes,
                       num_parents_mating=2,
                       gene_type=int,
                       fitness_func=fitness_func,
                       mutation_type=mutation_func,
                       mutation_probability=0.6,
                        init_range_low=0,
                       init_range_high=2,
                       )

print("------------POBLACION INICIAL-----------------")
print(utils.desnormalizar(ga_instance.initial_population[0]), utils.desnormalizar(ga_instance.initial_population[1]))

ga_instance.run()
print("---------------FINAL--------------")
print(utils.desnormalizar(ga_instance.population[0]), utils.desnormalizar(ga_instance.population[1]))
ga_instance.plot_fitness()