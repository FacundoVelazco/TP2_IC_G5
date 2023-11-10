import pygad
import numpy as np
import utils



genes = 24

initial_pop = [[1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0,1],
               [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,0]]


capas_ocultas = 3
def fitness_func(pygadClass, solution, solution_idx):
    # if np.all(solution > 0):
        numerosEnCapas, funcionesEnCapas = utils.desnormalizar(solution)
        print(numerosEnCapas,funcionesEnCapas)
        output = utils.evaluateANN(capas_ocultas, numerosEnCapas, funcionesEnCapas)
        fitness = (1 / output) * 10000
       
        print('fitness: ' + str(fitness))
        return fitness
    # else:
    #     return 0


ga_instance = pygad.GA(num_generations=5,
                       sol_per_pop=2,
                       initial_population=initial_pop,
                       num_genes=genes,
                       num_parents_mating=2,
                       gene_type=int,
                       fitness_func=fitness_func,
                       mutation_type="random",
                       mutation_probability=0.7
                       )

ga_instance.run()

print(utils.desnormalizar(ga_instance.initial_population[0]) , utils.desnormalizar(ga_instance.initial_population[1]))
print("-----------------------------")
print(utils.desnormalizar(ga_instance.population[0]),utils.desnormalizar(ga_instance.population[1]))
ga_instance.plot_fitness()
