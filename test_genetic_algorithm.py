import math
from evolalgo import GeneticAlgorithm

def sphere(phenotype):
	X = list(phenotype.values())
	res = sum(x**2 for x in X)
	return {'sphere': res}

def schwefel(phenotype):
	X = list(phenotype.values())
	res = 418.9809 * len(X) - sum(x * math.sin(math.sqrt(abs(x))) for x in X)
	return {'schwefel': res}

ga = GeneticAlgorithm(pop_size=100, num_gen=100, num_pts=4)
for i in range(10):
	ga.add_variable(f'x{i + 1}', (-500, 500), 12)
ga.set_fitness_func(schwefel, 'schwefel')
ga.execute(verbose=1)