import math
import numpy as np
import operator as op
from copy import deepcopy

def bits_4_range(lower, upper, precision):
	return int(math.ceil(math.log2((upper - lower) * math.pow(10, precision) + 1)))

def gray_2_binary(gray):
	binary = np.zeros(gray.size)
	binary[0] = bit = gray[0]
	for i in range(1, gray.size):
		if gray[i] == 1:
			bit = 1 - bit
		binary[i] = bit
	return binary

def binary_2_decimal(binary):
	i, decimal = 0, 0
	for bit in binary[::-1]:
		if bit == 1:
			decimal += math.pow(2, i)
		i += 1
	return decimal

def fit_2_range(decimal, min_val, max_val, bits):
	return min_val + decimal * (max_val - min_val) / (math.pow(2, bits) - 1)


class Individual(object):
	"""docstring for Individual"""
	def __init__(self):
		super(Individual, self).__init__()
		self.genotype = list()
		self.fenotype = None
		self.fitness = None


class GeneticAlgorithm(object):
	"""docstring for GeneticAlgorithm"""
	def __init__(self, pop_size=2, n_generations=1, cxpb=0.9, n_points=1, mutpb=0.0, minmax='min', seed=None):
		super(GeneticAlgorithm, self).__init__()
		self.pop_size = pop_size
		self.n_generations = n_generations
		self.cxpb = cxpb
		self.n_points = n_points
		self.mutpb = mutpb
		self.parents = list()
		self.population = list()
		self.variable_conf = dict()
		self.variable_names = list()
		self.n_vars = 0
		self.n_bits = list()
		self.total_bits = 0
		self.func = None
		self.comp = op.lt if minmax == 'min' else op.gt
		if seed is not None and seed >= 0:
			np.random.seed(seed)
		self.seed = np.random.get_state()[1][0]

	def set_variable_conf(self, conf):
		for var in conf:
			self.add_variable(var['name'], var['values'], var['precision'])

	def add_variable(self, name, values=(0, 0), precision=0):
		self.variable_conf[name] = {'values': values, 'precision': precision}
		self.variable_names.append(name)
		bits = bits_4_range(values[0], values[1], precision)
		self.n_bits.append(bits)
		self.total_bits += bits
		self.n_vars += 1

	def set_fitness_func(self, func):
		self.func = func

	def initialize_pop(self):
		for i in range(self.pop_size):
			ind = self.initialize_ind()
			self.population.append(ind)
		# Mutation adjustment
		if self.mutpb < 0.0:
			self.mutpb = 1.0 / self.total_bits

	def initialize_ind(self):
		ind = Individual()
		for n in self.n_bits:
			ind.genotype.append(np.random.randint(2, size=n))
		return ind

	def decode_pop(self, pop):
		for ind in pop:
			self.decode_ind(ind)

	def decode_ind(self, ind):
		ind.fenotype = np.zeros(self.n_vars)
		for i in range(self.n_vars):
			gray = ind.genotype[i]
			binary = gray_2_binary(gray)
			decimal = binary_2_decimal(binary)
			min_val, max_val = self.variable_conf[self.variable_names[i]]['values']
			bits = self.n_bits[i]
			ind.fenotype[i] = fit_2_range(decimal, min_val, max_val, bits)

	def evaluation_pop(self, pop):
		for ind in pop:
			self.evaluation_ind(ind)

	def evaluation_ind(self, ind):
		ind.fitness = self.func(ind.fenotype)

	def tournament_selection(self):
		idx1 = idx2 = -1
		self.parents.clear()
		for i in range(self.pop_size):
			idx1 = np.random.randint(self.pop_size)
			idx2 = idx1
			while idx2 == idx1:
				idx2 = np.random.randint(self.pop_size)
			fitness1 = self.population[idx1].fitness
			fitness2 = self.population[idx2].fitness
			winner = idx1 if self.comp(fitness1, fitness2) else idx2
			self.parents.append(winner)

	def crossover_pop(self):
		for i in range(0, self.pop_size, 2):
			idx1 = self.parents[i]
			idx2 = self.parents[i + 1]
			if np.random.rand() < self.cxpb:
				self.crossover_ind(self.population[idx1], self.population[idx2])
			else:
				self.population.append(deepcopy(self.population[idx1]))
				self.population.append(deepcopy(self.population[idx2]))

	def crossover_ind(self, ind1, ind2):
		points = self.select_cx_points(self.n_points)
		new1 = deepcopy(ind1)
		new2 = deepcopy(ind2)
		idx, swap = 0, False
		for i in range(self.n_vars):
			for j in range(self.n_bits[i]):
				if points[idx]:
					swap = not swap
				new1.genotype[i][j] = ind2.genotype[i][j] if swap else ind1.genotype[i][j]
				new2.genotype[i][j] = ind1.genotype[i][j] if swap else ind2.genotype[i][j]
				idx += 1
		self.population.append(new1)
		self.population.append(new2)

	def select_cx_points(self, n):
		count = 0
		points = [False] * self.total_bits
		while count < n:
			idx = np.random.randint(1, self.total_bits)
			if not points[idx]:
				points[idx] = True
				count += 1
		return points

	def mutation_pop(self, pop):
		for ind in pop:
			self.mutation_ind(ind)

	def mutation_ind(self, ind):
		for i in range(self.n_vars):
			for j in range(self.n_bits[i]):
				if np.random.rand() < self.mutpb:
					ind.genotype[i][j] = 1 - ind.genotype[i][j]

	def survivor_selection(self, reverse=False):
		self.population.sort(key=lambda ind: ind.fitness, reverse=reverse)
		self.population = self.population[:self.pop_size]

	def execute(self):
		self.initialize_pop()
		self.decode_pop(self.population)
		self.evaluation_pop(self.population)
		self.survivor_selection()
		gbest = self.population[0].fitness
		# self.print_ind(self.population[0])
		for i in range(self.n_generations):
			self.tournament_selection()
			self.crossover_pop()
			self.mutation_pop(self.population[self.pop_size:])
			self.decode_pop(self.population[self.pop_size:])
			self.evaluation_pop(self.population[self.pop_size:])
			self.survivor_selection()
			fbest = self.population[0].fitness
			# self.print_ind(self.population[0])
			if self.comp(fbest, gbest):
				gbest = fbest
		return (gbest, self.seed)

	def print_ind(self, ind):
		print(np.array(ind.genotype).ravel(), sep='')
		for i in range(self.n_vars):
			print('\tx{}: {}'.format(i + 1, ind.fenotype[i]), sep='')
		print('\tf(x): {}'.format(ind.fitness))


def sphere(xval):
	return sum([x**2 for x in xval])

# alg = GeneticAlgorithm(100, 100, mutpb=-1)
# alg.add_variable('x1', values=(-5.12, 5.12), precision=2)
# alg.add_variable('x2', values=(-5.12, 5.12), precision=2)
# alg.add_variable('x3', values=(-5.12, 5.12), precision=2)
# alg.add_variable('x4', values=(-5.12, 5.12), precision=2)
# alg.add_variable('x5', values=(-5.12, 5.12), precision=2)
# alg.set_fitness_func(sphere)
# print(alg.execute())