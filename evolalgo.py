import math
import numpy as np
import operator as op
from copy import deepcopy
from datetime import datetime

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
		self.phenotype = list()
		self.fitness = None


class GeneticAlgorithm(object):
	"""docstring for GeneticAlgorithm"""
	def __init__(self, pop_size=2, num_gen=1, cxpb=0.9, cxtype='npoint', num_pts=1, mutpb=-1.0, minmax='min', seed=None):
		super(GeneticAlgorithm, self).__init__()
		self.pop_size = pop_size
		self.num_gen = num_gen
		self.cxpb = cxpb
		self.cxtype = cxtype
		self.num_pts = num_pts
		self.mutpb = mutpb
		self.minmax = minmax
		self.parents = list()
		self.population = list()
		self.variable_conf = dict()
		self.variable_name = list()
		self.variable_bits = list()
		self.num_vars = 0
		self.total_bits = 0
		self.func = None
		self.crossover_types = {'npoint': self.npoint_crossover, 'binary': self.binary_crossover}
		self.crossover_ind = self.crossover_types.get(self.cxtype, self.npoint_crossover)
		self.comp = op.lt if minmax == 'min' else op.gt
		self.reverse = False if minmax == 'min' else True
		self.seed = seed if seed is not None and seed >= 0 else np.random.randint(10000000)
		np.random.seed(self.seed)

	def add_variable(self, name, bounds=(0, 0), precision=0):
		self.variable_conf[name] = {'bounds': bounds, 'precision': precision}
		self.variable_name.append(name)
		bits = bits_4_range(bounds[0], bounds[1], precision)
		self.variable_bits.append(bits)
		self.total_bits += bits
		self.num_vars += 1

	def set_fitness_func(self, func):
		self.func = func

	def initialize_pop(self):
		self.population.clear()
		for i in range(self.pop_size):
			ind = self.initialize_ind()
			self.population.append(ind)
		# Mutation adjustment
		if self.mutpb < 0.0:
			self.mutpb = 1.0 / self.total_bits

	def initialize_ind(self):
		ind = Individual()
		for n in self.variable_bits:
			ind.genotype.append(np.random.randint(2, size=n))
		return ind

	def decode_pop(self, pop):
		for ind in pop:
			self.decode_ind(ind)

	def decode_ind(self, ind):
		ind.phenotype.clear()
		for i in range(self.num_vars):
			gray = ind.genotype[i]
			binary = gray_2_binary(gray)
			decimal = binary_2_decimal(binary)
			bits = self.variable_bits[i]
			min_val, max_val = self.variable_conf[self.variable_name[i]]['bounds']
			precision = self.variable_conf[self.variable_name[i]]['precision']
			value = fit_2_range(decimal, min_val, max_val, bits)
			value = round(value, precision) if precision > 0 else int(value)
			ind.phenotype.append(value)

	def evaluation_pop(self, pop):
		for ind in pop:
			self.evaluation_ind(ind)

	def evaluation_ind(self, ind):
		ind.fitness = self.func(ind.phenotype)

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

	def binary_crossover(self, ind1, ind2):
		new1 = deepcopy(ind1)
		new2 = deepcopy(ind2)
		for i in range(self.num_vars):
			for j in range(self.variable_bits[i]):
				swap = np.random.rand() < 0.5
				new1.genotype[i][j] = ind2.genotype[i][j] if swap else ind1.genotype[i][j]
				new2.genotype[i][j] = ind1.genotype[i][j] if swap else ind2.genotype[i][j]
		self.population.append(new1)
		self.population.append(new2)

	def npoint_crossover(self, ind1, ind2):
		points = self.select_crossover_points(self.num_pts)
		new1 = deepcopy(ind1)
		new2 = deepcopy(ind2)
		idx, swap = 0, False
		for i in range(self.num_vars):
			for j in range(self.variable_bits[i]):
				if points[idx]:
					swap = not swap
				new1.genotype[i][j] = ind2.genotype[i][j] if swap else ind1.genotype[i][j]
				new2.genotype[i][j] = ind1.genotype[i][j] if swap else ind2.genotype[i][j]
				idx += 1
		self.population.append(new1)
		self.population.append(new2)

	def select_crossover_points(self, n):
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
		for i in range(self.num_vars):
			for j in range(self.variable_bits[i]):
				if np.random.rand() < self.mutpb:
					ind.genotype[i][j] = 1 - ind.genotype[i][j]

	def sort_population(self, pop):
		pop.sort(key=lambda ind: ind.fitness, reverse=self.reverse)

	def survivor_selection(self):
		self.sort_population(self.population)
		self.population = self.population[:self.pop_size]

	def execute(self):
		duration = 0
		elapsed = datetime.now()
		self.initialize_pop()
		self.decode_pop(self.population)
		self.evaluation_pop(self.population)
		self.sort_population(self.population)
		gbest = self.population[0]
		elapsed = datetime.now() - elapsed
		duration += elapsed.total_seconds()
		# print('[{}]\tf(x) = {}\t{}\t{} sec'.format(0, gbest.fitness, gbest.phenotype, elapsed.total_seconds()))
		print(f"[{0}]\tf(x) = {gbest.fitness}\t{gbest.phenotype}\t{elapsed.total_seconds()} sec")
		for i in range(self.num_gen):
			elapsed = datetime.now()
			self.tournament_selection()
			self.crossover_pop()
			self.mutation_pop(self.population[self.pop_size:])
			self.decode_pop(self.population[self.pop_size:])
			self.evaluation_pop(self.population[self.pop_size:])
			self.survivor_selection()
			fbest = self.population[0]
			if self.comp(fbest.fitness, gbest.fitness):
				gbest = deepcopy(fbest)
			elapsed = datetime.now() - elapsed
			duration += elapsed.total_seconds()
			# print('[{}]\tf(x) = {}\t{}\t{} sec'.format(i + 1, gbest.fitness, gbest.phenotype, elapsed.total_seconds()))
			print(f"[{i + 1}]\tf(x) = {gbest.fitness}\t{gbest.phenotype}\t{elapsed.total_seconds()} sec")
		gen_avg = duration / (self.num_gen + 1)
		ind_avg = duration / (self.pop_size * (self.num_gen + 1))
		# print('Elapsed time: {} sec\nAverage per generation: {} sec\nAverage per individual: {} sec'.format(duration, gen_avg, ind_avg))
		print(f"Elapsed time: {duration} sec\nAverage per generation: {gen_avg} sec\nAverage per individual: {ind_avg} sec")
		return (gbest, self.seed)