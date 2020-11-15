import math
import numpy as np
import pandas as pd
import operator as op
from copy import deepcopy
from time import perf_counter


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
		self.phenotype = dict()
		self.fitness = None


class GeneticAlgorithm(object):
	"""docstring for GeneticAlgorithm"""
	def __init__(self, pop_size=2, num_gen=0, cx_pr=0.9, cx_type='npoint', num_pts=1, mut_pr=1.0, mut_down=(None, None), minmax='min', seed=None):
		super(GeneticAlgorithm, self).__init__()
		self.pop_size = pop_size
		self.num_gen = num_gen
		self.cx_pr = cx_pr
		self.cx_type = cx_type
		self.num_pts = num_pts
		self.mut_pr = mut_pr
		self.mut_down = mut_down
		self.down_rate = 0
		self.gen_lim = 0
		self.minmax = minmax
		self.parents = list()
		self.population = list()
		self.variable_info = dict()
		self.variable_name = list()
		self.num_vars = 0
		self.total_bits = 0
		self.function = None
		self.metric_name = None
		self.crossover_type = {'npoint': self.npoint_crossover, 'binary': self.binary_crossover}
		self.crossover_ind = self.crossover_type.get(self.cx_type, self.npoint_crossover)
		self.compare = op.lt if minmax == 'min' else op.gt
		self.reverse = False if minmax == 'min' else True
		self.history = list()
		self.seed = seed if seed != None and seed >= 0 else np.random.randint(10000000)
		np.random.seed(self.seed)

	def add_variable(self, name, bounds=(0, 0), precision=0, size=1):
		self.variable_name.append(name)
		bits = bits_4_range(bounds[0], bounds[1], precision) * size
		self.variable_info[name] = {'bounds': bounds, 'precision': precision, 'bits': bits, 'size': size}
		self.total_bits += bits
		self.num_vars += 1

	def set_fitness_func(self, function, metric_name):
		self.function = function
		self.metric_name = metric_name

	def initialize_pop(self):
		self.population.clear()
		for i in range(self.pop_size):
			ind = self.initialize_ind()
			self.population.append(ind)

	def initialize_ind(self):
		ind = Individual()
		for i in range(self.num_vars):
			name = self.variable_name[i]
			bits = self.variable_info[name]['bits']
			ind.genotype.append(np.random.randint(2, size=bits))
		return ind

	def mutation_adjustment(self):
		if self.mut_pr >= 1.0:
			self.mut_pr = self.mut_pr / self.total_bits
		if None not in self.mut_down:
			fin_mut = self.mut_down[0]
			self.gen_lim = self.mut_down[1]
			fin_mut = fin_mut / self.total_bits if fin_mut >= 1.0 else fin_mut
			dist = self.mut_pr - fin_mut if fin_mut > 0.0 and fin_mut < self.mut_pr else 0
			self.down_rate = dist / self.gen_lim if self.gen_lim < self.num_gen else 0

	def decode_pop(self, pop):
		for ind in pop:
			self.decode_ind(ind)

	def decode_ind(self, ind):
		for i in range(self.num_vars):
			name = self.variable_name[i]
			info = self.variable_info[name]
			gene = ind.genotype[i]
			value = self.decode_var(gene, info)
			ind.phenotype[name] = value

	def decode_var(self, gene, info):
		size = info['size']
		bits = info['bits'] // size
		precision = info['precision']
		min_val, max_val = info['bounds']
		gene = gene.reshape((size, bits))
		mapped = list()
		for part in gene:
			binary = gray_2_binary(part)
			decimal = binary_2_decimal(binary)
			value = fit_2_range(decimal, min_val, max_val, bits)
			value = round(value, precision) if precision > 0 else int(value)
			mapped.append(value)
		return np.array(mapped) if size > 1 else mapped[0]
			
	def evaluation_pop(self, pop):
		for ind in pop:
			self.evaluation_ind(ind)

	def evaluation_ind(self, ind):
		ind.fitness = self.function(ind.phenotype)

	def tournament_selection(self):
		idx1 = idx2 = -1
		self.parents.clear()
		# Change randint to sample
		for i in range(self.pop_size):
			idx1 = np.random.randint(self.pop_size)
			idx2 = idx1
			while idx2 == idx1:
				idx2 = np.random.randint(self.pop_size)
			fitness1 = self.population[idx1].fitness[self.metric_name]
			fitness2 = self.population[idx2].fitness[self.metric_name]
			winner = idx1 if self.compare(fitness1, fitness2) else idx2
			self.parents.append(winner)

	def crossover_pop(self):
		for i in range(0, self.pop_size, 2):
			idx1 = self.parents[i]
			idx2 = self.parents[i + 1]
			if np.random.rand() < self.cx_pr:
				self.crossover_ind(self.population[idx1], self.population[idx2])
			else:
				self.population.append(deepcopy(self.population[idx1]))
				self.population.append(deepcopy(self.population[idx2]))

	def binary_crossover(self, ind1, ind2):
		new1 = deepcopy(ind1)
		new2 = deepcopy(ind2)
		for i in range(self.num_vars):
			name = self.variable_name[i]
			bits = self.variable_info[name]['bits']
			for j in range(bits):
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
			name = self.variable_name[i]
			bits = self.variable_info[name]['bits']
			for j in range(bits):
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
			name = self.variable_name[i]
			bits = self.variable_info[name]['bits']
			for j in range(bits):
				if np.random.rand() < self.mut_pr:
					ind.genotype[i][j] = 1 - ind.genotype[i][j]

	def sort_population(self, pop):
		pop.sort(key=lambda ind: ind.fitness[self.metric_name], reverse=self.reverse)

	def survivor_selection(self):
		self.sort_population(self.population)
		self.population = self.population[:self.pop_size]

	def execute(self, verbose=1, display=['fitness', 'phenotype', 'elapsed']):
		self.history.clear()
		duration = perf_counter()
		elapsed = perf_counter()
		self.initialize_pop()
		self.mutation_adjustment()
		self.decode_pop(self.population)
		self.evaluation_pop(self.population)
		self.sort_population(self.population)
		gbest = self.population[0]
		elapsed = perf_counter() - elapsed
		self.send_to_history(gbest, elapsed)
		if verbose > 1:
			self.display_progress(0, gbest, elapsed, display)
		for i in range(self.num_gen):
			elapsed = perf_counter()
			self.tournament_selection()
			self.crossover_pop()
			self.mutation_pop(self.population[self.pop_size:])
			self.decode_pop(self.population[self.pop_size:])
			self.evaluation_pop(self.population[self.pop_size:])
			self.survivor_selection()
			fbest = self.population[0]
			if self.compare(fbest.fitness[self.metric_name], gbest.fitness[self.metric_name]):
				gbest = deepcopy(fbest)
			if i < self.gen_lim:
				self.mut_pr -= self.down_rate
			elapsed = perf_counter() - elapsed
			self.send_to_history(gbest, elapsed)
			if verbose > 1:
				self.display_progress(i + 1, gbest, elapsed, display)
		duration = perf_counter() - duration
		if verbose > 0:
			self.display_summary(duration)
		return gbest

	def display_progress(self, gen, ind, elapsed, display):
		output = f"[{gen}]\t"
		display_dict = {'fitness': f"f(x) = {ind.fitness[self.metric_name]}",
						'phenotype': f"{ind.phenotype}",
						'elapsed': f"{elapsed}s"}
		for item in display:
			output += display_dict[item] + '\t'
		print(output)

	def display_summary(self, duration):
		gen_avg = duration / (self.num_gen + 1)
		ind_avg = duration / (self.pop_size * (self.num_gen + 1))
		template = "Elapsed time: {}s\nAvg. time per generation: {}s\nAvg. time per individual: {}s"
		print(template.format(duration, gen_avg, ind_avg))

	def send_to_history(self, ind, elapsed):
		entry = [deepcopy(ind), elapsed]
		self.history.append(entry)

	def get_seed(self):
		return self.seed

	def save(self, filepath, include_initial_pop=False):
		history = self.history if include_initial_pop else self.history[1:]
		columns = deepcopy(self.variable_name)
		metric_name = list(self.population[0].fitness.keys())
		metric_name[metric_name.index(self.metric_name)] = 'fitness'
		columns.extend(metric_name)
		columns.append('elapsed')
		data = list()
		for entry in history:
			ind = entry[0]
			elapsed = entry[1]
			row = list(ind.phenotype.values())
			row.extend(list(ind.fitness.values()))
			row.append(elapsed)
			data.append(row)
		D = pd.DataFrame(data=data, columns=columns)
		D.to_csv(filepath, index=False)
		
