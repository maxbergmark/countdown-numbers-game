import pyopencl as cl
import numpy as np
from itertools import combinations, combinations_with_replacement
from time import perf_counter as clock
from collections import defaultdict
from math import factorial
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from data_set import (
	DataSet, CombinedDataSet, time_function, 
	print_data_set_header, print_data_set_footer)
from configuration import *

KERNEL_NAME = "countdown_kernel.cl"

class CountdownGame:

	def __init__(self):
		self._operators = None
		self.total_kernel_time = 0
		if len(sys.argv) == 2:
			self.output_filename = sys.argv[1]
		else:
			Path("./data").mkdir(parents=True, exist_ok=True)
			self.output_filename = f"./data/output_{NUM_NUMBERS}.csv"
		print(f"Output filename: {self.output_filename}")
		self.setup_opencl()
		self.make_kernel()
		self.output_dict = defaultdict(
			lambda: np.zeros((MAX_TARGET,), dtype=np.int32))
		self.extra_stats = defaultdict(int)
		
		self.generate_data_sets()
		print(f"\nNumber of batches: {DataSet.num_batches}")
		print(f"Number of data sets: {len(self.numbers)}")
		print(f"Number of expressions: {DataSet.total_expressions}")
		print(("Total number of permutations: "
			f"{DataSet.total_perms} ({DataSet.total_perms:.3e})"))
		self.output_np = np.zeros(
			(len(self.numbers), NUM_NUMBERS + MAX_TARGET), dtype=np.int32)

	@time_function
	def setup_opencl(self):
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx, 
			properties=cl.command_queue_properties.PROFILING_ENABLE)

	@time_function
	def make_kernel(self):
		kernel = open(KERNEL_NAME, "r").read()
		include_power = 1 if POWER_OPERATOR in OPERATORS else 0
		self.prg = cl.Program(self.ctx, kernel).build(
			options=[
				"-D", f"NUM_NUMBERS={NUM_NUMBERS}",
				"-D", f"NUM_SYMBOLS={NUM_SYMBOLS}",
				"-D", f"NUM_TOKENS={NUM_TOKENS}",
				"-D", f"MAX_TARGET={MAX_TARGET}",
				"-D", f"NUM_EXTRA_VALUES={NUM_EXTRA_VALUES}",
				"-D", f"SUBTRACTION_FAIL_INDEX={SUBTRACTION_FAIL_INDEX}",
				"-D", f"DIVISION_FAIL_INDEX={DIVISION_FAIL_INDEX}",
				"-D", f"POWER_FAIL_INDEX={POWER_FAIL_INDEX}",
				"-D", f"PERMUTATION_FAIL_INDEX={PERMUTATION_FAIL_INDEX}",
				"-D", f"PERMUTATION_SUCCESS_INDEX={PERMUTATION_SUCCESS_INDEX}",
				"-D", f"INCLUDE_POWER={include_power}"
			]
		)

	@property
	def operators(self):
		if self._operators is None:
			self._operators = list(map(
				"".join, 
				combinations_with_replacement(OPERATORS, NUM_SYMBOLS)
			))
		return self._operators

	def map_operators(self, operators):
		return tuple(map(
			lambda o: (OPERATORS.index(o) + 1) * (-1), 
			operators
		))

	def get_numbers(self):
		big_numbers = [25, 50, 75, 100]
		small_numbers = list(range(1, 11))*2
		choices = set(map(
			lambda l: tuple(sorted(l)), 
			combinations(big_numbers + small_numbers, NUM_NUMBERS)))
		return sorted(choices)

	def calculate_perms(self, expression):
		perms = factorial(NUM_TOKENS)
		token_set = set(expression)
		for token in token_set:
			perms //= factorial(expression.count(token))
		return perms

	@time_function
	def generate_data_sets(self):
		data = defaultdict(list)
		mapped_operators = list(map(self.map_operators, self.operators))
		self.numbers = self.get_numbers()

		for o in mapped_operators:
			for n in self.numbers:
				expression = sorted(o) + list(n)
				perms = self.calculate_perms(expression)
				data[perms].append(expression)

		self.data_sets = []
		for i, (num_perms, expressions) in enumerate(data.items()):
			self.data_sets.append(DataSet(i, num_perms, expressions, self.ctx))


	def run_all_data_sets_sequential(self):
		print()
		print_data_set_header()
		DataSet.simulation_start_time = datetime.now()
		t0 = clock()
		for data_set in self.data_sets:
			data_set.start_kernel(self.prg, self.queue)
			data_set.await_kernel(self.queue)
			data_set.collect_data(self.output_dict, self.extra_stats)
			self.total_kernel_time += data_set.kernel_time

		t1 = clock()
		print_data_set_footer()
		self.total_elapsed = t1 - t0


	def run_all_data_sets_parallel(self):
		print()
		t0 = clock()
		for data_set in d:
			data_set.start_kernel(self.prg, self.queue)

		for data_set in d:
			data_set.await_kernel(self.queue)

		for data_set in d:
			data_set.collect_data(self.output_dict, self.extra_stats)
			self.total_kernel_time += data_set.kernel_time

		t1 = clock()
		self.total_elapsed = t1 - t0

	def run_all_data_sets_combined(self):
		print()
		t0 = clock()
		d = self.data_sets[:]
		combined_set = CombinedDataSet(d, self.ctx)
		combined_set.start_kernel(self.prg, self.queue)
		combined_set.await_kernel(self.queue)
		combined_set.collect_data(self.output_dict, self.extra_stats)
		self.total_kernel_time += combined_set.kernel_time
		t1 = clock()
		self.total_elapsed = t1 - t0

	def print_extra_stats(self):
		print()
		for key, value in self.extra_stats.items():
			print(f"{key + ':':24s} {value:14d} ({value:.3e})")
		print()
		print(f"Total calculation time: {self.total_elapsed:.2f}s")
		print(f"Total kernel time: {self.total_kernel_time:.2f}s")

	def verify_and_save(self):
		total_permutations = 0
		if POWER_OPERATOR in OPERATORS:
			checksums = {5: 1726774085, 6: 276255154662}
		else:
			checksums = {2: 516, 3: 62418, 4: 7468178, 
				5: 927111333, 6: 119547486361, 7: 15889428184286}
		if NUM_NUMBERS not in checksums:
			target_sum = 0
		else:
			target_sum = checksums[NUM_NUMBERS]
	
		for v in self.output_dict.values():
			total_permutations += v.sum()

		self.print_extra_stats()

		if total_permutations != target_sum:
			print(f"\nError exists: {total_permutations} != {target_sum}")

		for i, k in enumerate(sorted(self.output_dict.keys())):
			self.output_np[i,:NUM_NUMBERS] = k
			self.output_np[i,NUM_NUMBERS:] = self.output_dict[k]

		with open(self.output_filename, "wb") as f:
			np.savetxt(f, self.output_np, fmt='%d', delimiter=",")

if __name__ == "__main__":
	game = CountdownGame()
	game.run_all_data_sets_sequential()
	# game.run_all_data_sets_parallel()
	game.verify_and_save()
