import pyopencl as cl
import numpy as np
from itertools import combinations, combinations_with_replacement
from time import perf_counter as clock
from collections import defaultdict
from math import factorial
import sys
import matplotlib.pyplot as plt

NUM_NUMBERS = 6
NUM_SYMBOLS = NUM_NUMBERS - 1
NUM_TOKENS = NUM_NUMBERS + NUM_SYMBOLS
MAX_TARGET = 1000

NUM_EXTRA_VALUES = 4
SUBTRACTION_FAIL_INDEX = 1000
DIVISION_FAIL_INDEX = 1001
PERMUTATION_FAIL_INDEX = 1002
PERMUTATION_SUCCESS_INDEX = 1003

KERNEL_NAME = "countdown_kernel.cl"

def time_function(f):
	def wrapped(*args, **kw):
		print(f"{f.__name__:20s}", end="\t", flush=True)
		t0 = clock()
		res = f(*args, **kw)
		t1 = clock()
		print(f"DONE ({t1-t0:.2f}s)")
		return res
	return wrapped

def round_to_warp_size(n):
	return n if n % 32 == 0 else n + 32 - (n % 32)

class CombinedDataSet:

	def __init__(self, data_sets, ctx):
		self.data_sets = data_sets
		self.num_perms = [d.num_perms for d in data_sets]
		self.n = sum([d.n for d in data_sets])
		self.rounded_n = sum([d.rounded_n for d in data_sets])
		self.total_dataset_perms = sum([d.total_dataset_perms for d in data_sets])
		self.total_perms = self.total_dataset_perms
		self.idx = 0

		self.expressions_np = np.zeros((self.rounded_n, NUM_TOKENS), dtype=np.int32)
		offset = 0
		for d in data_sets:
			self.expressions_np[offset:offset+d.n,:] = d.expressions_np
			offset += d.rounded_n

		self.num_data_sets = np.int32(len(data_sets))
		print("generated combined data set")
		self.setup_buffers(ctx)

	def setup_buffers(self, ctx):
		mf = cl.mem_flags

		self.data_set_sizes_np = np.array([d.n for d in self.data_sets], dtype=np.int32)
		self.data_set_start_idxs_np = np.cumsum(
			[0] + list(self.data_set_sizes_np), dtype=np.int32)
		
		self.data_set_num_perms_np = np.array(
			[d.num_perms for d in self.data_sets], dtype=np.int32)

		self.result_np = np.empty(
			(self.rounded_n, MAX_TARGET + NUM_EXTRA_VALUES), dtype=np.int32)

		self.expressions_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.expressions_np)
		self.result_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.result_np)
		self.data_set_start_idxs_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.data_set_start_idxs_np)
		self.data_set_sizes_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.data_set_sizes_np)
		self.data_set_num_perms_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.data_set_num_perms_np)

	def start_kernel(self, prg, queue):
		current_part = 100 * self.total_dataset_perms / self.total_perms
		print(f"Running batch {self.idx+1:2d}/{DataSet.num_batches:2d}:", 
			f"({current_part:7.3f}%)", 
			f"{self.n:6d} items, {sum(self.num_perms):8d} permutations")

		cl.enqueue_copy(queue, self.expressions_g, self.expressions_np)
		cl.enqueue_copy(queue, self.data_set_start_idxs_g, self.data_set_start_idxs_np)
		cl.enqueue_copy(queue, self.data_set_sizes_g, self.data_set_sizes_np)
		cl.enqueue_copy(queue, self.data_set_num_perms_g, self.data_set_num_perms_np)

		self.event = prg.evaluate(queue, (self.rounded_n,), None, 
			self.expressions_g, self.result_g, self.data_set_start_idxs_g, 
			self.data_set_sizes_g, self.data_set_num_perms_g, self.num_data_sets)

	def await_kernel(self, queue):
		self.event.wait()
		t1_ns = self.event.profile.end
		t0_ns = self.event.profile.start
		self.kernel_time = 1e-9*(t1_ns - t0_ns)
		self.copy_event = cl.enqueue_copy(queue, self.result_np, self.result_g)
		DataSet.completed_perms += self.total_dataset_perms
		progress = DataSet.completed_perms / DataSet.total_perms

		print(f"Kernel time: {self.kernel_time:7.3f}s,", end="\t")
		print(f"Done with {100 * progress:6.2f}%")

	# expecting 24738480 * 2 = 49476960
	# got 22988476 (24738480 - 22988476 = 1750004)
	# 1750004 + 22988476 = 24738480
	def collect_data(self, output_dict, extra_stats):
		self.copy_event.wait()
		"""
		print(self.data_set_num_perms_np)
		print(self.data_set_sizes_np)
		print(self.data_set_start_idxs_np)
		print(self.rounded_n)
		print(self.result_np.shape)
		print(self.result_np)
		print(self.result_np[0:480,:MAX_TARGET].sum())
		print(self.result_np[480:,:MAX_TARGET].sum())
		plt.imshow(self.result_np**.01)
		plt.show()
		"""
		for i in range(self.rounded_n):
			numbers = tuple(self.expressions_np[i,-NUM_NUMBERS:])
			counts = self.result_np[i,:MAX_TARGET]
			output_dict[numbers] += counts
			self.update_extra_stats(i, extra_stats)

	def update_extra_stats(self, i, extra_stats):
		keys = ("division_fails", "subtraction_fails", 
			"permutation_fails", "permutation_successes")
		indices = (DIVISION_FAIL_INDEX, SUBTRACTION_FAIL_INDEX, 
			PERMUTATION_FAIL_INDEX, PERMUTATION_SUCCESS_INDEX)

		for key, index in zip(keys, indices):
			extra_stats[key] += self.result_np[i,index]

		total_evaluations = self.result_np[i,:MAX_TARGET].sum()
		extra_stats["total_evaluations"] += total_evaluations


class DataSet:

	num_batches = 0
	completed_perms = 0
	total_perms = 0

	def __init__(self, idx, num_perms, expressions, ctx):
		self.idx = idx
		self.num_perms = num_perms
		self.expressions_np = np.array(expressions, dtype=np.int32)
		self.n = len(expressions)
		self.rounded_n = round_to_warp_size(self.n)
		self.total_dataset_perms = self.num_perms * self.n

		DataSet.total_perms += self.total_dataset_perms
		DataSet.num_batches += 1
		self.setup_buffers(ctx)

	def setup_buffers(self, ctx):
		mf = cl.mem_flags

		self.data_set_start_idxs_np = np.array([0, self.rounded_n], dtype=np.int32)
		self.data_set_sizes_np = np.array([self.n], dtype=np.int32)
		self.data_set_num_perms_np = np.array([self.num_perms], dtype=np.int32)

		self.result_np = np.empty(
			(self.rounded_n, MAX_TARGET + NUM_EXTRA_VALUES), dtype=np.int32)

		self.expressions_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.expressions_np)
		self.result_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.result_np)
		self.data_set_start_idxs_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.data_set_start_idxs_np)
		self.data_set_sizes_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.data_set_sizes_np)
		self.data_set_num_perms_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.data_set_num_perms_np)


	def start_kernel(self, prg, queue):
		current_part = 100 * self.total_dataset_perms / self.total_perms
		print(f"Running batch {self.idx+1:2d}/{DataSet.num_batches:2d}:", 
			f"({current_part:7.3f}%)", 
			f"{self.n:6d} items, {self.num_perms:8d} permutations")

		cl.enqueue_copy(queue, self.expressions_g, self.expressions_np)
		cl.enqueue_copy(queue, self.data_set_start_idxs_g, self.data_set_start_idxs_np)
		cl.enqueue_copy(queue, self.data_set_sizes_g, self.data_set_sizes_np)
		cl.enqueue_copy(queue, self.data_set_num_perms_g, self.data_set_num_perms_np)

		self.event = prg.evaluate(queue, (self.rounded_n,), None, 
			self.expressions_g, self.result_g, self.data_set_start_idxs_g, 
			self.data_set_sizes_g, self.data_set_num_perms_g, np.int32(1))

	def await_kernel(self, queue):
		self.event.wait()
		t1_ns = self.event.profile.end
		t0_ns = self.event.profile.start
		self.kernel_time = 1e-9*(t1_ns - t0_ns)
		self.copy_event = cl.enqueue_copy(queue, self.result_np, self.result_g)
		DataSet.completed_perms += self.total_dataset_perms
		progress = DataSet.completed_perms / DataSet.total_perms

		print(f"Kernel time: {self.kernel_time:7.3f}s,", end="\t")
		print(f"Done with {100 * progress:6.2f}%")

	def collect_data(self, output_dict, extra_stats):
		self.copy_event.wait()
		for i in range(self.n):
			numbers = tuple(self.expressions_np[i,-NUM_NUMBERS:])
			counts = self.result_np[i,:MAX_TARGET]
			output_dict[numbers] += counts
			self.update_extra_stats(i, extra_stats)

	def update_extra_stats(self, i, extra_stats):
		keys = ("division_fails", "subtraction_fails", 
			"permutation_fails", "permutation_successes")
		indices = (DIVISION_FAIL_INDEX, SUBTRACTION_FAIL_INDEX, 
			PERMUTATION_FAIL_INDEX, PERMUTATION_SUCCESS_INDEX)

		for key, index in zip(keys, indices):
			extra_stats[key] += self.result_np[i,index]

		total_evaluations = self.result_np[i,:MAX_TARGET].sum()
		extra_stats["total_evaluations"] += total_evaluations


class CountdownGame:

	def __init__(self):
		self._operators = None
		self.total_kernel_time = 0
		if len(sys.argv) == 2:
			self.output_filename = sys.argv[1]
		else:
			self.output_filename = "/tmp/output.csv"
		print(f"Output filename: {self.output_filename}")
		self.setup_opencl()
		self.make_kernel()
		self.output_dict = defaultdict(
			lambda: np.zeros((MAX_TARGET,), dtype=np.int32))
		self.extra_stats = defaultdict(int)
		
		self.generate_data_sets()
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
		self.prg = cl.Program(self.ctx, kernel).build()


	@property
	def operators(self):
		if self._operators is None:
			self._operators = list(map(
				"".join, 
				combinations_with_replacement("+-*/", NUM_SYMBOLS)
			))
		return self._operators

	def map_operators(self, operators):
		return tuple(map(
			lambda o: ("+-*/".index(o) + 1) * (-1), 
			operators
		))

	def get_numbers(self):
		big_numbers = [25, 50, 75, 100]
		small_numbers = list(range(1, NUM_TOKENS))*2
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
		for n in self.numbers:
			for o in mapped_operators:
				expression = sorted(o) + list(n)
				perms = self.calculate_perms(expression)
				data[perms].append(expression)

		self.data_sets = []
		for i, (num_perms, expressions) in enumerate(data.items()):
			self.data_sets.append(DataSet(i, num_perms, expressions, self.ctx))


	def run_all_data_sets(self):
		print()
		t0 = clock()
		d = self.data_sets[:]
		combined_set = CombinedDataSet(d, self.ctx)
		combined_set.start_kernel(self.prg, self.queue)
		combined_set.await_kernel(self.queue)
		combined_set.collect_data(self.output_dict, self.extra_stats)
		self.total_elapsed = 0
		return

		for data_set in d:
			data_set.start_kernel(self.prg, self.queue)
			# break

		for data_set in d:
			data_set.await_kernel(self.queue)
			# break

		for data_set in d:
			# data_set.collect_data(self.output_dict, self.extra_stats)
			self.total_kernel_time += data_set.kernel_time
			# break

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
		target_sum = 119547486361
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


game = CountdownGame()
game.run_all_data_sets()
game.verify_and_save()