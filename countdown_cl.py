import pyopencl as cl
import numpy as np
from itertools import combinations
from time import perf_counter as clock
from collections import defaultdict
import sys

NUM_NUMBERS = 6
NUM_SYMBOLS = NUM_NUMBERS - 1
NUM_TOKENS = NUM_NUMBERS + NUM_SYMBOLS
MAX_TARGET = 1000

NUM_EXTRA_VALUES = 4
SUBTRACTION_FAIL_INDEX = 1000
DIVISION_FAIL_INDEX = 1001
PERMUTATION_FAIL_INDEX = 1002
PERMUTATION_SUCCESS_INDEX = 1003

def fac(n):
	p = 1
	for i in range(1, n+1):
		p *= i
	return p

def time_function(f):
	def wrapped(*args, **kw):
		print(f"{f.__name__:20s}", end="\t", flush=True)
		t0 = clock()
		res = f(*args, **kw)
		t1 = clock()
		print(f"DONE ({t1-t0:.2f}s)")
		return res
	return wrapped

class DataSet:

	def __init__(self, num_perms, expressions, ctx):
		self.num_perms = num_perms
		self.expressions_np = np.array(expressions, dtype=np.int32)
		self.n = len(expressions)
		self.num_perms = self.num_perms * self.n
		self.total_perms = 0
		self.setup_buffers(ctx)

	# @time_function
	def setup_buffers(self, ctx):
		mf = cl.mem_flags
		self.dims_np = np.array(
			[self.n, NUM_TOKENS, self.num_perms], dtype=np.int32)
		self.result_np = np.zeros(
			(self.n, MAX_TARGET + NUM_EXTRA_VALUES), dtype=np.int32)
		# self.output_np = np.zeros(
			# (len(self.numbers), NUM_NUMBERS + MAX_TARGET), dtype=np.int32)

		self.expressions_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.expressions_np)
		self.result_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.result_np)
		self.dims_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dims_np)

	def run(self, prg, queue, output_dict):

		elapsed = self.start_kernel(prg, queue)

	def run_kernel(self, prg, queue, output_dict):
		t0 = clock()
		cl.enqueue_copy(queue, self.expressions_g, self.expressions_np)
		cl.enqueue_copy(queue, self.dims_g, self.dims_np)
		# queue.finish()
		self.event = prg.evaluate(queue, (self.n,), None, 
			self.expressions_g, self.result_g, self.dims_g)
		self.event.wait()
		t1 = clock()
		self.total_elapsed = t1 - t0
		elapsed = 1e-9*(self.event.profile.end - self.event.profile.start)
		cl.enqueue_copy(queue, self.result_np, self.result_g)
		queue.finish()

		# print(self.num_perms)
		print(f"Elapsed: {elapsed:7.3f}s / {self.total_elapsed:7.3f}s,", end="\t")
		print(f"Done with {100 * self.num_perms / self.total_perms:6.2f}%\n")
		for i in range(self.n):
			numbers = tuple(self.expressions_np[i,-NUM_NUMBERS:])
			counts = self.result_np[i,:MAX_TARGET]
			# print(self.expressions_np[i,:])
			# if counts.sum() > 0:
				# print(counts)
			# self.update_extra_stats(i)
			output_dict[numbers] += counts

		return elapsed


	def start_kernel(self, prg, queue):
		t0 = clock()
		cl.enqueue_copy(queue, self.expressions_g, self.expressions_np)
		cl.enqueue_copy(queue, self.dims_g, self.dims_np)
		# queue.finish()
		self.event = prg.evaluate(queue, (self.n,), None, 
			self.expressions_g, self.result_g, self.dims_g)
		t1 = clock()
		self.total_elapsed = t1 - t0

	def await_kernel(self, queue, output_dict):
		self.event.wait()
		elapsed = 1e-9*(self.event.profile.end - self.event.profile.start)
		cl.enqueue_copy(queue, self.result_np, self.result_g)
		queue.finish()

		# print(self.num_perms)
		print(f"Elapsed: {elapsed:7.3f}s / {self.total_elapsed:7.3f}s,", end="\t")
		print(f"Done with {100 * self.num_perms / self.total_perms:6.2f}%\n")
		for i in range(self.n):
			numbers = tuple(self.expressions_np[i,-NUM_NUMBERS:])
			counts = self.result_np[i,:MAX_TARGET]
			# print(self.expressions_np[i,:])
			# if counts.sum() > 0:
				# print(counts)
			# self.update_extra_stats(i)
			output_dict[numbers] += counts

		return elapsed



class CountdownGame:

	def __init__(self):
		self._operators = None
		self.total_kernel_time = 0
		self.kernel_filename = "countdown_kernel.cl"
		if len(sys.argv) == 2:
			self.output_filename = sys.argv[1]
		else:
			self.output_filename = "/tmp/output.csv"
		print(f"Output filename: {self.output_filename}")
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx, 
			properties=cl.command_queue_properties.PROFILING_ENABLE)
		self.output_dict = defaultdict(
			lambda: np.zeros((MAX_TARGET,), dtype=np.int32))
		self.extra_stats = defaultdict(int)
		
		self.generate_data_sets()
		self.setup_opencl()
		self.make_kernel()


	@time_function
	def setup_opencl(self):
		mf = cl.mem_flags
		self.dims_np = np.array(list(self.data_np.shape) + [0], dtype=np.int32)
		self.result_np = np.zeros(
			(self.dims_np[0], MAX_TARGET + NUM_EXTRA_VALUES), dtype=np.int32)
		self.output_np = np.zeros(
			(len(self.numbers), NUM_NUMBERS + MAX_TARGET), dtype=np.int32)

		self.data_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.data_np)
		self.result_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.result_np)
		self.dims_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dims_np)

	@time_function
	def make_kernel(self):
		kernel = open(self.kernel_filename, "r").read()
		self.prg = cl.Program(self.ctx, kernel).build(["-cl-fast-relaxed-math"])


	@property
	def operators(self):
		if self._operators is None:
			self._operators = [
				"+++++", "++++-", "++++*", "++++/", "+++--", "+++-*", "+++-/", 
				"+++**", "+++*/", "+++//", "++---", "++--*", "++--/", "++-**", 
				"++-*/", "++-//", "++***", "++**/", "++*//", "++///", "+----", 
				"+---*", "+---/", "+--**", "+--*/", "+--//", "+-***", "+-**/", 
				"+-*//", "+-///", "+****", "+***/", "+**//", "+*///", "+////", 
				"-----", "----*", "----/", "---**", "---*/", "---//", "--***", 
				"--**/", "--*//", "--///", "-****", "-***/", "-**//", "-*///", 
				"-////", "*****", "****/", "***//", "**///", "*////", "/////"
			]
		return self._operators

	def map_operators(self, operators):
		return tuple(map(
			lambda o: " +-*/".index(o)*-1, 
			operators
		))

	def get_numbers(self):
		big_numbers = [25, 50, 75, 100]
		small_numbers = list(range(1, NUM_TOKENS))*2
		choices = set(map(
			lambda l: tuple(sorted(l)), 
			combinations(big_numbers + small_numbers, NUM_NUMBERS)))
		return sorted(choices)

	def calculate_perms(self, data_set):
		perms = fac(len(data_set))
		token_set = set(data_set)
		for token in token_set:
			perms //= fac(data_set.count(token))
		return perms

	@time_function
	def generate_data_sets(self):
		data = defaultdict(list)
		mapped_operators = list(map(self.map_operators, self.operators))
		self.numbers = self.get_numbers()
		for n in self.numbers:
			for o in mapped_operators:
				data_set = sorted(o) + list(n)
				perms = self.calculate_perms(data_set)
				data[perms].append(data_set)

		self.np_data = defaultdict(np.ndarray)
		self.data_sets = []
		for k, v in data.items():
			self.np_data[k] = np.array(v)
			self.data_sets.append(DataSet(k, v, self.ctx))

		self.max_data_size = max(map(len, data.values()))
		self.data_np = np.zeros(
			(self.max_data_size, NUM_TOKENS), dtype = np.int32)

	def calculate_permutations(self):
		total_perms = 0
		for data_set in self.data_sets:
			total_perms += data_set.num_perms
		# for k, v in self.np_data.items():
			# total_perms += k * v.shape[0]
		return total_perms

	def run_all_data_sets_old(self):
		parsed_perms = 0
		print()
		self.total_perms = self.calculate_permutations()
		for i, (num_perms, data) in enumerate(self.np_data.items()):
			current_part = 100 * num_perms * data.shape[0] / self.total_perms
			print(f"Running batch {i+1:2d}/{len(self.np_data):2d}:", 
				f"({current_part:7.3f}%)", 
				f"{data.shape[0]:6d} items, {num_perms:8d} permutations")
			elapsed, parsed_perms = self.run_single_data_set(
				num_perms, data, parsed_perms)
			self.total_kernel_time += elapsed
			break

	def run_all_data_sets(self):
		self.total_perms = self.calculate_permutations()
		for data_set in self.data_sets:
			data_set.total_perms = self.total_perms
			# data_set.start_kernel(self.prg, self.queue)
			# data_set.await_kernel(self.queue, self.output_dict)
			data_set.run_kernel(self.prg, self.queue, self.output_dict)
			break

		# for data_set in self.data_sets:
			# data_set.await_kernel(self.queue, self.output_dict)
			# break

	def update_extra_stats(self, i):
		keys = ("division_fails", "subtraction_fails", 
			"permutation_fails", "permutation_successes")
		indices = (DIVISION_FAIL_INDEX, SUBTRACTION_FAIL_INDEX, 
			PERMUTATION_FAIL_INDEX, PERMUTATION_SUCCESS_INDEX)

		for key, index in zip(keys, indices):
			self.extra_stats[key] += self.result_np[i,index]

		total_evaluations = self.result_np[i,:MAX_TARGET].sum()
		self.extra_stats["total_evaluations"] += total_evaluations

	"""
	def run_single_data_set(self, num_perms, data, parsed_perms):
		t0 = clock()
		self.data_np[:data.shape[0],:] = data
		self.dims_np[:2] = data.shape
		self.dims_np[2] = num_perms

		elapsed = self.run_kernel()
		parsed_perms += num_perms * data.shape[0]
		t1 = clock()
		total_elapsed = t1 - t0
		print(f"Elapsed: {elapsed:7.3f}s / {total_elapsed:7.3f}s,", end="\t")
		print(f"Done with {100 * parsed_perms / self.total_perms:6.2f}%\n")
		return elapsed, parsed_perms
	def run_kernel(self):
		cl.enqueue_copy(self.queue, self.data_g, self.data_np)
		cl.enqueue_copy(self.queue, self.dims_g, self.dims_np)

		self.queue.finish()
		event = self.prg.evaluate(self.queue, (self.dims_np[0],), None, 
			self.data_g, self.result_g, self.dims_g)
		event.wait()
		elapsed = 1e-9*(event.profile.end - event.profile.start)
		cl.enqueue_copy(self.queue, self.result_np, self.result_g)
		self.queue.finish()

		for i in range(self.dims_np[0]):
			numbers = tuple(self.data_np[i,-NUM_NUMBERS:])
			counts = self.result_np[i,:MAX_TARGET]
			self.update_extra_stats(i)
			self.output_dict[numbers] += counts

		return elapsed
	"""

	def print_extra_stats(self):
		for key, value in self.extra_stats.items():
			print(f"{key + ':':24s} {value:14d} ({value:.3e})")
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