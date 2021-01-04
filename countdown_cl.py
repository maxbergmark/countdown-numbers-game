import pyopencl as cl
import numpy as np
from itertools import combinations
from time import perf_counter as clock
from collections import defaultdict

NUM_NUMBERS = 6
NUM_SYMBOLS = NUM_NUMBERS - 1
NUM_TOKENS = NUM_NUMBERS + NUM_SYMBOLS
MAX_TARGET = 1000

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

class CountdownGame:

	def __init__(self):
		self._operators = None
		self.filename = "countdown_kernel.cl"
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx, 
			properties=cl.command_queue_properties.PROFILING_ENABLE)
		self.output_dict = defaultdict(
			lambda: np.zeros((MAX_TARGET,), dtype=np.int32))


	@time_function
	def setup_opencl(self):
		mf = cl.mem_flags
		self.dims_np = np.array(list(self.data_np.shape) + [0], dtype=np.int32)
		self.result_np = np.zeros((self.dims_np[0], MAX_TARGET), dtype=np.int32)
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
		kernel = open(self.filename, "r").read()
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
		for k, v in data.items():
			self.np_data[k] = np.array(v)

		self.max_data_size = max(map(len, data.values()))
		self.data_np = np.zeros(
			(self.max_data_size, NUM_TOKENS), dtype = np.int32)

	def calculate_permutations(self):
		total_perms = 0
		for k, v in self.np_data.items():
			total_perms += k * v.shape[0]
		return total_perms

	def run_all_data_sets(self):
		parsed_perms = 0
		self.total_perms = self.calculate_permutations()
		for i, (num_perms, data) in enumerate(self.np_data.items()):
			current_part = 100 * num_perms * data.shape[0] / self.total_perms
			print(f"Running batch {i+1:2d}/{len(self.np_data):2d}:", 
				f"({current_part:7.3f}%)", 
				f"{data.shape[0]:6d} items, {num_perms:8d} permutations")
			parsed_perms = self.run_single_data_set(num_perms, 
				data, parsed_perms)
			# if i == 3:
				# break

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
		return parsed_perms

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
			counts = self.result_np[i,:]
			self.output_dict[numbers] += counts

		return elapsed

	@time_function
	def verify_and_save(self):
		total_permutations = 0
		for v in self.output_dict.values():
			total_permutations += v.sum()

		if total_permutations != 119547486361:
			print(f"\nError exists: {total_permutations} != {119547486361}")

		# with open("/tmp/output_opencl.csv", "w") as f:
			# pass
		for i, k in enumerate(sorted(self.output_dict.keys())):
			# if i % 100 == 0:
				# print(i, k)
			self.output_np[i,:NUM_NUMBERS] = k
			self.output_np[i,NUM_NUMBERS:] = self.output_dict[k]
			# counts_str = np.char.mod('%d', self.output_dict[k])
			# line = (
				# ",".join(map(str, k)) + ","
				# ",".join(counts_str)
				# ",".join(map(str, self.output_dict[k])) + "\n"
			# )
			# f.write(line)

		np.savetxt("/tmp/output_opencl.csv", self.output_np, delimiter=",")


game = CountdownGame()
# game.calculate_permutations()
game.generate_data_sets()
game.setup_opencl()
game.make_kernel()
game.run_all_data_sets()
game.verify_and_save()