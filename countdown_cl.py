import pyopencl as cl
import numpy as np
from itertools import combinations
from time import perf_counter as clock
from collections import defaultdict
import pickle

# This script is functional but incomplete. Do not use it unless you know
# what you're doing, and have looked at the code. 

def fac(n):
	p = 1
	for i in range(1, n+1):
		p *= i
	return p

class CountdownGame:

	def __init__(self):
		self._operators = None
		self.filename = "countdown_kernel.cl"
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx, 
			properties=cl.command_queue_properties.PROFILING_ENABLE)
		self.output_dict = defaultdict(lambda: np.zeros((1000,), dtype=np.int32))

	def setup_opencl(self):
		print("Creating buffers", end="\t", flush=True)
		mf = cl.mem_flags
		# self.generate_data_set()
		self.dims_np = np.array(list(self.data_np.shape) + [0], dtype=np.int32)
		self.result_np = np.zeros((self.dims_np[0], 1000), dtype=np.int32)
		self.output_np = np.zeros((13243, 1006), dtype=np.int32)

		self.data_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.data_np)
		self.result_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.result_np)
		self.dims_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dims_np)
		print("DONE")

	def make_kernel(self):
		print("Compiling kernel", end="\t", flush=True)
		kernel = open(self.filename, "r").read()
		t0 = clock()
		self.prg = cl.Program(self.ctx, kernel).build(["-cl-fast-relaxed-math"])
		t1 = clock()
		compilation_time = t1-t0
		print(f"DONE ({compilation_time:.2f}s)")

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
		small_numbers = list(range(1, 11))*2
		choices = set(map(
			lambda l: tuple(sorted(l)), 
			combinations(big_numbers + small_numbers, 6)))
		return sorted(choices)

	def calculate_perms(self, data_set):
		perms = fac(len(data_set))
		token_set = set(data_set)
		for token in token_set:
			perms //= fac(data_set.count(token))
		return perms

	def generate_data_set(self):
		print("Generating data set", end="\t", flush=True)
		data = defaultdict(list)
		mapped_operators = list(map(self.map_operators, self.operators))
		self.numbers = self.get_numbers()
		for n in self.numbers:
			# if n != (1, 1, 2, 2, 3, 3):
				# continue
			for o in mapped_operators:
				data_set = sorted(o) + list(n)
				perms = self.calculate_perms(data_set)
				data[perms].append(data_set)
				# print(n, o)
		self.np_data = defaultdict(np.ndarray)
		for k, v in data.items():
			self.np_data[k] = np.array(v)
		self.max_data_size = max(map(len, data.values()))

		# print(np_data.keys(), self.max_data_size)
		self.data_np = np.zeros((self.max_data_size,11), dtype = np.int32)
		print("DONE")

	def calculate_permutations(self):
		total_perms = 0
		for k, v in self.np_data.items():
			total_perms += k * v.shape[0]
		return total_perms

	def run_all_data_sets(self):
		parsed_perms = 0
		total_perms = self.calculate_permutations()
		for i, (num_perms, data) in enumerate(self.np_data.items()):
			current_part = 100 * num_perms * data.shape[0] / total_perms
			print(f"Running data set {i+1:2d}/{len(self.np_data):2d}:", 
				data.shape, num_perms, f"({current_part:7.3f}%)")

			self.data_np[:data.shape[0],:] = data
			self.dims_np[:2] = data.shape
			self.dims_np[2] = num_perms
			self.result_np[:] = 0
			self.run_kernel()
			parsed_perms += num_perms * data.shape[0]
			print(f"Done with {100 * parsed_perms / total_perms:6.2f}%\n")
			# if i == 2:
				# break
		with open('test.pickle', 'wb') as f:
			pickle.dump(dict(self.output_dict), f)

	def run_kernel(self):
		cl.enqueue_copy(self.queue, self.data_g, self.data_np)
		cl.enqueue_copy(self.queue, self.dims_g, self.dims_np)
		cl.enqueue_copy(self.queue, self.result_g, self.result_np)

		self.queue.finish()
		event = self.prg.evaluate(self.queue, (self.dims_np[0]*1+0,), None, 
			self.data_g, self.result_g, self.dims_g)
		event.wait()
		elapsed = 1e-9*(event.profile.end - event.profile.start)
		cl.enqueue_copy(self.queue, self.result_np, self.result_g)
		self.queue.finish()

		for i in range(self.dims_np[0]):
			numbers = tuple(self.data_np[i,-6:])
			counts = self.result_np[i,:]
			# if numbers == (1, 1, 2, 2, 3, 3):
				# print(counts)
			self.output_dict[numbers] += counts
			# print(numbers, counts[28])
		# self.queue.finish()
		# print(self.result_np)
		print(f"Elapsed: {elapsed:7.3f}s, Maximum value: {self.result_np.max()}")
		# np.savetxt("output_py.csv", self.result_np, delimiter=",")

	def verify_and_save(self):
		total_permutations = 0
		for v in self.output_dict.values():
			total_permutations += v.sum()

		if total_permutations == 119547486361:
			print("Data is accurate!")
		else:
			print("Error exists: {total_permutations} != {119547486361}")

		with open("output_opencl.csv", "w") as f:
			for k in sorted(self.output_dict.keys()):
				f.write(",".join(map(str, k)) + ",")
				f.write(",".join(map(str, self.output_dict[k])) + "\n")



game = CountdownGame()
# game.calculate_permutations()
game.generate_data_set()
game.setup_opencl()
game.make_kernel()
game.run_all_data_sets()
# game.run_kernel()
game.verify_and_save()