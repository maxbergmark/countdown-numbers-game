import pyopencl as cl
import numpy as np
from itertools import combinations
from time import perf_counter as clock

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

	def setup_opencl(self):
		mf = cl.mem_flags
		self.generate_data_set()
		self.dims_np = np.array(list(self.data_np.shape) + [0], dtype = np.int32)
		self.result_np = np.zeros((self.dims_np[0], 1000), dtype = np.int32)

		self.data_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.data_np)
		self.result_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.result_np)
		self.dims_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dims_np)

	def make_kernel(self):
		kernel = open(self.filename, "r").read()
		t0 = clock()
		self.prg = cl.Program(self.ctx, kernel).build(["-cl-fast-relaxed-math"])
		t1 = clock()
		self.compilation_time = t1-t0

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

	def generate_data_set(self):
		data = []
		mapped_operators = list(map(self.map_operators, self.operators))
		self.numbers = self.get_numbers()
		for n in self.numbers:
			for o in mapped_operators:
				data.append(sorted(o) + list(n))
				# print(n, o)
		self.data_np = np.array(data, dtype = np.int32)

	def calculate_permutations(self):
		mapped_operators = list(map(self.map_operators, self.operators))
		self.numbers = self.get_numbers()
		total_perms = 0
		for n in self.numbers:
			for o in mapped_operators:
				data_set = sorted(o) + list(n)
				token_set = set(data_set)
				perms = fac(len(data_set))
				for token in token_set:
					perms //= fac(data_set.count(token))
				# print(len(data_set))
				total_perms += perms
		print(total_perms, f"{total_perms:e}")

	def run_kernel(self):
		print("running kernel")
		print(self.data_np.shape)
		print(self.result_np.shape)
		cl.enqueue_copy(self.queue, self.data_g, self.data_np)
		cl.enqueue_copy(self.queue, self.dims_g, self.dims_np)
		cl.enqueue_copy(self.queue, self.result_g, self.result_np)

		self.queue.finish()
		event = self.prg.evaluate(self.queue, (self.dims_np[0]//1000,), None, 
			self.data_g, self.result_g, self.dims_g)
		event.wait()
		elapsed = 1e-6*(event.profile.end - event.profile.start)
		cl.enqueue_copy(self.queue, self.result_np, self.result_g)
		self.queue.finish()
		print()
		# self.queue.finish()
		print(self.result_np)
		print(self.result_np.max())
		print("Elapsed (ms):", elapsed)
		# np.savetxt("output_py.csv", self.result_np, delimiter=",")

game = CountdownGame()
# game.calculate_permutations()
game.setup_opencl()
game.make_kernel()
game.run_kernel()