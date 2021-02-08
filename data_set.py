import pyopencl as cl
import numpy as np

from configuration import *

class DataSet:

	num_batches = 0
	completed_perms = 0
	total_perms = 0

	def __init__(self, idx, num_perms, expressions, ctx):
		print(f"Created dataset with {len(expressions):8d} expressions and "
			f"{num_perms:12d} permutations ({len(expressions)*num_perms:.3e})")
		self.idx = idx
		self.ctx = ctx
		self.num_perms = num_perms
		self.expressions_np = np.array(expressions, dtype=np.int32)
		self.n = len(expressions)
		self.total_dataset_perms = self.num_perms * self.n

		DataSet.total_perms += self.total_dataset_perms
		DataSet.num_batches += 1

	def setup_buffers(self, ctx):
		mf = cl.mem_flags

		self.expression_dtype = np.dtype([
			("operators", np.int32, (NUM_SYMBOLS,)),
			("numbers", np.int32, (NUM_NUMBERS,))
		])
		self.result_dtype = np.dtype([
			("counts", np.int32, MAX_TARGET),
			("extra_stats", np.int32, NUM_EXTRA_VALUES)
		])

		self.dims_np = np.array(
			[self.n, NUM_TOKENS, self.num_perms], dtype=np.int32)
		self.result_np = np.empty(
			(self.n,), dtype=self.result_dtype)

		self.expressions_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.expressions_np)
		self.result_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.result_np)
		self.dims_g = cl.Buffer(ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dims_np)


	def start_kernel(self, prg, queue):
		self.setup_buffers(self.ctx)
		current_part = 100 * self.total_dataset_perms / self.total_perms
		print(f"Running batch {self.idx+1:2d}/{DataSet.num_batches:2d}:", 
			f"({current_part:7.3f}%)", 
			f"{self.n:6d} items, {self.num_perms:8d} permutations")

		cl.enqueue_copy(queue, self.expressions_g, self.expressions_np)
		cl.enqueue_copy(queue, self.dims_g, self.dims_np)
		self.event = prg.evaluate(queue, (self.n,), None, 
			self.expressions_g, self.result_g, self.dims_g)

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
			counts = self.result_np[i]["counts"]
			output_dict[numbers] += counts
			self.update_extra_stats(i, extra_stats)
		self.result_np = None
		self.expressions_g = None
		self.result_g = None

	def update_extra_stats(self, i, extra_stats):
		keys = ("division_fails", "subtraction_fails", 
			"permutation_fails", "permutation_successes")
		indices = (DIVISION_FAIL_INDEX, SUBTRACTION_FAIL_INDEX, 
			PERMUTATION_FAIL_INDEX, PERMUTATION_SUCCESS_INDEX)

		for key, index in zip(keys, indices):
			extra_stats[key] += self.result_np[i]["extra_stats"][index]

		total_evaluations = self.result_np[i]["counts"].sum()
		extra_stats["total_evaluations"] += total_evaluations
