import pyopencl as cl
import numpy as np
from time import perf_counter as clock

from configuration import *

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
			[0] + [d.rounded_n for d in self.data_sets], dtype=np.int32)
		
		self.data_set_num_perms_np = np.array(
			[d.num_perms for d in self.data_sets], dtype=np.int32)

		self.result_dtype = np.dtype([
			("counts", np.int32, MAX_TARGET),
			("extra_stats", np.int32, NUM_EXTRA_VALUES)
		])

		self.result_np = np.empty(
			(self.rounded_n,), dtype=self.result_dtype)

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

	@time_function
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

	@time_function
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
	@time_function
	def collect_data(self, output_dict, extra_stats):
		self.copy_event.wait()
		# plt.imshow(self.result_np**.01)
		# plt.show()

		for i in range(self.rounded_n):
			numbers = tuple(self.expressions_np[i,-NUM_NUMBERS:])
			if sum(numbers) == 0:
				continue
			counts = self.result_np[i]["counts"]
			output_dict[numbers] += counts
			self.update_extra_stats(i, extra_stats)

	def update_extra_stats(self, i, extra_stats):
		keys = ("division_fails", "subtraction_fails", 
			"permutation_fails", "permutation_successes")
		indices = (DIVISION_FAIL_INDEX, SUBTRACTION_FAIL_INDEX, 
			PERMUTATION_FAIL_INDEX, PERMUTATION_SUCCESS_INDEX)

		for key, index in zip(keys, indices):
			extra_stats[key] += self.result_np[i]["extra_stats"][index]

		total_evaluations = self.result_np[i]["counts"].sum()
		extra_stats["total_evaluations"] += total_evaluations


class DataSet:

	num_batches = 0
	completed_perms = 0
	total_perms = 0

	def __init__(self, idx, num_perms, expressions, ctx):
		self.idx = idx
		self.ctx = ctx
		self.num_perms = num_perms
		self.expressions_np = np.array(expressions, dtype=np.int32)
		self.n = len(expressions)
		self.rounded_n = round_to_warp_size(self.n)
		self.total_dataset_perms = self.num_perms * self.n

		DataSet.total_perms += self.total_dataset_perms
		DataSet.num_batches += 1
#		self.setup_buffers(ctx)

	def setup_buffers(self, ctx):
		mf = cl.mem_flags

		self.data_set_start_idxs_np = np.array([0, self.rounded_n], dtype=np.int32)
		self.data_set_sizes_np = np.array([self.n], dtype=np.int32)
		self.data_set_num_perms_np = np.array([self.num_perms], dtype=np.int32)

		self.result_dtype = np.dtype([
			("counts", np.int32, MAX_TARGET),
			("extra_stats", np.int32, NUM_EXTRA_VALUES)
		])

		self.result_np = np.empty(
			(self.n,), dtype=self.result_dtype)

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
		self.setup_buffers(self.ctx)
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
			counts = self.result_np[i]["counts"]
			output_dict[numbers] += counts
			self.update_extra_stats(i, extra_stats)
		print("perm fail:", extra_stats["permutation_fails"])
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
