import numpy as np
import matplotlib.pyplot as plt
from sympy import isprime
from enum import Enum
import os

def pretty_print(f):
	def wrapped(*args, **kw):
		n = len(f.__name__)
		print(f"--- {f.__name__} {'-'*(35-n)}")
		res = f(*args, **kw)
		print("-"*40)
		return res
	return wrapped

class Visualizer(Enum):
	BOARDS = 0
	NUMBERS = 1

class Analyzer:

	def __init__(self, include_below_100 = False):
		self.include = include_below_100
		self.format_data()

	def load_data(self):
		dat_filename = "output.dat"
		csv_filename = "output.csv"
		if os.path.isfile(dat_filename):
			data = np.fromfile(dat_filename, dtype=np.int32)
		elif os.path.isfile(csv_filename):
			data = np.loadtxt(open(csv_filename, "rb"), 
				delimiter=",", skiprows=0, dtype=np.int32)
			data.tofile(dat_filename)
		else:
			print("File needs to be generated before running analysis")
			quit()
		return data

	def format_data(self):
		data = self.load_data()
		n = data.shape[0]
		data.shape = (n // 1006, 1006)
		self.offset = 0 if self.include else 100

		self.boards = data[:,:6]
		self.counts = data[:,6+self.offset:]
		self.numbers = np.arange(self.offset, 1000)
		self.numbers_per_board = (self.counts[:,:] > 0).sum(axis=1)
		self.boards_per_number = (self.counts[:,:] > 0).sum(axis=0)

	def plot_percentiles(self):
		x = np.linspace(0, 100, 1000)
		d = np.percentile(self.numbers_per_board, x)
		plt.plot(x, d)
		plt.xlabel("Percentile")
		plt.ylabel("Reachable target numbers")
		plt.title("Percentiles for number of reachable target numbers per board")
		plt.show()

	@pretty_print
	def easiest_numbers(self):
		mask = (self.boards_per_number == self.boards_per_number.max())
		print("Easiest target numbers:")
		print(f"    {self.numbers[mask]}")
		print("Number of boards that can form these numbers:",
			self.boards_per_number.max())
		print("Boards that can't reach these numbers:")
		for target in self.numbers[mask]:
			print("Target:", target)
			for row in self.boards[self.counts[:,target - self.offset] == 0]:
				print(f"    {row}")

	@pretty_print
	def hardest_numbers(self):
		indices = self.boards_per_number.argsort()
		ordered_numbers = self.numbers[indices]
		ordered_counts = (self.counts[:,indices] > 0).sum(axis=0)
		print("Hardest numbers to reach:")
		for number, count in zip(ordered_numbers[:10], ordered_counts[:10]):
			print(f"    {number}: {count} boards")

	@pretty_print
	def easiest_boards(self):
		mask = self.numbers_per_board == self.numbers_per_board.max()
		num_easy_boards = (mask).sum()
		easy_boards = self.boards[mask,:]
		print(("Number of boards that can reach all "
			f"{len(self.numbers)} numbers: {num_easy_boards}"))
		high_digits_hist = [[] for _ in range(5)]
		for eb in easy_boards:
			high_digits_hist[(eb > 10).sum()].append(eb)
		print("Distribution of high digits (25-100) among these boards:")
		for i, n in enumerate(high_digits_hist):
			print(f"    {i} high digits: {len(n)} boards")
			if i in (0, 3):
				for eb in high_digits_hist[i]:
					print(" "*8 + str(eb))


	@pretty_print
	def hardest_boards(self):
		indices = self.numbers_per_board.argsort()
		ordered_boards = self.boards[indices,:]
		ordered_counts = (self.counts[indices,:] > 0).sum(axis=1)
		print("Hardest boards:")
		top_n = 20
		for idx, board, count in zip(indices[:top_n], 
				ordered_boards[:top_n], ordered_counts[:top_n]):
			
			reachable = self.numbers[self.counts[idx,:] > 0]
			with np.printoptions(threshold=15):
				print(f"    {board}: {count} numbers ({reachable})")

	def format_boards_per_number(self, name, avg, std_dev, ax):
		print(("Average number of boards able to reach "
			f"{name + ' numbers:':20s}\t{avg:8.2f} ± {std_dev:8.2f}"))
		ax.set_title(f"Distribution of {name} numbers reachable by x boards")
		ax.set_xlabel("Number of boards")
		ax.set_ylabel("Frequency")
		ax.set_xlim([9000, len(self.boards)])
		ax.set_ylim([0, 250])

	def format_numbers_per_board(self, name, avg, std_dev, ax):
		print((f"Average number of "
			f"{name + ' numbers reachable per board:':40s}"
			f"\t{avg:8.2f} ± {std_dev:8.2f}"))
		ax.set_title(f"Distribution of boards reaching x {name} numbers")
		ax.set_xlabel("Possible target numbers")
		ax.set_ylabel("Frequency")
		ax.set_ylim([0, 12000])

	def get_indices(self):
		all_indices = self.numbers - self.offset
		even_indices = self.numbers[::2] - self.offset
		odd_indices = self.numbers[1::2] - self.offset
		vf = np.vectorize(isprime)
		prime_indices = self.numbers[vf(self.numbers)] - self.offset
		nonprime_indices = self.numbers[~vf(self.numbers)] - self.offset
		high_indices = self.numbers[len(self.numbers)//2:] - self.offset
		low_indices = self.numbers[:len(self.numbers)//2] - self.offset

		return (all_indices, even_indices, odd_indices, prime_indices, 
			nonprime_indices, high_indices, low_indices)

	@pretty_print
	def even_odd_prime(self, visualizer):
		if visualizer == Visualizer.BOARDS:
			formatter = self.format_boards_per_number
			axis = 0
		else:
			formatter = self.format_numbers_per_board
			axis = 1

		indices = self.get_indices()
		names = ("all", "even", "odd", "prime", "non-prime", "high", "low")

		plt.figure(figsize=(16, 18))
		for i, (idxs, name) in enumerate(zip(indices, names)):
			if name == "all":
				ax = plt.subplot2grid((5, 2), (0, 0), rowspan=2, colspan=2)
			else:
				row, col = (i-1)//2+2, (i-1)%2
				ax = plt.subplot2grid((5, 2), (row, col))
			
			c = (self.counts[:,idxs] > 0).sum(axis=axis)
			avg = c.mean()
			std_dev = c.std()

			ax.hist(c, bins=30)
			ax.plot([avg, avg], [0, 100000], 'r')
			formatter(name, avg, std_dev, ax)
		plt.subplots_adjust(hspace=0.25, top=0.97, left=0.04, right=0.98, bottom=0.04)
		plt.show()

	def run_analysis(self):
		self.plot_percentiles()
		self.easiest_numbers()
		self.hardest_numbers()
		self.easiest_boards()
		self.hardest_boards()
		self.even_odd_prime(Visualizer.BOARDS)
		self.even_odd_prime(Visualizer.NUMBERS)

a = Analyzer(include_below_100=False)
a.run_analysis()