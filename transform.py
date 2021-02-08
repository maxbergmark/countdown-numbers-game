# just a quick script to fix the output from a faulty simulation

import numpy as np

csv_filename = "output.csv"
output_filename = "output_7_filtered.csv"
data = np.loadtxt(open(csv_filename, "rb"), 
	delimiter=",", skiprows=0, dtype=np.int32)
output = []
for row in data:
	if 11 in row or 12 in row[:7]:
		continue
	output.append(row)

output = np.array(output)
print(len(output))
np.savetxt(output_filename, output, delimiter=",")