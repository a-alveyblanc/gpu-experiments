import numpy as np
import matplotlib.pyplot as plt

tb_sizes = []
timings = []
with open("timings.txt", "r") as f:
    while (line := f.readline()):
        str_data = line.split(",")
        tb_sizes.append(int(str_data[0]))
        timings.append(float(str_data[1]))

tb_sizes = np.array(tb_sizes)
timings = np.array(timings)

k = np.arange(len(timings))
kp1 = np.roll(k, -1)

# rough check for knees
# NOTE: this won't look correct if the system is noisy (applications using
# hardware acceleration, streaming, video going, multiple monitors, etc.)
knee_points = np.where(abs(timings[k[:-1]] - timings[kp1[:-1]]) > 1e-1)[0]

# print knees and show that all knee points are a multiple of the first knee
# point, i.e. the SM size
print(tb_sizes[knee_points])
# enable if the system isn't noisy
# for tb_size in tb_sizes[knee_points][1:]:
#     print(tb_size % tb_sizes[0])

# plot
plt.plot(tb_sizes, timings, label="timing data")
plt.plot(tb_sizes[knee_points], timings[knee_points], 'o', 
         c='r', 
         markersize=5,
         label="knee points")
plt.legend()
plt.savefig("timing.pdf")
plt.show()
