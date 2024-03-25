import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

colors = ['#005C69', '#023618', '#E9B44C', '#B0413E', '#83C5BE']

data_density_1e13 = []
data_density_1e12 = []
data_density_1e11 = []

with open("data/question_3/data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()


for line in lines:

    if line.startswith("dwelltime_5e-08"):
        split_line = line.split("\t")
        data_density_1e13.append(float(split_line[1]))

    if line.startswith("dwelltime_5e-05"):
        split_line = line.split("\t")
        data_density_1e12.append(float(split_line[1]))

    if line.startswith("dwelltime_0.05"):
        split_line = line.split("\t")
        data_density_1e11.append(float(split_line[1]))

all_data = [data_density_1e13, data_density_1e12, data_density_1e11]

# sns.set_context("poster")
sns.boxplot(data=all_data, palette=colors)
# plt.ylim(1e-13, 1e-12)
plt.xticks([0, 1, 2], ["50e-9", "50e-6", "50e-3"])
plt.xlabel("Délai par pixel [s]")
plt.ylabel(r"Coefficient de diffusion [$\mu$m²/s]")
plt.axhline(1e-12, c=colors[3], label="Simulated diff. coef.")
plt.legend()
# plt.semilogy()
plt.tight_layout()
plt.savefig("data/question_3/RICS_pixeldwelltime.pdf")
plt.show()