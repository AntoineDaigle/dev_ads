import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

colors = ['#005C69', '#023618', '#E9B44C', '#B0413E', '#83C5BE']

data_density_01 = []
data_density_1 = []
data_density_10 = []

with open("data/question_1/data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()


for line in lines:

    if line.startswith("density_0.1_"):
        split_line = line.split("\t")
        data_density_01.append(float(split_line[1]))

    if line.startswith("density_1_"):
        split_line = line.split("\t")
        data_density_1.append(float(split_line[1]))

    if line.startswith("density_10_"):
        split_line = line.split("\t")
        data_density_10.append(float(split_line[1]))

all_data = [data_density_01, data_density_1, data_density_10]

# sns.set_context("poster")
sns.boxplot(data=all_data, palette=colors)
plt.xticks([0, 1, 2], ["0.1", "1", "10"])
plt.xlabel("Densité [éms/faisceau laser]")
plt.ylabel(r"Coefficient de diffusion [$\mu$m²/s]")
plt.axhline(1e-12, c=colors[3], label="Simulated diff. coef.")
plt.legend()
# plt.semilogy()
plt.tight_layout()
plt.savefig("RICS_density.pdf")
plt.show()