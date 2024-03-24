import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

colors = ['#005C69', '#023618', '#E9B44C', '#B0413E', '#83C5BE']

data_density_1e13 = []
data_density_1e12 = []
data_density_1e11 = []

with open("data/question_2/data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()


for line in lines:

    if line.startswith("diff_coeff_1e-13_"):
        split_line = line.split("\t")
        data_density_1e13.append(float(split_line[1]))

    if line.startswith("diff_coeff_1e-12"):
        split_line = line.split("\t")
        data_density_1e12.append(float(split_line[1]))

    if line.startswith("diff_coeff_1e-11"):
        split_line = line.split("\t")
        data_density_1e11.append(float(split_line[1]))

all_data = [data_density_1e13, data_density_1e12, data_density_1e11]

# sns.set_context("poster")
sns.boxplot(data=all_data, palette=colors)
# plt.ylim(1e-13, 1e-10)
# plt.xticks([0, 1, 2], ["0.1", "1", "10"])
plt.xlabel("Densité [éms/faisceau laser]")
plt.ylabel(r"Coefficient de diffusion [$\mu$m²/s]")
# plt.axhline(1e-12, c=colors[3], label="Simulated diff. coef.")
# plt.legend()
# plt.semilogy()
plt.scatter(0, 1e-13, c=colors[0], s=75)
plt.scatter(1, 1e-12, c=colors[1], s=75)
plt.scatter(2, 1e-11, c=colors[2], s=75)
plt.tight_layout()
plt.savefig("data/question_2/RICS_diff_coeff.pdf")
plt.show()