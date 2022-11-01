import json
from matplotlib import pyplot as plt

from matplotlib import rc

# there will be 3 figures
with open("output/density.json") as f:
    density = json.load(f)

def plot(x, y, yerr, color, label, linestyle="solid", fill=False):
    plt.plot(x, y, marker=" ", ls=linestyle, color=color, label=label)
    plt.errorbar(x, y, yerr=yerr, fmt="none", elinewidth=1, color=color, capsize=2)
    if fill:
        plt.fill_between(x, [a_i - b_i for a_i, b_i in zip(y, yerr)], [a_i + b_i for a_i, b_i in zip(y, yerr)], alpha=0.25, color=color)

actual_dens = {
    "cora": 10556,
    "citeseer": 9104,
    "lastfm": 55612
}

for dataset in ["cora", "citeseer", "lastfm"]:
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 14
        })
    rc("figure", figsize=(4,3))

    fig = plt.figure()

    x = ["1","2","3","4","5","6","7","8"]
    plt.plot(x, [actual_dens[dataset] for i in x], linestyle=':', marker=" ", color="green", label="$\Vert A\Vert_1$")
    plot(x, [density[dataset][i][0] for i in x], [density[dataset][i][1] for i in x], color="blue", label="$\Vert\hat A\Vert_1$", fill=False)

    plt.yscale("log")
    plt.xlabel("$\epsilon$")
    plt.savefig(f"figures/density_{dataset}.pdf", bbox_inches='tight')
    legend=plt.legend()
    ax = plt.gca()
    # there will be figures popping up
    plt.close(fig)

# then create a new image
# adjust the figure size as necessary
fig_leg = plt.figure()
ax_leg = fig_leg.add_subplot()
# add the legend from the previous axes
ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=2)
# hide the axes frame and the x/y labels
ax_leg.axis('off')
fig_leg.savefig("figures/density_legend.pdf", bbox_inches='tight')