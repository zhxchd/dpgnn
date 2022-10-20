import json
from matplotlib import pyplot as plt

from matplotlib import rc

# there will be 9 figures
with open("output/results.json") as f:
    blink_res = json.load(f)
with open("output/bl_results.json") as f:
    bl_res = json.load(f)

def plot(x, y, yerr, color, label, fill=False):
    plt.plot(x, y, marker=" ", color=color, label=label)
    plt.errorbar(x, y, yerr=yerr, elinewidth=1, color=color, capsize=2)
    if fill:
        plt.fill_between(x, [a_i - b_i for a_i, b_i in zip(y, yerr)], [a_i + b_i for a_i, b_i in zip(y, yerr)], alpha=0.25, color=color)

for dataset in ["cora", "citeseer", "lastfm"]:
    for model in ["gcn", "graphsage", "gat"]:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 14
        })
        rc("figure", figsize=(4,3))

        x = ["1","2","3","4","5","6","7","8"]

        # draw non private accuracy (upper bound)
        plt.plot(x, [blink_res[dataset][model]["None"][0] for i in x], linestyle=':', marker=" ", color="green", label="$\epsilon=\infty$ (non-private)")

        # draw pure privacy MLP (lower bound)
        plt.plot(x, [blink_res[dataset]["mlp"]["None"][0] for i in x], linestyle=':', marker=" ", color="red", label="$\epsilon=0$ (MLP)")

        plot(x, [blink_res[dataset][model][i][0] for i in x], [blink_res[dataset][model][i][1] for i in x], color="blue", label="Blink (ours)", fill=False)
        plot(x, [bl_res[dataset][model]["ldpgcn"][i][0] for i in x], [bl_res[dataset][model]["ldpgcn"][i][1] for i in x], color="orange", label="L-DPGCN", fill=False)
        plot(x, [bl_res[dataset][model]["rr"][i][0] for i in x], [bl_res[dataset][model]["rr"][i][1] for i in x], color="c", label="RR", fill=False)

        # plt.ylim(ymin=0.2, ymax=0.9)
        if plt.gca().get_ylim()[0] > 0.5:
            plt.ylim(ymin=0.5)
        plt.xlabel("$\epsilon$")
        plt.ylabel("Accuracy (\%)")

        plt.savefig(f"figures/{dataset}_{model}.pdf", bbox_inches='tight')
        legend=plt.legend()
        ax = plt.gca()
        # there will be figures popping up
        plt.show()

# then create a new image
# adjust the figure size as necessary
fig_leg = plt.figure()
ax_leg = fig_leg.add_subplot()
# add the legend from the previous axes
ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=5)
# hide the axes frame and the x/y labels
ax_leg.axis('off')
fig_leg.savefig("figures/legend.pdf", bbox_inches='tight')