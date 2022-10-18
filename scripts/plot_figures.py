import json
from matplotlib import pyplot as plt

from matplotlib import rc

# there will be 9 figures
with open("output/results.json") as f:
    blink_res = json.load(f)
with open("output/bl_results.json") as f:
    bl_res = json.load(f)

for dataset in ["cora", "citeseer", "lastfm"]:
    for model in ["gcn", "graphsage", "gat"]:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif"
        })
        rc("figure", figsize=(4,3))

        x = ["1","2","3","4","5","6","7","8"]

        # draw non private accuracy (upper bound)
        plt.plot(x, [blink_res[dataset][model]["None"][0] for i in x], linestyle=':', marker=" ", color="green", label="$\epsilon=\infty$ (non-private)")

        # draw pure privacy MLP (lower bound)
        plt.plot(x, [blink_res[dataset]["mlp"]["None"][0] for i in x], linestyle=':', marker=" ", color="red", label="$\epsilon=0$ (MLP)")

        # draw blink performance on 1 to 8
        plt.plot(x, [blink_res[dataset][model][i][0] for i in x], marker=" ", color="blue", label="Blink (ours)")
        plt.errorbar(x, [blink_res[dataset][model][i][0] for i in x], yerr=[blink_res[dataset][model][i][1] for i in x], fmt='|', elinewidth=2, color="blue")
        
        # draw blink performance on 1 to 8
        plt.plot(x, [bl_res[dataset][model]["ldpgcn"][i][0] for i in x], marker=" ", color="orange", label="L-DPGCN")
        plt.errorbar(x, [bl_res[dataset][model]["ldpgcn"][i][0] for i in x], yerr=[bl_res[dataset][model]["ldpgcn"][i][1] for i in x], fmt='|', elinewidth=2, color="orange")

        plt.ylim(ymin=0.3, ymax=0.9)
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
ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=4)
# hide the axes frame and the x/y labels
ax_leg.axis('off')
fig_leg.savefig("figures/legend.pdf", bbox_inches='tight')