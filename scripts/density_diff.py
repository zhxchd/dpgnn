import json
import logging
import sys
from datetime import datetime

sys.path.append('../src')
import numpy as np
import blink
from data import make_dataset

# setup logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"log/density_"+datetime.now().strftime('%y%m%d_%H%M%S.txt')),
        logging.StreamHandler(sys.stdout)
    ])

logging.info(f"Experiments on the density difference between actual graph and estimated graph.")

datasets = ["cora", "citeseer", "lastfm"]
eps_list = [1,2,3,4,5,6,7,8]
results = {}

for data in datasets:
    graph = make_dataset(data, root="../data")
    linkless_graph = graph.clone()
    linkless_graph.edge_index = None
    logging.info(f"Density of actual graph {data} is {graph.num_edges}")
    for eps in eps_list:
        A_hat_l1 = np.zeros(30)
        for i in range(30):
            client = blink.Client(eps=eps, delta=0.1, data=graph)
            server = blink.Server(eps=eps, delta=0.1, data=linkless_graph)
            priv_adj, priv_deg = client.AddLDP()
            server.receive(priv_adj, priv_deg)
            server.estimate()
            A_hat_l1[i] = server.est_edge_index.shape[1]
        logging.info(f"Density on {data} with eps={eps}: {A_hat_l1.mean()} ({A_hat_l1.std()})")
        logging.info(f"Density on {data} with eps={eps}: Saving result to output/density.json")
        with open("output/density.json") as f:
            d = json.load(f)
        if data not in d:
            d[data] = {}
        d[data][str(eps)] = [A_hat_l1.mean(), A_hat_l1.std()]
        with open('output/density.json', 'w') as fp:
            json.dump(d, fp, indent=2)