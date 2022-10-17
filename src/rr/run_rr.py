import numpy as np
import rr

def run_rr(graph, linkless_graph, model_name, eps, hp, num_trials):
    val_loss = np.zeros(num_trials)
    test_acc = np.zeros(num_trials)
    for i in range(num_trials):
        client = rr.Client(eps=eps, data=graph)
        server = rr.Server(eps=eps, data=linkless_graph)

        priv_adj = client.AddLDP()
        server.receive(priv_adj)
        server.estimate()
        log = server.fit(model_name, hparam=hp)
        val_loss[i] = log[:,1].min()
        test_acc[i] = log[np.argmin(log[:,1])][2]
    return val_loss, test_acc