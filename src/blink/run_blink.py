import numpy as np
import blink

def run_blink(graph, linkless_graph, model_name, eps, hp, num_trials):
    val_f1 = np.zeros(num_trials)
    test_f1 = np.zeros(num_trials)
    # non private, there's no client
    if eps == None:
        for i in range(num_trials):
            server = blink.Server(None, None, graph)
            log = server.fit(model_name, hparam=hp) # [train_loss, val_loss, val_f1, test_f1]
            step = np.argmin(log[:,1]) # early stopping at lowest validation loss
            val_f1[i] = log[step,2]
            test_f1[i] = log[step,3]
    # link LDP with blink
    else:
        for i in range(num_trials):
            client = blink.Client(eps=eps, delta=hp["delta"], data=graph)
            server = blink.Server(eps=eps, delta=hp["delta"], data=linkless_graph)

            priv_adj, priv_deg = client.AddLDP()
            server.receive(priv_adj, priv_deg)
            server.estimate()
            log = server.fit(model_name, hparam=hp)
            step = np.argmin(log[:,1]) # early stopping at lowest validation loss
            val_f1[i] = log[step,2]
            test_f1[i] = log[step,3]
    return val_f1, test_f1 # validation f1 is used in grid search for model selection