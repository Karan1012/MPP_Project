
def copy_parameters(from_params, to_params):
    for tp, fp in zip( [p for p in to_params], from_params):
        tp.data.copy_(fp.data.clone().detach())