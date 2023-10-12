#%%
import numpy as np
import itertools

# data
input_label = ('heavy', 'light', 'naphtha')
product_label = ("butadiene", "propene", "ethylene")

# optimization target for single product
optim = {
    "butadiene": lambda x:x[0],
    "propene": lambda x:x[1],
    "ethylene": lambda x:x[2]
}

# Light/Medium/Heavy steam cracking
crack_mode_label = ('crack1', 'crack2', 'crack3')

# labeling: heavy, light, naphtha
distill_data = {
    'heavy': np.array([2.5, 0.45, 0.15]),
    'medium': np.array([0.1, 0.5, 1.5]),
    'light': np.array([0.1, 0.2, 0.3])/1.5,
    'bc': np.array([0.15, 0.5, 0.2])/0.5
}

# 1.2 if use multiblock, 1.0 if use chemical reactor
cracking_eff = 1.2

# labeling: (output)[heavy, light, naphtha]
crack_transfer_data = {
    'heavy': {
        'crack1': np.array([0, 300, 50]) / 1000 * cracking_eff,
        'crack2': np.array([0, 200, 200]) / 1000 * cracking_eff,
        'crack3': np.array([0, 100, 125]) / 1000 * cracking_eff,
    },
    'light': {
        'crack1': np.array([150, 0, 400]) / 1000 * cracking_eff,
        'crack2': np.array([100, 0, 250]) / 1000 * cracking_eff,
        'crack3': np.array([50, 0, 100]) / 1000 * cracking_eff,
    },
    'naphtha': {
        'crack1': np.array([75, 150, 0]) / 1000 * cracking_eff,
        'crack2': np.array([50, 100, 0]) / 1000 * cracking_eff,
        'crack3': np.array([25, 50, 0]) / 1000 * cracking_eff,
    },
}

#labeling: (output)[butadiene, propene, ethylene]
crack_product_data = {
    'heavy': {
        'crack1': np.array([15, 30, 50]) / 1000 * cracking_eff,
        'crack2': np.array([25, 50, 75]) / 1000 * cracking_eff,
        'crack3': np.array([50, 100, 150]) / 1000 * cracking_eff,
    },
    'light': {
        'crack1': np.array([60, 150, 50]) / 1000 * cracking_eff,
        'crack2': np.array([75, 200, 150]) / 1000 * cracking_eff,
        'crack3': np.array([50, 250, 250]) / 1000 * cracking_eff,
    },
    'naphtha': {
        'crack1': np.array([150, 200, 200]) / 1000 * cracking_eff,
        'crack2': np.array([100, 400, 350]) / 1000 * cracking_eff,
        'crack3': np.array([50, 300, 500]) / 1000 * cracking_eff,
    },
}

#%%
def get_transfer_matrix(idx):
    return np.transpose(np.stack([crack_transfer_data[input_label[i]][idx[i]] for i in range(3)]))

def get_product_matrix(idx):
    return np.transpose(np.stack([crack_product_data[input_label[i]][idx[i]] for i in range(3)]))

def optimize(oil_type, score):
    max_score = float("-inf")
    max_idx = ()
    crack_mode_iter = itertools.product(crack_mode_label, crack_mode_label, crack_mode_label)
    for idx in crack_mode_iter:
        transfer_matrix = get_transfer_matrix(idx)
        product_matrix = get_product_matrix(idx)
        total_cracking_consumption = np.linalg.inv(np.eye(3) - transfer_matrix) @ distill_data[oil_type]
        total_product = product_matrix @ total_cracking_consumption
        current_score = score(total_product)
        if current_score > max_score:
            max_score = current_score
            max_idx = idx

    transfer_matrix = get_transfer_matrix(max_idx)
    product_matrix = get_product_matrix(max_idx)
    total_cracking_consumption = np.linalg.inv(np.eye(3) - transfer_matrix) @ distill_data[oil_type]
    total_product = product_matrix @ total_cracking_consumption
    current_score = score(total_product)
    print("Optimized data for " + oil_type + " oil")
    print("Cracking setup:\n" + "\n".join(["{}: {}, input(B) per 1B oil: {:.4f}".format(input_label[i], max_idx[i], total_cracking_consumption[i]) for i in range(3)]) + "\n\n")
    print("Product(B) per 1B oil:\n" + "\n".join(["{}: {:.4f}".format(product_label[i], total_product[i]) for i in range(len(product_label))]) + "\n\n")

#%%
# example
optimize("light", optim["propene"])
optimize("bc", optim["propene"])
