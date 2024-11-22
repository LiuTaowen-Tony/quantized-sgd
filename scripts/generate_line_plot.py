# %%
import pandas as pd 
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
session = "act-noise-lr-0.001-int-sto"
runs = api.runs(f"tony_t_liu/{session}")


meta_data = {
    "batch_size": [],
    "act_fl": [],
    "weight_fl": [],
    "lr": [],
}

stat_data = {
    "grad_norm_w1": [],
    "grad_norm_w2": [],
    # "lr": [],
    # "test_acc": [],
    
    # "test_loss": [],
    # "grad_norm_entire": [],
    # "grad_norm_entire_ema": [],
    # "lp_grad_norm": [],
    #"full_grad_norm": [],
}

for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    history = run.history()
    for key in stat_data:
        if key in run.summary._json_dict:
            stat_data[key].append(history[key][-20:].mean())
        else:
            stat_data[key].append(None)

    for key in meta_data:
        if key in run.config:
            meta_data[key].append(run.config[key])
        else:
            meta_data[key].append(None)

runs_df = pd.DataFrame(meta_data | stat_data)

runs_df.head()
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

# is_same_input = (runs_df["same_input"] == True)
# def back_man(i):
#     return runs_df["back_man_width"] == i
# runs_df = runs_df[runs_df["batch_size"] != 1]


def plot(l: pd.DataFrame, label):
    plt.plot(l['batch_size'], l['grad_norm_w2'], marker='o', linestyle='-', label=label)

# # Loop through each man_width and plot lines for both data frames
# for i in [0, 1]:
#     act_df = runs_df[act_man(i) & weight_man(23)]
#     # actlow = act_df[rounding("stochastic") & back_man(i) & ~is_same_input]
#     # actlow_nearest = act_df[rounding("nearest") & back_man(i)]
#     # actlow_same = act_df[rounding("stochastic") & is_same_input]
#     # back_man = runs_df[rounding("stochastic") & act_man(i) & back_man(23) & ~is_same_input]
#     plot(act_df, f"man_width = {i}")
#     # plot(actlow_nearest, f"man_width = {i}, rounding = nearest,")
#     # plot(actlow_same, f"man_width = {i}, rounding = stochastic, same_input = True")
#     # plot(back_man, f"man_width = {i}, rounding = stochastic, back=23")
# fl = "weight_fl"
fl = "act_fl"

df1 = runs_df[[fl, 'batch_size', 'grad_norm_w2']]

df2 = df1.groupby(['batch_size',fl, ]).mean().reset_index()
print(df2)
def act_man(i):
    return df2[fl] == i
# def rounding(i):
#     return runs_df["act_rounding"] == i
def weight_man(i):
    return df2[fl] == i

# biggest_possible_bitwidth = 8

for i in [1,2,3,4,5,6]:
    act_df = df2[act_man(i)]
    # actlow = act_df[rounding("stochastic") & back_man(i) & ~is_same_input]
    # actlow_nearest = act_df[rounding("nearest") & back_man(i)]
    # actlow_same = act_df[rounding("stochastic") & is_same_input]
    # back_man = runs_df[rounding("stochastic") & act_man(i) & back_man(23) & ~is_same_input]
    plot(act_df, f"{fl} precision = 2^-{i}")
    # plot(actlow_nearest, f"man_width = {i}, rounding = nearest,")
    # plot(actlow_same, f"man_width = {i}, rounding = stochastic, same_input = True")
    # plot(back_man, f"man_width = {i}, rounding = stochastic, back=23")

# for i in [2,3]:
#     act_df = df2[act_man(biggest_possible_bitwidth) & weight_man(i)]
#     # actlow = act_df[rounding("stochastic") & back_man(i) & ~is_same_input]
#     # actlow_nearest = act_df[rounding("nearest") & back_man(i)]
#     # actlow_same = act_df[rounding("stochastic") & is_same_input]
#     # back_man = runs_df[rounding("stochastic") & act_man(i) & back_man(23) & ~is_same_input]
#     plot(act_df, f"weight possible values = {i}")

# act_df = df2[act_man(biggest_possible_bitwidth) & weight_man(biggest_possible_bitwidth)]
# actlow = act_df[rounding("stochastic") & back_man(i) & ~is_same_input]
# actlow_nearest = act_df[rounding("nearest") & back_man(i)]
# actlow_same = act_df[rounding("stochastic") & is_same_input]
# back_man = runs_df[rounding("stochastic") & act_man(i) & back_man(23) & ~is_same_input]
# plot(act_df, f"baseline")

# base_line = runs_df[(runs_df["weight_man_width"] == 23) & (runs_df["act_man_width"] == 23)]
# plt.plot(base_line['batch_size'], base_line['full_grad_norm'], marker='o', linestyle='-', label='base line')


# Adding labels and title
plt.xlabel('Batch Size')
plt.ylabel('Final Gradient Norm')
plt.title('Gradient Norm vs. Batch Size')
plt.grid(True)

# Adding the legend
plt.legend()

# Optionally, set the x-axis to log scale if the batch sizes are not uniformly distributed
# plt.xscale('log')
# plt.hlines([0.0085, 0.0069, 0.0053, 0.0038], 500, 4096)




plt.show()


# %%



