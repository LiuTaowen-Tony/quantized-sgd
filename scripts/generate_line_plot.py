# %%
import pandas as pd 
import wandb
from ml_utils.log_util import ExperimentMetrics

a = ["int-sto"]
b = [ "no"]
c = [ "784,100,10"]

# experiment_name = "act-int-sto-sampling-yes-model-784,100,100,100,10"
def process_experiment(i,j,k:str):
    experiment_name = f"improve-act-{i}-sampling-{j}-model-{k}"
    metrics = ExperimentMetrics(f"experiment_metrics/{experiment_name}", lazy=True)


    # print(metrics.runs_summary)
    # remove even rows except for the 0th row

    metrics._load_runs()
    # try:
    #     metrics._load_runs()
    # except:
    #     metrics.runs = {}
    #     metrics.runs_summary = metrics.runs_summary[metrics.runs_summary.index % 2 == 0]
    #     metrics._load_runs()



    new_metrics = {f"grad_norm_{i}": [] for i in range(k.count(',') * 2) }
    for id, run in metrics.runs.items():
        for key in new_metrics:
            new_metrics[key].append(run[key][-2000:].mean())

    print(new_metrics)

    metrics.runs_summary = metrics.runs_summary.assign(**new_metrics)

    print(metrics.runs_summary)
    metrics.save("experiment_metrics", experiment_name)


def make_plot_for_experiment(i,j,k):
    experiment_name = f"improve-act-{i}-sampling-{j}-model-{k}"
    metrics = ExperimentMetrics(f"/home/tl2020/quantized-sgd/experiment_metrics/{experiment_name}", lazy=True)
    metrics._load_runs()
    runs_df = metrics.runs_summary
    import matplotlib.pyplot as plt

    fl = "act_fl"
    print(runs_df)
    df1 = runs_df[[fl, 'batch_size', *[f'grad_norm_{i}' for i in range(k.count(',') * 2)] ]]
    df2 = df1.groupby(['batch_size',fl, ]).mean().reset_index()
    
    def act_man(i):
        return df2[fl] == i
    def plot(l: pd.DataFrame, ii, label):
        plt.plot(l['batch_size'], l[f'grad_norm_{ii}'], marker='o', linestyle='-', label=label)

    for ii in range(k.count(',') * 2):
        plt.figure(figsize=(10, 8))
        for i in [1,2,3,4,5,6]:
            act_df = df2[act_man(i)]
            plot(act_df, ii, f"{fl} precision = 2^-{i}")

        plt.xlabel('Batch Size')
        plt.ylabel('Final Gradient Norm')
        plt.title('Gradient Norm vs. Batch Size')
        plt.grid(True)

        plt.legend()

        plt.show()
        plt.savefig(f"plots/{experiment_name}_grad_norm_{ii}.png")


for i in a:
    for j in b:
        for k in c:
            process_experiment(i,j,k)
            make_plot_for_experiment(i,j,k)