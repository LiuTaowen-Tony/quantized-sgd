import json
import itertools
import subprocess
import time
import copy

from ml_utils import job_queue 
from threading import Thread 

JobQueue = job_queue.JobQueue

# Define the parameter space
params_space = {
    "act_fl": [1,2,3,4],
    "weight_fl": [100],
    "batch_size": [512,1024, 2048,4096],
    "lr": [0.002, ],
    "steps": [20000],
    "quantizer_type": ["noise"],
}


# Function to create and run the command
def get_cmd(params):
    # Generate the command to run the experiment
    cmd = ["python", "exp4.py", ]
    json_dict = {}
    for key, value in params.items():
        cmd.append(f"--{key} {value}")
    cmd.append(f"--do_sampling no")
    cmd.append(f"--experiment_name act-noise-lr-0.0015-gaussian ")
    # Run the command and return the process
    return " ".join(cmd)

queue = JobQueue(4, 1)

# Generate all possible combinations of parameters
cmds = []
for values in itertools.product(*params_space.values()):
    # Create a dictionary of parameters
    params = {key: value for key, value in zip(params_space.keys(), values)}
    # Add the job to the queue
    cmds.append(get_cmd(params))


# Run the jobs
# print(cmds)
# for cmd in cmds:
#     threads = []
#     print(cmd)
#     for device in range(4):
#         cmd_ = f"CUDA_VISIBLE_DEVICES={device } {cmd}"
#         thread = Thread(target= lambda : subprocess.run(cmd_, shell=True))
#         thread.start()
#         threads.append(thread)
#     for thread in threads:
#         thread.join()
    

        