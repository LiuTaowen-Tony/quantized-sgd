from ml_utils.job_queue import JobQueue

# Define the parameter space
params_space = {
    "act_fl": [1,2,3,4],
    "weight_fl": [100],
    "batch_size": [512,1024, 2048,4096],
    "lr": [0.001, ],
    "steps": [20000],
    "quantizer_type": ["int-sto"],
    "do_sampling": ["no"],
    "runs": [i for i in range(4)],
    "model_config": [ "784,100,10", "784,100,100,10", "784,100,100,100,10"],
}

# Function to create and run the command
def get_cmd(params):
    # Generate the command to run the experiment
    cmd = ["python", "mnist_mlp_simplify.py", ]
    json_dict = {}
    del params["runs"]
    for key, value in params.items():
        cmd.append(f"--{key} {value}")
    cmd.append(f"--do_sampling no")
    cmd.append(f"--experiment_name improve-act-{params['quantizer_type']}-sampling-{params['do_sampling']}-model-{params['model_config']} ")
    # Run the command and return the process
    return " ".join(cmd)

queue = JobQueue([0,1,2,3], 2)

# Generate all possible combinations of parameters
cmds = queue.expand_param_space(params_space, get_cmd)
for i in cmds:
    print(i)

queue.map(cmds)

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
    

        