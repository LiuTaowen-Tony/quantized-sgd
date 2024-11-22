import math
import torch.nn.grad
import tqdm
import torch
from torch import nn
from low_precision_utils import quant
from ml_utils import log_util
from utils import load_data
from dataclasses import dataclass
from ml_utils.args import DataClassArgumentParser
from typing import Tuple

@dataclass
class Args:
    dataset: str = "mnist"
    model_config: str = "784,100,100,10"
    lr: float = 0.003
    steps: int = 20000
    scaling: float = 1.0
    batch_size: int = 2048
    weight_fl: int = 100
    act_fl: int = 100
    do_sampling: bool = False
    quantizer_type: str = "int_sto"

def get_args() -> Tuple[Args, log_util.LoggerArgs]:
    parser = DataClassArgumentParser((Args, log_util.LoggerArgs))
    parsed = parser.parse_args_into_dataclasses()
    return parsed
args, logger_args = get_args()

dtype = torch.float32
device = torch.device("cuda")


def compute_acc(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    acc = torch.sum(preds == labels).item() / len(labels)
    return acc

class MLP(nn.Module):
    def __init__(self, neurons):
        super().__init__()
        self.neurons = neurons
        self.layers = torch.nn.ModuleList()
        for i in range(len(neurons) - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            if i != len(neurons) - 2:
                self.layers.append(nn.ReLU())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Wrapped(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quant_wrapper = quant.QuantWrapper(model, quant.FP32_SCHEME)
        self.logits = None

    def forward(self, x):
        logits = self.model(x)
        self.logits = logits
        return logits
    
    @property
    def weight2(self):
        return self.quant_wrapper.module.layers[-1].weight.data

    def act_noisy_back(self, y, act_fl = 100, scaling = 1.0,  qweight2 = None, quantizer_type = "int-sto"):
        save_weight = self.weight2
        if qweight2 is not None:
            self.quant_wrapper.module.layers[-1].weight.data = qweight2

        loss = nn.CrossEntropyLoss( reduction='sum')(self.logits, y,)
        loss *= scaling
        if quantizer_type == "noise":
            quantizer = quant.NoiseQuant(fl=act_fl, scaling=scaling)
        elif quantizer_type == "int-sto":
            quantizer = quant.IntQuant(fl=act_fl)
        elif quantizer_type == "int-deter":
            quantizer = quant.IntQuant(fl=act_fl, stochastic=False)
        
        if act_fl == 100:
            quantizer = quant.FPQuant()

        self.quant_wrapper.apply_quant_scheme(quant.QuantScheme(
            bact = quantizer,
            goact = quantizer,
            goweight = quantizer,
        ))

        self.quant_wrapper.zero_grad()
        loss.backward()

        divisor = len(y) * scaling
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = param.grad / divisor
        if qweight2 is not None:
            self.quant_wrapper.module.layers[-1].weight.data = save_weight
        return [param.grad for param in self.model.parameters()]

    def weight_update(self, grads, lr):
        for param, grad in zip(self.model.parameters(), grads):
            param.data -= lr * grad

def test_result(logits, labels):
    loss = nn.CrossEntropyLoss( reduction='sum')(logits, labels,)
    accuracy = compute_acc(logits, labels)
    return (accuracy, loss / len(labels))


def sanity_check(model: Wrapped, X, y):
    X2 = X.repeat(2, 1)
    y2 = y.repeat(2)
    # 1. copy the input, same gradient
    logits = model.forward(X)
    dw1, *_ = model.act_noisy_back(y)
    logits2 = model.forward(X2)
    dw1_2, *_ = model.act_noisy_back(y2)
    print(dw1.norm(), dw1_2.norm())
    assert math.isclose(dw1.norm().item(), dw1_2.norm().item(), rel_tol=1e-2)

    # 2. full precision scaling doesn't change the gradient
    logits = model.forward(X)
    dw1, *_= model.act_noisy_back(y, scaling=1.0)
    logits2 = model.forward(X)
    dw1_2, *_ = model.act_noisy_back(y, scaling=2.0)
    print(dw1.norm(), dw1_2.norm())
    assert math.isclose(dw1.norm().item(), dw1_2.norm().item(), rel_tol=1e-2)

    # set weight2 to be itself equvalent to not set
    qweigth2 = model.weight2
    logits = model.forward(X)
    dw1, *_= model.act_noisy_back(y, qweight2=qweigth2)
    logits2 = model.forward(X2)
    dw1_2, *_ = model.act_noisy_back(y2, qweight2=None)
    print(dw1.norm(), dw1_2.norm())
    assert math.isclose(dw1.norm().item(), dw1_2.norm().item(), rel_tol=1e-2)


    # 3. weight quantization, change batch size doesn't change the gradient
    qweigth2 = quant.IntQuant(fl=2).quant(model.weight2)
    logits = model.forward(X)
    dw1, *_= model.act_noisy_back(y, qweight2=qweigth2)
    logits2 = model.forward(X2)
    dw1_2, *_ = model.act_noisy_back(y2, qweight2=qweigth2)
    print(dw1.norm(), dw1_2.norm())
    assert math.isclose(dw1.norm().item(), dw1_2.norm().item(), rel_tol=1e-2)

def train_step(model: Wrapped, X, y, args: Args, logger: log_util.Logger):
    logits = model.forward(X)
    if args.weight_fl == 100:
        qweight2 = model.weight2
    else:
        qweight2 = quant.IntQuant(fl=args.weight_fl).quant(model.weight2)

    grads = model.act_noisy_back(y, args.act_fl, args.scaling, qweight2, args.quantizer_type)
    model.weight_update(grads, args.lr)
    acc, loss = test_result(logits, y)
    result = {"train_acc": acc, "train_loss": loss}
    bar.update(1)
    bar.set_postfix(result)
    logger.log(result)
    if stepi % (logger_args.wandb_interval) == 0 or stepi >= args.steps:
        logits = model.forward(X_train)
        grads = model.act_noisy_back(y_train) 
        acc, loss = test_result(logits, y_train)
        result = {"train_acc": acc, "train_loss": loss}
        for i, g in enumerate(grads):
            result[f"grad_norm_{i}"] = g.norm().item()
        logger.log(result)


data_loader = load_data.get_data_loader(args.dataset)
X_train, y_train, X_test, y_test = data_loader(device)
if not args.do_sampling:
    X_train= X_train[:512]
    y_train = y_train[:512]

def str_config_to_list(config):
    str_numbers = config.split(",")
    return list(map(int, str_numbers))

model_config = str_config_to_list(args.model_config)

model = MLP(model_config)
model = Wrapped(model).to(device)

bar = tqdm.tqdm(range(args.steps))
stepi = 0
logger = log_util.Logger.from_args(logger_args, hparam_or_hparam_list=args)
sanity_check(model, X_train[:512], y_train[:512])

while True:
    if args.do_sampling:
        for X, y in load_data.getBatches(X_train, y_train, args.batch_size):
            if stepi >= args.steps:
                exit()
            train_step(model, X, y, args, logger)
            stepi += 1
    else:
        if stepi >= args.steps:
            logger.save_experiment()
            exit()
        X = X_train.repeat(int(args.batch_size / 512), 1)
        y = y_train.repeat(int(args.batch_size / 512))
        train_step(model, X, y, args, logger)
        stepi += 1
