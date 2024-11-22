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


def compute_acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    acc = torch.sum(preds == labels).item() / len(labels)
    return acc

class quant_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.Function, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, act_quantizer: quant.QuantMethod) -> torch.Tensor:
        ctx.act_quantizer = act_quantizer
        input_shape = input.shape
        input = input.view(-1, input_shape[-1]) 
        ctx.save_for_backward(input, weight, bias)
        result =  input.mm(weight.t()) 
        result.view(*input_shape[:-1], -1)
        result += bias
        return result



    @staticmethod
    def backward(ctx: torch.autograd.Function, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        input, weight, bias = ctx.saved_tensors
        act_quantizer: quant.QuantMethod = ctx.act_quantizer
        grad_output = grad_output.view(-1, grad_output.shape[-1])

        grad_output = act_quantizer.quant(grad_output)
        input = act_quantizer.quant(input)

        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        grad_bias = grad_output.sum(0)

        grad_input = grad_input.view(*grad_output.shape[:-1], -1)
        return grad_input, grad_weight, grad_bias, None

class BActQuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bact_quantizer: quant.QuantMethod, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.bact_quantizer = bact_quantizer
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quant_linear.apply(input, self.weight, self.bias, self.bact_quantizer)


class MLP(nn.Module):
    def __init__(self, neurons: list[int]) -> None:
        super().__init__()
        self.neurons = neurons
        self.layers = torch.nn.ModuleList()
        for i in range(len(neurons) - 1):
            self.layers.append(BActQuantLinear(neurons[i], neurons[i + 1], quant.FP32))
            if i != len(neurons) - 2:
                self.layers.append(nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        self.logits = x
        return x

    def find_noise_grad(self, X: torch.Tensor, y: torch.Tensor, act_fl: int = 100, scaling: float = 1.0, quantizer_type: str = "int-sto") -> list[torch.Tensor]:
        if quantizer_type == "noise":
            var = (2 ** -act_fl) / 48
            std = math.sqrt(var)
            quantizer = quant.NoiseQuant(std=std,)
        elif quantizer_type == "int-sto":
            quantizer = quant.IntQuant(fl=act_fl)
        elif quantizer_type == "int-deter":
            quantizer = quant.IntQuant(fl=act_fl, round_mode="nearest")

        if act_fl == 100:
            quantizer = quant.FPQuant()

        for layer in self.layers:
            if isinstance(layer, BActQuantLinear):
                layer.bact_quantizer = quantizer

        self.forward(X)
        loss = nn.CrossEntropyLoss( reduction='sum')(self.logits, y,)
        loss *= scaling
        self.zero_grad()
        loss.backward()

        divisor = len(y) * scaling
        for param in self.parameters():
            if param.grad is not None:
                param.grad = param.grad / divisor
        return [param.grad for param in self.parameters()]

    def weight_update(self, grads: list[torch.Tensor], lr: float) -> None:
        for param, grad in zip(self.parameters(), grads):
            param.data -= lr * grad

def test_result(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, torch.Tensor]:
    loss = nn.CrossEntropyLoss( reduction='sum')(logits, labels,)
    accuracy = compute_acc(logits, labels)
    return (accuracy, loss / len(labels))


def sanity_check(model: MLP, X: torch.Tensor, y: torch.Tensor) -> None:
    X2 = X.repeat(2, 1)
    y2 = y.repeat(2)
    # 1. copy the input, same gradient
    dw1, *_ = model.find_noise_grad(X, y)
    dw1_2, *_ = model.find_noise_grad(X2, y2)
    print(dw1.norm(), dw1_2.norm())
    assert math.isclose(dw1.norm().item(), dw1_2.norm().item(), rel_tol=1e-2)

    # 2. full precision scaling doesn't change the gradient
    dw1, *_= model.find_noise_grad(X, y, scaling=1.0)
    dw1_2, *_ = model.find_noise_grad(X, y, scaling=2.0)
    print(dw1.norm(), dw1_2.norm())
    assert math.isclose(dw1.norm().item(), dw1_2.norm().item(), rel_tol=1e-2)

    # 3. whatever the quantizer type and precision, the forward gives the same
    model.find_noise_grad(X, y, act_fl=1)
    logits1 = model.logits
    model.find_noise_grad(X, y, act_fl=2)
    logits2 = model.logits
    model.find_noise_grad(X, y, act_fl=3, quantizer_type="int-deter")
    logits3 = model.logits
    print(logits1.norm(), logits2.norm(), logits3.norm())
    assert math.isclose(logits1.norm().item(), logits2.norm().item(), rel_tol=1e-2)



    # # set weight2 to be itself equvalent to not set
    # qweigth2 = model.weight2
    # logits = model.forward(X)
    # dw1, *_= model.find_noise_grad(y, qweight2=qweigth2)
    # logits2 = model.forward(X2)
    # dw1_2, *_ = model.find_noise_grad(y2, qweight2=None)
    # print(dw1.norm(), dw1_2.norm())
    # assert math.isclose(dw1.norm().item(), dw1_2.norm().item(), rel_tol=1e-2)


    # # 3. weight quantization, change batch size doesn't change the gradient
    # qweigth2 = quant.IntQuant(fl=2).quant(model.weight2)
    # logits = model.forward(X)
    # dw1, *_= model.find_noise_grad(y, qweight2=qweigth2)
    # logits2 = model.forward(X2)
    # dw1_2, *_ = model.find_noise_grad(y2, qweight2=qweigth2)
    # print(dw1.norm(), dw1_2.norm())
    # assert math.isclose(dw1.norm().item(), dw1_2.norm().item(), rel_tol=1e-2)

def train_step(model: MLP, X: torch.Tensor, y: torch.Tensor, args: Args, logger: log_util.Logger) -> None:
    grads = model.find_noise_grad(X, y, args.act_fl, args.scaling, args.quantizer_type)
    model.weight_update(grads, args.lr)
    acc, loss = test_result(model.logits, y)
    result = {"train_acc": acc, "train_loss": loss}
    bar.update(1)
    bar.set_postfix(result)
    logger.log(result)
    if stepi % (logger_args.wandb_interval) == 0 or stepi >= args.steps:
        grads = model.find_noise_grad(X_train, y_train) 
        acc, loss = test_result(model.logits, y_train)
        result = {"train_acc": acc, "train_loss": loss}
        for i, g in enumerate(grads):
            result[f"grad_norm_{i}"] = g.norm().item()
        logger.log_same_iter(result)

data_loader = load_data.get_data_loader(args.dataset)
X_train, y_train, X_test, y_test = data_loader(device)
if not args.do_sampling:
    X_train= X_train[:512]
    y_train = y_train[:512]

def str_config_to_list(config: str) -> list[int]:
    str_numbers = config.split(",")
    return list(map(int, str_numbers))

model_config = str_config_to_list(args.model_config)

model = MLP(model_config).to(device)
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
