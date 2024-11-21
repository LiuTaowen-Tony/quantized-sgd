import torch.nn.grad
import tqdm
import torch
from torch import nn
from low_precision_utils import quant
from ml_utils import log_util
from utils import load_data
from dataclasses import dataclass
from ml_utils.args import DataClassArgumentParser

@dataclass
class Args:
    dataset: str = "mnist"
    lr: float = 0.003
    steps: int = 20000
    scaling: float = 1.0
    batch_size: int = 2048
    weight_fl: int = 100
    act_fl: int = 100
    do_sampling: bool = False
    quantizer_type: str = "int_sto"

parser = DataClassArgumentParser(Args)
args: Args = parser.parse_args_into_dataclasses()[0]

dtype = torch.float32
device = torch.device("cuda")

def softmax_crossentropy_backward(logits, labels):
    batch_size = logits.size(0)
    probs = torch.softmax(logits, dim=-1)
    labels_one_hot = torch.zeros_like(logits)
    labels_one_hot[torch.arange(batch_size), labels] = 1.0
    grad = (probs - labels_one_hot) 
    return grad

def softmax_crossentropy(logits, labels):
    batch_size = logits.size(0)
    probs = torch.softmax(logits, dim=-1)
    labels_one_hot = torch.zeros_like(logits)
    labels_one_hot[torch.arange(batch_size), labels] = 1.0
    loss = -torch.sum(labels_one_hot * torch.log(probs + 1e-8)) 
    return loss

def compute_acc(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    acc = torch.sum(preds == labels).item() / len(labels)
    return acc

class Model():
    def __init__(self):
        super().__init__()
        self.weight1 = torch.rand(784, 100, dtype=dtype, device=device) * (2 / (100 + 784))
        self.weight2 = torch.rand(100, 10, dtype=dtype, device=device) * (2 / (100 + 10))
        self.bias1 = torch.zeros(100, dtype=dtype, device=device)
        self.bias2 = torch.zeros(10, dtype=dtype, device=device)
        self.z1 = None
        self.a1 = None
        self.z2 = None

    def forward(self, x):
        self.a0 = x
        z1 = x @ self.weight1 + self.bias1
        self.z1 = z1
        a1 = torch.relu(z1)
        self.a1 = a1
        z2 = a1 @ self.weight2 + self.bias2
        self.z2 = z2
        return z2

    def act_noisy_back(self, y, act_precision_level=100, scaling=1.0, qweight2=None, quantizer_type="int_sto"):
        if quantizer_type == "int_sto":
            quantizer = quant.IntQuant(fl=act_precision_level)
        elif quantizer_type == "int_deter":
            quantizer = quant.IntQuant(fl=act_precision_level, stochastic=False)
        elif quantizer_type == "noise":
            # keep same scala of variance of individual quantization
            var = (2 ** act_precision_level)
            std = torch.sqrt(var)
            quantizer = quant.NoiseQuant(std=std)
            
        if act_precision_level == 100:
            quantizer.quant = lambda x: x


        if qweight2 is None:
            qweight2 = self.weight2
        dz2 = softmax_crossentropy_backward(self.z2, y)
        dz2 = dz2 * scaling
        qa1 = quantizer.quant(self.a1)
        qdz2 = quantizer.quant(dz2)
        qa0 = quantizer.quant(self.a0)
        da1 = (qdz2) @ (qweight2).t()
        
        dz1 = da1 * (self.z1 > 0).to(torch.float32)
        qdz1 = quantizer.quant(dz1)
        self.dz2 = dz2
        self.dz1 = dz1
        self.da1 = da1

        divider = len(y) * scaling

        dw2 = (qa1).t() @ (qdz2) / divider
        db2 = dz2.sum(dim=0) / divider
        dw1 = (qa0).t() @ (qdz1) / divider
        db1 = dz1.sum(dim=0) / divider
        return dw1, db1, dw2, db2


    def weight_update(self, dw1, db1, dw2, db2, lr):
        self.weight1 -= lr * dw1 
        self.bias1 -= lr * db1 
        self.weight2 -= lr * dw2 
        self.bias2 -= lr * db2



def test_result(logits, labels):
    loss = softmax_crossentropy(logits, labels)
    accuracy = compute_acc(logits, labels)
    return (accuracy, loss / len(labels))


def sanity_check(model: Model, X, y):
    X2 = X.repeat(2, 1)
    y2 = y.repeat(2)
    # 1. copy the input, same gradient
    logits = model.forward(X)
    dw1, *_ = model.act_noisy_back(y)
    logits2 = model.forward(X2)
    dw1_2, *_ = model.act_noisy_back(y2)
    print(dw1.norm(), dw1_2.norm())

    # 2. full precision scaling doesn't change the gradient
    logits = model.forward(X)
    dw1, *_= model.act_noisy_back(y, scaling=1.0)
    logits2 = model.forward(X)
    dw1_2, *_ = model.act_noisy_back(y, scaling=2.0)
    print(dw1.norm(), dw1_2.norm())

    # 3. weight quantization, change batch size doesn't change the gradient
    qweigth2 = quant.IntQuant(fl=2).quant(model.weight2)
    logits = model.forward(X)
    dw1, *_= model.act_noisy_back(y, qweight2=qweigth2)
    logits2 = model.forward(X2)
    dw1_2, *_ = model.act_noisy_back(y2, qweight2=qweigth2)
    print(dw1.norm(), dw1_2.norm())

def train_step(model: Model, X, y, args, logger):
    logits = model.forward(X)
    if args.weight_fl == 100:
        qweight2 = model.weight2
    else:
        qweight2 = quant.IntQuant(fl=args.weight_fl).quant(model.weight2)

    dw1, db1, dw2, db2= model.act_noisy_back(y, args.act_fl, args.scaling, qweight2, args.quantizer_type)
    model.weight_update(dw1, db1, dw2, db2, args.lr)
    acc, loss = test_result(logits, y)
    result = {"train_acc": acc, "train_loss": loss}
    bar.update(1)
    bar.set_postfix(result)
    logger.log(result)
    if stepi % (args.wandb_interval) == 0 or stepi >= args.steps:
        logtis = model.forward(X_train)
        dw1, db1, dw2, db2 = model.act_noisy_back(y_train) 
        acc, loss = test_result(model.z2, y_train)
        grad_norm_w1 = torch.sqrt((dw1 ** 2).sum()).item()
        grad_norm_w2 = torch.sqrt((dw2 ** 2).sum()).item()
        logger.log_same_iter({
            "grad_norm_w1": grad_norm_w1,
            "grad_norm_w2": grad_norm_w2,
            "test_acc": acc,
            "test_loss": loss,
        })

data_loader = load_data.get_data_loader(args.dataset)
X_train, y_train, X_test, y_test = data_loader(device)
if not args.do_sampling:
    X_train= X_train[:512]
    y_train = y_train[:512]
model = Model()
bar = tqdm.tqdm(range(args.steps))
stepi = 0
logger = log_util.Logger.from_args(args)
sanity_check(model, X_train[:512], y_train[:512])

while True:
    with torch.no_grad():
        if args.do_sampling:
            for X, y in load_data.getBatches(X_train, y_train, args.batch_size):
                if stepi >= args.steps:
                    exit()
                train_step(model, X, y, args, logger)
                stepi += 1
        else:
            if stepi >= args.steps:
                exit()
            X = X_train.repeat(int(args.batch_size / 512), 1)
            y = y_train.repeat(int(args.batch_size / 512))
            train_step(model, X, y, args, logger)
            stepi += 1
