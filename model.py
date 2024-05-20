import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import utils


def get_names_and_params(network):
    def _get_names_and_params(module):
        pnames, params = [], []
        for param_name, param in module.named_parameters():
            param_name = param_name.replace(".", "__")
            pnames.append(param_name)
            params.append(param)
        return pnames, params

    lpnames, lparams = _get_names_and_params(network.layers)
    cpnames, cparams = _get_names_and_params(
        network.classifiers[network.classifier_num]
    )
    param_names = lpnames + cpnames
    params = lparams + cparams
    return param_names, params


class MLP(nn.Module):
    def __init__(self, input_size, output_sizes,
                 hidden_size=400,
                 hidden_layer_num=2,
                 hidden_dropout_prob=.5,
                 input_dropout_prob=.2,
                 lamda=40,
                 will_consolidate=False,
                 epochs_per_task=None,
                 opt_name=None,
                 lr=None,
                 seed=None
                 ):
        # Configurations.
        super().__init__()
        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_sizes = output_sizes
        self.lamda = lamda
        self.will_consolidate = will_consolidate
        self.epochs_per_task = epochs_per_task
        self.opt_name = opt_name
        self.lr = lr
        self.seed = seed

        # Layers.
        self.layers = nn.ModuleList([
            # input
            nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
            nn.Dropout(self.input_dropout_prob),
            # hidden
            *((nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
               nn.Dropout(self.hidden_dropout_prob)) * self.hidden_layer_num),
        ])
        # output
        self.classifiers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, output_size)
                for output_size in output_sizes
            ]
        )
        self.classifier_num = 0

    @property
    def name(self):
        return (
            'MLP'
            '-lambda{lamda}'
            '-in{input_size}-out{output_size}'
            '-h{hidden_size}x{hidden_layer_num}'
            '-dropout_in{input_dropout_prob}_hidden{hidden_dropout_prob}'
            '-will_consolidate_{will_consolidate}'
            '-epochs_per_task={epochs_per_task}'
            '-opt_name={opt_name}'
            '-lr={lr}'
            '-seed={seed}'
        ).format(
            lamda=self.lamda,
            input_size=self.input_size,
            output_size=self.output_sizes[0],
            hidden_size=self.hidden_size,
            hidden_layer_num=self.hidden_layer_num,
            input_dropout_prob=self.input_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
            will_consolidate=self.will_consolidate,
            epochs_per_task=self.epochs_per_task,
            opt_name=self.opt_name,
            lr=self.lr,
            seed=self.seed
        )

    def set_classifier_num(self, num):
        if self.classifier_num + 1 == num:
            # Copy weights from previous classifier to this one, so EWC starts
            # where it left off.
            self.classifiers[num].weight.data = \
                self.classifiers[num - 1].weight.data.clone()
            if self.classifiers[num].bias is not None:
                self.classifiers[num].bias.data = \
                    self.classifiers[num - 1].bias.data.clone()
        self.classifier_num = num

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.classifiers[self.classifier_num](x)

    def estimate_fisher(self, dataset, sample_size, batch_size=32):
        # sample loglikelihoods from the dataset.
        data_loader = utils.get_data_loader(dataset, batch_size)
        loglikelihoods = []
        for x, y in data_loader:
            x = x.view(batch_size, -1)
            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
            loglikelihoods.append(
                F.log_softmax(self(x), dim=1)[range(batch_size), y.data]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()

        # Only select the parameters of the active classifier. There will
        # likely be an opportunity/need to copy weights and buffers between
        # successive classifiers soon.
        param_names, params = get_names_and_params(self)

        loglikelihood_grads = zip(*[autograd.grad(
            l, params,
            retain_graph=i < len(loglikelihoods),
            allow_unused=True
        ) for i, l in enumerate(loglikelihoods, 1)])

        loglikelihood_grads = [
            torch.stack(gs) for gs in loglikelihood_grads if gs is not None
        ]
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        param_names, params = get_names_and_params(self)

        for n, p in zip(param_names, params):
            n = n.replace('.', '__')
            self.register_buffer(
                '{}_mean'.format(n), p.data.clone()
            )
            self.register_buffer(
                '{}_fisher'.format(n), fisher[n].data.clone()
            )

    def ewc_loss(self, cuda=False):
        try:
            losses = []
            param_names, params = get_names_and_params(self)
            for n, p in zip(param_names, params):
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_mean'.format(n))
                fisher = getattr(self, '{}_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            return (self.lamda/2)*sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
