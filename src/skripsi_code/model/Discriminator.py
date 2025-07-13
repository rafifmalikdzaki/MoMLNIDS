import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Tuple, Optional
from torchsummary import summary


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambd: float, reverse: bool = True) -> Tensor:
        ctx.lambd = lambd
        ctx.reverse = reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], None, None]:
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None


def gradient_reverse(x: Tensor, lambd: float = 1.0, reverse: bool = True) -> Tensor:
    return GradReverse.apply(x, lambd, reverse)


class DomainDiscriminator(nn.Module):
    def __init__(
        self,
        input_nodes: int,
        num_domains: int = 3,
        hidden_nodes: List[int] = None,
        grl: int = True,
        reverse: bool = True,
    ) -> None:
        super(DomainDiscriminator, self).__init__()

        self.in_features = None

        self.input_nodes: int = input_nodes
        self.output_nodes: int = num_domains
        self.hidden_nodes: List[int] = [self.input_nodes] + (
            hidden_nodes if hidden_nodes else [10] * 3
        )
        self.hidden_layers: int = len(hidden_nodes)

        self.grl = grl
        self.reverse = reverse
        self.lambd: float = 0.0

        self.fc_modules: nn.ModuleList = nn.ModuleList()

        for i in range(self.hidden_layers):
            self.fc_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_nodes[i], self.hidden_nodes[i + 1]),
                    nn.BatchNorm1d(self.hidden_nodes[i + 1]),
                    nn.ELU(),
                    nn.Dropout(0.25),
                )
            )

        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_nodes[-1], self.output_nodes)
        )

    def set_lambd(self, lambd):
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        if self.grl:
            x: Tensor = gradient_reverse(x, self.lambd, self.reverse)

        for layers in self.fc_modules:
            x: Tensor = layers(x)

        domain_output = self.output_layer(x)
        return domain_output
