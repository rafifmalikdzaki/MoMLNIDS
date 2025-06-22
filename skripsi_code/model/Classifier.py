import torch.nn as nn
import torch
from torch import Tensor
from typing import List
from torchsummary import summary


class ClassifierANN(nn.Module):
    def __init__(
        self,
        input_nodes: int,
        num_class: int = 3,
        hidden_nodes: List[int] = None,
    ) -> None:
        super(ClassifierANN, self).__init__()

        self.input_nodes: int = input_nodes
        self.output_nodes: int = num_class
        self.hidden_nodes: List[int] = [self.input_nodes] + (
            hidden_nodes if hidden_nodes else [10] * 3
        )

        self.fc_modules: nn.ModuleList = nn.ModuleList()

        if hidden_nodes:
            self.hidden_layers: int = len(hidden_nodes)

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

    def forward(self, x: Tensor) -> Tensor:
        for layers in self.fc_modules:
            x: Tensor = layers(x)

        domain_output = self.output_layer(x)
        return domain_output


if __name__ == "__main__":
    model = ClassifierANN(input_nodes=20, num_class=3, hidden_nodes=[64, 32, 16]).to(
        "cuda"
    )
    print(model)
    x = torch.randn(5, 20).to("cuda")  # Batch of 5, input size of 20
    print(model(x))
    summary(model, (20,))
