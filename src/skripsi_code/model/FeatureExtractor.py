import torch.nn as nn
import torch
from torch import Tensor
from torchsummary import summary
from typing import List


class DGFeatExt(nn.Module):
    def __init__(self, input_nodes: int, hidden_nodes: List[int] = None) -> None:
        super(DGFeatExt, self).__init__()
        self.input_nodes: int = input_nodes
        self.hidden_nodes: List[int] = [self.input_nodes] + (
            hidden_nodes if hidden_nodes else [10] * 3
        )
        self.hidden_layers: int = len(hidden_nodes)

        self.fc_modules: nn.ModuleList = nn.ModuleList()
        self.norm = nn.BatchNorm1d(self.input_nodes)

        """
        Singla, A., Bertino, E. and Verma, D., 2020. Preparing Network Intrusion Detection Deep Learning Models with Minimal Data Using Adversarial Domain Adaptation. In: Proceedings of the 15th ACM Asia Conference on Computer and Communications Security. [online] ASIA CCS ’20: The 15th ACM Asia Conference on Computer and Communications Security. Taipei Taiwan: ACM. pp.127–140. https://doi.org/10.1145/3320269.3384718.
        """

        for i in range(self.hidden_layers):
            self.fc_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_nodes[i], self.hidden_nodes[i + 1]),
                    nn.BatchNorm1d(self.hidden_nodes[i + 1]),
                    nn.ELU(),
                    nn.Dropout(0.3),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        for layers in self.fc_modules:
            x: Tensor = layers(x)

        return x

    def feature_extraction(
        self, x: Tensor, extracted_layers: List[int] = None
    ) -> List[Tensor]:
        if not extracted_layers:
            extracted_layers: List[int] = [1, 2]

        result: List[Tensor] = []

        for layers in range(self.hidden_layers):
            x: Tensor = self.fc_modules[layers](x)
            if layers in extracted_layers:
                result.append(x)

        return result


if __name__ == "__main__":
    from torchinfo import summary

    model = DGFeatExt(input_nodes=20, hidden_nodes=[64, 32, 16]).to("cuda")
    print(model)
    x = torch.randn(5, 20).to("cuda")  # Batch of 5, input size of 20
    print(model(x))
    summary(model, input_size=(5, 20))
    # print(model.feat_ext(x))
