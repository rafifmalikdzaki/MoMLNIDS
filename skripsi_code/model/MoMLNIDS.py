import torch
import torch.nn as nn
from wandb.integration.sklearn.plot.classifier import classifier

from skripsi_code.model.FeatureExtractor import DGFeatExt
from skripsi_code.model.Discriminator import DomainDiscriminator
from skripsi_code.model.Classifier import ClassifierANN
from torchsummary import summary


class MoMLDNIDS(nn.Module):
    def __init__(
        self,
        input_nodes,
        hidden_nodes,
        classifier_nodes,
        num_domains,
        num_class,
        single_layer=True,
    ):
        super().__init__()
        self.FeatureExtractorLayer = DGFeatExt(input_nodes, hidden_nodes)
        self.DomainClassifier = DomainDiscriminator(
            hidden_nodes[-1], num_domains, classifier_nodes
        )

        self.LabelClassifier = ClassifierANN(
            hidden_nodes[-1],
            num_class,
            classifier_nodes.clear() if single_layer else classifier_nodes,
        )

    def forward(self, x):
        invariant_features = self.FeatureExtractorLayer(x)
        output_class = self.LabelClassifier(invariant_features)
        output_domain = self.DomainClassifier(invariant_features)
        return output_class, output_domain


if __name__ == "__main__":
    x = torch.randn(1000, 43).to("cuda")  # Batch of 5, input size of 20
    model = MoMLDNIDS(
        input_nodes=x.size(dim=1),
        hidden_nodes=[64, 32, 16, 10],
        classifier_nodes=[64, 32, 16],
        num_domains=3,
        num_class=2,
    ).to("cuda")
    print(model)
    print(model(x))
    summary(model, (x.size(dim=1),))
