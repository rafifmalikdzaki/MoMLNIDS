from src.skripsi_code.model.MoMLNIDS import MoMLDNIDS
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
import torch
from torch import nn, optim
from copy import deepcopy


def split_domain(domains, split_idx, print_domain=True):
    source_domain = deepcopy(domains)
    target_domain = [source_domain.pop(split_idx)]
    if print_domain:
        print("Source domain: ", end="")
        for domain in source_domain:
            print(domain, end=", ")
        print("Target domain: ", end="")
        for domain in target_domain:
            print(domain)
    return source_domain, target_domain


def get_model_learning_rate(
    model, extractor_weight=1.0, classifier_weight=1.0, discriminator_weight=1.0
):
    return [
        (model.FeatureExtractorLayer, 1.0 * extractor_weight),
        (model.DomainClassifier, 1.0 * discriminator_weight),
        (model.LabelClassifier, 1.0 * classifier_weight),
    ]


def get_optimizer(model, init_lr, weight_decay=0.01, amsgrad=True):
    if not amsgrad:
        return optim.SGD(
            model.parameters(),
            lr=init_lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    else:
        return optim.AdamW(
            model.parameters(), lr=init_lr, weight_decay=weight_decay, amsgrad=amsgrad
        )


def get_learning_rate_scheduler(optimizer, t_max=25):
    return CosineAnnealingLR(optimizer, t_max, eta_min=1e-5)


if __name__ == "__main__":
    x = torch.randn(1000, 43).to("cuda")  # Batch of 5, input size of 20
    network = MoMLDNIDS(
        input_nodes=x.size(dim=1), hidden_nodes=[64, 32, 16], num_domains=3, num_class=2
    ).to("cuda")
    print(network)
    print(network(x))
    summary(network, (x.size(dim=1),))
