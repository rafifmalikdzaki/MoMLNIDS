import wandb
from torch import nn
from skripsi_code.utils.loss import EntropyLoss, MaximumSquareLoss
import torch
from torchmetrics import F1Score, Precision, Recall, ROC, AUROC, AveragePrecision, MatthewsCorrCoef, ConfusionMatrix, Specificity
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler


def check_gradient_norm(model, threshold=1e4):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    if total_norm > threshold:
        print(f"Warning: Gradient norm {total_norm} exceeds threshold {threshold}")
    return total_norm


def check_nan_inf_in_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN detected in gradient of parameter: {name}")
            if torch.isinf(param.grad).any():
                print(f"Inf detected in gradient of parameter: {name}")


def train(
    model,
    train_data,
    optimizers,
    device,
    epoch,
    num_epoch,
    filename,
    disc_weight=None,
    class_weight=None,
    label_smooth=None,
    entropy_weight=1.0,
    grl_weight=1.0,
):
    scaler = GradScaler()

    # Reloading buffer
    # train_data.dataset.dataset.reload_buffer()

    # Loss Functions
    class_criterion = nn.CrossEntropyLoss(
        weight=class_weight, label_smoothing=label_smooth
    )
    domain_criterion = nn.CrossEntropyLoss(
        weight=disc_weight, label_smoothing=label_smooth
    )
    entropy_criterion = EntropyLoss()

    P = epoch / num_epoch
    alpha = (2.0 / (1.0 + np.exp(-10 * P)) - 1) * grl_weight
    beta = (2.0 / (1.0 + np.exp(-10 * P)) - 1) * entropy_weight

    model.DomainClassifier.set_lambd(alpha)
    model.train()

    running_loss_class = 0.0
    running_correct_class = 0.0

    running_loss_domain = 0.0
    running_correct_domain = 0.0

    running_loss_entropy = 0.0

    data_size = 0

    num_classes = model.ClassClassifier.output_dim # Assuming this is how to get num_classes
    
    # Initialize TorchMetrics for training
    f1_metric_train = F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    precision_metric_train = Precision(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    recall_metric_train = Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    auroc_metric_train = AUROC(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    avg_precision_metric_train = AveragePrecision(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    mcc_metric_train = MatthewsCorrCoef(task="multiclass", num_classes=num_classes).to(device)
    sensitivity_metric_train = Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device) # Sensitivity is Recall
    specificity_metric_train = Specificity(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    conf_matrix_train = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    all_labels_tensor = torch.tensor([], dtype=torch.long, device=device)
    all_predictions_tensor = torch.tensor([], dtype=torch.long, device=device)
    all_predictions_proba_tensor = torch.tensor([], dtype=torch.float, device=device)

    for data, class_label, domain_label in train_data:
        data, class_label, domain_label = (
            data.double().to(device),
            class_label.squeeze().long().to(device),
            domain_label.squeeze().long().to(device),
        )

        # Numerical instability is possible
        # data = torch.clamp(data, min=-1e4, max=1e4)  # Example clamping

        data_size += data.size(0)

        if torch.isnan(data).any() or torch.isinf(data).any():
            print("Input contains NaN or Inf values.")

        for optimizer in optimizers:
            optimizer.zero_grad()

        with autocast():
            output_class, output_domain = model(data)

            loss_class = class_criterion(output_class, class_label)
            loss_domain = domain_criterion(output_domain, domain_label)
            loss_entropy = entropy_criterion(output_class)

            total_loss = loss_class + loss_domain + loss_entropy * beta

        pred_class = output_class.squeeze().argmax(dim=1)
        pred_domain = output_domain.squeeze().argmax(dim=1)

        # print(f"{loss_class.item():.4f}, {loss_domain.item():.4f}, {loss_entropy.item():.4f}")

        scaler.scale(total_loss).backward()

        check_gradient_norm(model)
        check_nan_inf_in_gradients(model)

        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.update()

        all_labels_tensor = torch.cat((all_labels_tensor, class_label))
        all_predictions_tensor = torch.cat((all_predictions_tensor, pred_class))
        all_predictions_proba_tensor = torch.cat((all_predictions_proba_tensor, torch.nn.functional.softmax(output_class, dim=1)))

        f1_metric_train.update(pred_class, class_label)
        precision_metric_train.update(pred_class, class_label)
        recall_metric_train.update(pred_class, class_label)
        auroc_metric_train.update(output_class, class_label)
        avg_precision_metric_train.update(output_class, class_label)
        mcc_metric_train.update(pred_class, class_label)
        sensitivity_metric_train.update(pred_class, class_label)
        specificity_metric_train.update(pred_class, class_label)
        conf_matrix_train.update(pred_class, class_label)

        # Classifier Loss
        running_loss_class += loss_class.item() * data.size(0)
        running_correct_class += torch.sum(pred_class == class_label.data)

        # Domain Loss
        running_loss_domain += loss_domain.item() * data.size(0)
        running_correct_domain += torch.sum(pred_domain == domain_label.data)

        # Entropy Loss
        running_loss_entropy += loss_entropy.item() * data.size(0)

    # Class Loss on Epoch
    epoch_loss_class = running_loss_class / data_size
    epoch_acc_class = running_correct_class.double() / data_size

    # Domain Loss on Epoch
    epoch_loss_domain = running_loss_domain / data_size
    epoch_acc_domain = running_correct_domain.double() / data_size

    # Entropy Loss on Epoch
    epoch_loss_entropy = running_loss_entropy / data_size

    # Calculate metrics using TorchMetrics
    f1_train = f1_metric_train.compute()
    precision_train = precision_metric_train.compute()
    recall_train = recall_metric_train.compute()
    auroc_train = auroc_metric_train.compute()
    avg_precision_train = avg_precision_metric_train.compute()
    mcc_train = mcc_metric_train.compute()
    sensitivity_train = sensitivity_metric_train.compute()
    specificity_train = specificity_metric_train.compute()

    log = (
        f"Train: Epoch: {epoch}/{num_epoch} | Alpha: {alpha:.4f} |\n"
        f"LClass: {epoch_loss_class:.4f} | Acc Class: {epoch_acc_class:.4f} |\n"
        f"LDomain: {epoch_loss_domain:.4f} | Acc Domain: {epoch_acc_domain:.4f} |\n"
        f"Loss Entropy: {epoch_loss_entropy:.4f} |\n"
        f"F1 Score: {f1_train:.4f} Precision: {precision_train:.4f} Recall: {recall_train:.4f} \n"
        f"AUROC: {auroc_train:.4f} Avg Precision: {avg_precision_train:.4f} MCC: {mcc_train:.4f} \n"
        f"Sensitivity: {sensitivity_train:.4f} Specificity: {specificity_train:.4f}"
    )

    print(log)

    with open(filename, "a") as f:
        f.write(log + "\n")
    
    wandb.log({
        "Train/Loss_Class": epoch_loss_class,
        "Train/Acc_Class": epoch_acc_class,
        "Train/Loss_Domain": epoch_loss_domain,
        "Train/Acc_Domain": epoch_acc_domain,
        "Train/Loss_Entropy": epoch_loss_entropy,
        "Train/F1_Score": f1_train,
        "Train/Precision": precision_train,
        "Train/Recall": recall_train,
        "Train/AUROC": auroc_train,
        "Train/Average_Precision": avg_precision_train,
        "Train/MCC": mcc_train,
        "Train/Sensitivity": sensitivity_train,
        "Train/Specificity": specificity_train,
        "Train/Alpha": alpha
    }, step=epoch)
    
    return model, optimizers, len(train_data)



def eval(model, eval_data, device, epoch, filename):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    data_size = 0

    all_labels = torch.tensor([], dtype=torch.long, device=device)
    all_predictions = torch.tensor([], dtype=torch.long, device=device)
    all_predictions_proba = torch.tensor([], dtype=torch.float, device=device)

    num_classes = output.shape[1]

    # Initialize TorchMetrics for evaluation
    f1_metric_eval = F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    precision_metric_eval = Precision(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    recall_metric_eval = Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    auroc_metric_eval = AUROC(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    avg_precision_metric_eval = AveragePrecision(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    mcc_metric_eval = MatthewsCorrCoef(task="multiclass", num_classes=num_classes).to(device)
    sensitivity_metric_eval = Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device) # Sensitivity is Recall
    specificity_metric_eval = Specificity(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    conf_matrix_eval = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    with torch.inference_mode():
        for data, labels, *_ in eval_data:
            data, labels = (
                data.squeeze().double().to(device),
                labels.squeeze().long().to(device),
            )

            # get class labels
            output = model(data)[0].squeeze()
            prediction = output.argmax(dim=1)

            loss = criterion(output, labels)

            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(prediction == labels.data).item()
            data_size += data.size(0)

            all_labels.extend(labels.data)
            all_predictions.extend(prediction)
            
            f1_metric_eval.update(prediction, labels)
            precision_metric_eval.update(prediction, labels)
            recall_metric_eval.update(prediction, labels)
            auroc_metric_eval.update(output, labels)
            avg_precision_metric_eval.update(output, labels)
            mcc_metric_eval.update(prediction, labels)
            sensitivity_metric_eval.update(prediction, labels)
            specificity_metric_eval.update(prediction, labels)
            conf_matrix_eval.update(prediction, labels)

    epoch_loss = running_loss / data_size
    epoch_acc = running_corrects / data_size

    # Calculate metrics using TorchMetrics
    f1_eval = f1_metric_eval.compute()
    precision_eval = precision_metric_eval.compute()
    recall_eval = recall_metric_eval.compute()
    auroc_eval = auroc_metric_eval.compute()
    avg_precision_eval = avg_precision_metric_eval.compute()
    mcc_eval = mcc_metric_eval.compute()
    sensitivity_eval = sensitivity_metric_eval.compute()
    specificity_eval = specificity_metric_eval.compute()

    log = (
        f"Eval: Epoch: {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} \n"
        f"F1 Score: {f1_eval:.4f} Precision: {precision_eval:.4f} Recall: {recall_eval:.4f} \n"
        f"AUROC: {auroc_eval:.4f} Avg Precision: {avg_precision_eval:.4f} MCC: {mcc_eval:.4f} \n"
        f"Sensitivity: {sensitivity_eval:.4f} Specificity: {specificity_eval:.4f}"
    )
    print(log)

    with open(filename, "a") as f:
        f.write(log + "\n")

    # Log metrics to wandb
    wandb.log({
        "Val/Loss": epoch_loss,
        "Val/Accuracy": epoch_acc,
        "Val/F1_Score": f1_eval,
        "Val/Precision": precision_eval,
        "Val/Recall": recall_eval,
        "Val/AUROC": auroc_eval,
        "Val/Average_Precision": avg_precision_eval,
        "Val/MCC": mcc_eval,
        "Val/Sensitivity": sensitivity_eval,
        "Val/Specificity": specificity_eval
    }, step=epoch)

    # Log ROC Curve
    wandb.log({"Val/ROC_Curve": wandb.plot.roc_curve(y_true=torch.cat(all_labels).cpu(), y_probas=torch.cat(all_predictions_proba).cpu(), labels=[str(i) for i in range(num_classes)])}, step=epoch)

    # Log PR Curve
    wandb.log({"Val/PR_Curve": wandb.plot.pr_curve(y_true=torch.cat(all_labels).cpu(), y_probas=torch.cat(all_predictions_proba).cpu(), labels=[str(i) for i in range(num_classes)])}, step=epoch)

    # Log Confusion Matrix
    wandb.log({"Val/Confusion_Matrix": wandb.plot.confusion_matrix(preds=torch.cat(all_predictions).cpu(), y_true=torch.cat(all_labels).cpu(), class_names=[str(i) for i in range(num_classes)])}, step=epoch)}

    return epoch_acc
