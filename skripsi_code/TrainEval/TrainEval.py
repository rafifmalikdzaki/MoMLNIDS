import wandb
from torch import nn
from skripsi_code.utils.loss import EntropyLoss, MaximumSquareLoss
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
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

    all_labels = []
    all_predictions = []

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

        all_labels.extend(class_label.data.numpy(force=True))
        all_predictions.extend(pred_class.numpy(force=True))

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

    f1 = f1_score(all_labels, all_predictions, average="weighted")
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")

    log = (
        f"Train: Epoch: {epoch}/{num_epoch} | Alpha: {alpha:.4f} |\n"
        f"LClass: {epoch_loss_class:.4f} | Acc Class: {epoch_acc_class:.4f} |\n"
        f"LDomain: {epoch_loss_domain:.4f} | Acc Domain: {epoch_acc_domain:.4f} |\n"
        f"Loss Entropy: {epoch_loss_entropy:.4f} |\n"
        f"F1 Score: {f1:.4f} Precision: {precision:.4f} Recall: {recall:.4f} \n"
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
        "Train/F1_Score": f1,
        "Train/Precision": precision,
        "Train/Recall": recall,
        "Train/Alpha": alpha
    }, step=epoch)
    
    return model, optimizers



def eval(model, eval_data, device, epoch, filename):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    data_size = 0

    all_labels = []
    all_predictions = []

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

            all_labels.extend(labels.data.numpy(force=True))
            all_predictions.extend(prediction.numpy(force=True))

    epoch_loss = running_loss / data_size
    epoch_acc = running_corrects / data_size

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calculate F1 score, precision, and recall
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")

    # Calculate ROC AUC
    # Convert labels to one-hot for roc_auc_score
    num_classes = output.shape[1]
    all_labels_one_hot = np.eye(num_classes)[all_labels]
    all_predictions_proba = torch.nn.functional.softmax(output, dim=1).cpu().numpy()

    roc_auc = roc_auc_score(all_labels_one_hot, all_predictions_proba, average="weighted", multi_class="ovr")

    log = (
        f"Eval: Epoch: {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
        f"F1 Score: {f1:.4f} Precision: {precision:.4f} Recall: {recall:.4f} "
        f"ROC AUC: {roc_auc:.4f}"
    )
    print(log)

    with open(filename, "a") as f:
        f.write(log + "\n")

    # Log metrics to wandb
    wandb.log({
        "Eval/Loss": epoch_loss,
        "Eval/Accuracy": epoch_acc,
        "Eval/F1_Score": f1,
        "Eval/Precision": precision,
        "Eval/Recall": recall,
        "Eval/ROC_AUC": roc_auc
    }, step=epoch)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels_one_hot[:, i], all_predictions_proba[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_score(all_labels_one_hot[:, i], all_predictions_proba[:, i]):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    wandb.log({"Eval/ROC_Curve": wandb.Image(plt)}, step=epoch)
    plt.close()

    # Plot PR Curve
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels_one_hot[:, i], all_predictions_proba[:, i])
        plt.plot(recall_curve, precision_curve, label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    wandb.log({"Eval/PR_Curve": wandb.Image(plt)}, step=epoch)
    plt.close()

    # Log Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    wandb.log({"Eval/Confusion_Matrix": wandb.plot.confusion_matrix(preds=all_predictions, y_true=all_labels, class_names=[str(i) for i in range(num_classes)])}, step=epoch)

    return epoch_acc
