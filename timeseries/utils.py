import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

# plotters


def plot_loss_epoch(losses_show, val_losses):
    plt.figure(figsize=(6, 4), dpi=150)

    plt.grid()
    plt.ylim(0, 0.03)

    plt.plot(range(len(losses_show)), losses_show, label="Train Loss")
    plt.plot(
        range(
            len(losses_show) // len(val_losses),
            len(losses_show) + 1,
            len(losses_show) // len(val_losses),
        ),
        val_losses,
        label="Val Loss",
    )
    plt.show()


def plot_pred_target(predictions, idx):
    preds, targets = predictions[idx]

    plt.figure(figsize=(8, 4), dpi=150)
    plt.grid()
    plt.xticks(range(len(targets)))
    plt.plot(range(len(preds)), preds, label="Predictions")
    plt.plot(range(len(targets)), targets, label="Targets")
    plt.scatter(range(1, len(preds)), preds[1:], color="red")
    plt.scatter(range(1, len(targets)), targets[1:], color="green")
    plt.legend()
    plt.show()


def plot_mse(predictions):
    squared_errors = []

    for p, t in predictions:
        squared_error = ((t - p) ** 2).tolist()
        squared_errors.append(squared_error)

    squared_errors_df = pd.DataFrame(np.array(squared_errors[:-1]))

    squared_errors_mean = squared_errors_df.mean().values
    squared_errors_std = squared_errors_df.std().values
    x = range(len(squared_errors_mean))

    plt.figure(figsize=(8, 4), dpi=150)
    plt.title("Mean Squared Error")
    plt.grid()
    plt.plot(x, squared_errors_mean)
    plt.fill_between(
        x,
        squared_errors_mean - squared_errors_std,
        squared_errors_mean + squared_errors_std,
        alpha=0.2,
    )
    plt.show()


# testers


def step_by_step(model, device, test_loader, std, mean):
    model.eval()

    with torch.no_grad():
        predictions = []

        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            preds = []
            i = 0
            pred = 0
            for f, t in zip(X, y):
                if i > 0:
                    f[1] = pred
                    pred = model(f)
                    preds.append(pred.item())
                elif i == 0:
                    pred = model(f)
                    preds.append(pred.item())
                i += 1

            preds = np.array(preds) * std["next_consumption"] + mean["next_consumption"]
            y = np.array(y.cpu()) * std["next_consumption"] + mean["next_consumption"]
            predictions.append([preds, y])

    return predictions
