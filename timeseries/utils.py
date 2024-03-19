import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


def plot_pred_target(predictions, mean, std, idx):
    preds, targets = predictions[idx]

    # denormalize
    preds = np.array(preds) * std["next_consumption"] + mean["next_consumption"]
    targets = (
        np.array(targets.cpu()) * std["next_consumption"] + mean["next_consumption"]
    )

    plt.figure(figsize=(8, 4), dpi=150)
    plt.grid()
    plt.xticks(range(len(targets)))
    plt.plot(range(len(preds)), preds, label="Predictions")
    plt.plot(range(len(targets)), targets, label="Targets")
    plt.scatter(range(1, len(preds)), preds[1:], color="red")
    plt.scatter(range(1, len(targets)), targets[1:], color="green")
    plt.legend()
    plt.show()


def plot_mse(predictions, mean, std):
    squared_errors = []

    for p, t in predictions:
        p = np.array(p) * std["next_consumption"] + mean["next_consumption"]
        t = t.cpu().numpy() * std["next_consumption"] + mean["next_consumption"]

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
