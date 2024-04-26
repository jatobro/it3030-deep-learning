import numpy as np
import torch
from models.classifier import Classifier
from stacked_mnist import DataMode, StackedMNISTData
from torch.utils.data import DataLoader, TensorDataset
from utils import to_one_hot


def trainer():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=9)

    train_data = gen.get_full_data_set()
    test_data = gen.get_full_data_set(training=False)

    X_train, y_train = train_data
    X_test, y_test = test_data

    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)

    # img = X_train[np.random.randint(0, X_train.shape[0])]

    X_train = (
        torch.from_numpy(X_train.astype(np.float32)).to(device).permute(0, 3, 1, 2)
    )
    y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    X_test = torch.from_numpy(X_test.astype(np.float32)).to(device).permute(0, 3, 1, 2)
    y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = Classifier(image_depth=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = torch.nn.MSELoss()

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, loss: {total_loss}")

    torch.save(model.state_dict(), "trained_models/classifier.pth")

    model.eval()
    with torch.no_grad():
        correct_preds = 0
        test_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            test_loss += loss.item()

            pred = pred.argmax(dim=1, keepdim=True)
            truth = y.argmax(dim=1, keepdim=True)
            correct_preds += pred.eq(truth.view_as(pred)).sum().item()

        test_accuracy = correct_preds / len(test_loader.dataset)

    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {100*test_accuracy}%")


if __name__ == "__main__":
    trainer()
