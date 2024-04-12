import torch
from classifier import Classifier
from config import DEVICE
from stacked_mnist import StackedMNISTData
from standard import (
    Autoencoder,
    Decoder,
    Encoder,
    get_image_reconstructed,
    plot_image_reconstructed,
    train,
)
from torch.utils.data import DataLoader


def ae_basic():
    classifier = Classifier().to(DEVICE)
    standard = Autoencoder(encoder=Encoder(), decoder=Decoder()).to(DEVICE)

    classifier.load_state_dict(torch.load("models/verification_classifier.pth"))

    try:
        standard.load_state_dict(torch.load("models/standard_ae.pth"))
        print("Pre-trained model found, using it...")
    except FileNotFoundError:
        print("No model found, training a new one...")

        train_dataset = StackedMNISTData(root="data", train=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(standard.parameters(), lr=1e-3)

        epochs = 10

        for epoch in range(epochs):
            loss = train(standard, train_loader, loss_fn, optimizer)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {sum(loss) / len(loss)}")

        torch.save(standard.state_dict(), "models/standard_ae.pth")

    test_dataset = StackedMNISTData(root="data", train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=True)

    total = 0
    correct = 0

    standard.eval()
    classifier.eval()
    with torch.no_grad():
        for image, _ in test_loader:
            image = image.to(DEVICE)

            reconstructed = standard(image)

            label = classifier(image)
            pred = classifier(reconstructed)

            _, pred = torch.max(pred, 1)
            _, label = torch.max(label, 1)

            total += label.size(0)
            correct += (pred == label).sum().item()

    print("Accuracy:", correct / total)

    plot_image_reconstructed(get_image_reconstructed(standard, test_loader))


if __name__ == "__main__":
    print("Project 3: Autoencoders")
    ae_basic()
