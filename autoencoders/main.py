import matplotlib.pyplot as plt
import torch
from classifier import Classifier
from config import DEVICE
from stacked_mnist import DataMode, StackedMNISTData
from standard import (
    Autoencoder,
    Decoder,
    Encoder,
    get_image_reconstructed,
    test,
    train,
)
from torch.utils.data import DataLoader
from variational import VariationalAE, get_variational_image_reconstructed


def standard_verification(autoencoder, loader):
    classifier = Classifier().to(DEVICE)
    classifier.load_state_dict(torch.load("models/verification_classifier.pth"))

    total = 0
    correct = 0

    classifier.eval()
    autoencoder.eval()
    with torch.no_grad():
        for image, _ in loader:
            image = image.to(DEVICE)

            reconstructed = autoencoder(image)

            label = classifier(image)
            pred = classifier(reconstructed)

            _, label = torch.max(label, 1)
            _, pred = torch.max(pred, 1)

            total += label.size(0)
            correct += (pred == label).sum().item()

    return correct / total


def variational_verification(autoencoder, loader):
    classifier = Classifier().to(DEVICE)
    classifier.load_state_dict(torch.load("models/verification_classifier.pth"))

    total = 0
    correct = 0

    classifier.eval()
    autoencoder.eval()
    with torch.no_grad():
        for image, _ in loader:
            image = image.to(DEVICE)

            reconstructed, _, _ = autoencoder(image)

            label = classifier(image)
            pred = classifier(reconstructed)

            _, label = torch.max(label, 1)
            _, pred = torch.max(pred, 1)

            total += label.size(0)
            correct += (pred == label).sum().item()

    return correct / total


def plot_image_reconstructed(images_reconstructed):
    for image, reconstructed in images_reconstructed:
        image = image.view(-1, 28, 28)
        reconstructed = reconstructed.view(-1, 28, 28)

        plt.figure()
        plt.imshow(image[0].cpu().numpy())
        plt.title("Original")
        plt.show()

        plt.figure()
        plt.imshow(reconstructed[0].cpu().numpy())
        plt.title("Reconstructed")
        plt.show()


def ae_basic():
    standard = Autoencoder(encoder=Encoder(), decoder=Decoder()).to(DEVICE)

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

    print(f"Accuracy: {standard_verification(standard, test_loader)}")
    plot_image_reconstructed(get_image_reconstructed(standard, test_loader))


def ae_anom():
    k = 10

    standard = Autoencoder(encoder=Encoder(), decoder=Decoder()).to(DEVICE)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(standard.parameters(), lr=1e-3)

    try:
        standard.load_state_dict(torch.load("models/anom_ae.pth"))
        print("Pre-trained model found, using it...")
    except FileNotFoundError:
        train_dataset = StackedMNISTData(root="data", mode=DataMode.MISSING, train=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)

        epochs = 10

        losses = []
        for epoch in range(epochs):
            loss = train(standard, train_loader, loss_fn, optimizer)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {sum(loss) / len(loss)}")
            losses.extend(loss)

        torch.save(standard.state_dict(), "models/anom_ae.pth")

        plt.style.use("fivethirtyeight")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        plt.plot(losses[-1000:])

        plt.show()

    test_dataset = StackedMNISTData(root="data", mode=DataMode.MISSING, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=True)

    print(f"Verification Accuracy: {standard_verification(standard, test_loader)}")

    reconstruction_loss = test(standard, test_loader, loss_fn)

    print(
        f"Average Reconstruction Loss: {sum(reconstruction_loss) / len(reconstruction_loss)}"
    )

    plt.style.use("fivethirtyeight")
    plt.xlabel("Batch")
    plt.ylabel("Reconstruction Loss")

    plt.plot(reconstruction_loss)
    plt.show()


def vae_basic():
    variational = VariationalAE().to(DEVICE)
    variational.load_state_dict(torch.load("models/variational_ae.pth"))

    test_dataset = StackedMNISTData(root="data", train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=True)

    print(
        f"Verification Accuracy: {variational_verification(variational, test_loader)}"
    )
    plot_image_reconstructed(
        get_variational_image_reconstructed(variational, test_loader)
    )


if __name__ == "__main__":
    print("Project 3: Autoencoders")
    # ae_basic()
    # ae_anom()
    vae_basic()
