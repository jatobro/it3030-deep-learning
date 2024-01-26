from network import Network


def main():
    network = Network()
    pred = network.forward_pass()
    print(pred)


if __name__ == "__main__":
    main()
