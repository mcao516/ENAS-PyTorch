import torch

from src.config import Config
from src.cifar10.controller_model import ControllerModel


def get_acc(arc):
    return sum([i == 2 for i in arc]) / len(arc)


def main():
    """Start training process.
    """
    config = Config("train")
    
    # find cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.logger.info("Device used: {}".format(device))
    
    # build model
    config.logger.info("Building the model...")
    model = ControllerModel(config, device=device)

    # training
    config.logger.info("Start training...")

    num_epochs = 300
    for i in range(num_epochs):
        print('- Epoch: {}/{}'.format(i+1, num_epochs))
        (arc_1, arc_2), (logits_1, logits_2) = model.sample()
        acc = get_acc(arc_1 + arc_2)
        print("- accuracy: {}".format(acc))
        
        arc_1 = torch.tensor(arc_1, dtype=torch.long).to(device)
        arc_2 = torch.tensor(arc_2, dtype=torch.long).to(device)
        model.fit(acc, [(arc_1, logits_1), (arc_2, logits_2)])


if __name__ == "__main__":
    main()