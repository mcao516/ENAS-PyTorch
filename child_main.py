import torch

from src.config import Config
from src.data_utils import read_cifar10, get_mean_and_std, DatasetBuilder
from src.cifar10.child_model import ChildModel


def main():
    """Start training process.
    """
    config = Config("train")
    
    # read cifar10 data
    config.logger.info("Reading cifar10 data...")
    images, labels = read_cifar10(config.data_dir, valid_num=5000)
    # images['train'].shape == (45000, 3, 32, 32)

    # find cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.logger.info("Device used: {}".format(device))

    # build train, valid and test set
    mean, std = get_mean_and_std(images['train'])
    builder = DatasetBuilder(images['train'].shape[-1], mean=mean, std=std)
    
    config.logger.info("Building dataset...")
    train_set = builder.build_dataset(images['train'], labels['train'], 'train', config.batch_size, 
                             shuffle=True, num_workers=4, device=device)
    valid_set = builder.build_dataset(images['valid'], labels['valid'], 'dev', config.batch_size * 5, 
                             num_workers=4, device=device)
    test_set  = builder.build_dataset(images['test'], labels['test'], 'test', config.batch_size * 5, 
                             num_workers=4, device=device)

    config.logger.info("- Training set size: {}".format(len(train_set.dataset)))
    config.logger.info("- Dev set size: {}".format(len(valid_set.dataset)))
    config.logger.info("- Test set size: {}".format(len(test_set.dataset)))

    # build model
    config.logger.info("Building the model...")
    model = ChildModel(config, device=device)
    arcs = ([0, 0, 2, 4, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3, 5, 2, 3, 1], 
            [1, 0, 0, 3, 0, 1, 3, 4, 1, 3, 3, 3, 2, 4, 3, 3, 3, 2, 3, 3])
    model.set_arcs(arcs)
    
    # print architecture
    config.logger.info("- Model architecture:")
    config.logger.info("- Normal cell:    {}".format(arcs[0]))
    config.logger.info("- Reduction cell: {}".format(arcs[1]))

    # training
    config.logger.info("Start training...")
    model.fit(train_set, test_set, saving=True)


if __name__ == "__main__":
    main()