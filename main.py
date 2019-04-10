# -*- coding: utf-8 -*-

"""This python file is for ENAS model training.
   
   Author: Meng Cao
"""

import torch

from src.utils import Progbar
from src.config import Config
from src.cifar10.child_model import ChildModel
from src.cifar10.controller_model import ControllerModel
from src.data_utils import read_cifar10, get_mean_and_std, DatasetBuilder


def read_dataset(config, device):
    """Read the data in config file.
    """
    # read cifar10 data
    config.logger.info("Reading cifar10 data...")
    images, labels = read_cifar10(config.data_dir, valid_num=5000)
    # images['train'].shape == (45000, 3, 32, 32)

    # build train, valid and test set
    mean, std = get_mean_and_std(images['train'])
    builder = DatasetBuilder(images['train'].shape[-1], mean=mean, std=std)
    
    config.logger.info("Building dataset...")
    train_set = builder.build_dataset(images['train'], labels['train'], 'train', config.batch_size, 
                             shuffle=True, num_workers=4, device=device)
    valid_set = builder.build_dataset(images['valid'], labels['valid'], 'dev', config.batch_size * 3, 
                             num_workers=4, device=device)
    test_set  = builder.build_dataset(images['test'], labels['test'], 'test', config.batch_size * 3, 
                             num_workers=4, device=device)

    config.logger.info("- Training set size: {}".format(len(train_set.dataset)))
    config.logger.info("- Dev set size: {}".format(len(valid_set.dataset)))
    config.logger.info("- Test set size: {}".format(len(test_set.dataset)))

    return train_set, valid_set, test_set


def main():
    """Start training process.
    """
    config = Config("train")

    # find cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.logger.info("Device used: {}".format(device))
    
    # read dataset
    train_set, valid_set, test_set = read_dataset(config, device)
    
    # build model
    config.logger.info("Building the model...")
    model = ChildModel(config, device=device)
    controller = ControllerModel(config, device=device)

    best_acc = 0.
    for epoch in range(config.num_epochs):
        config.logger.info('Epoch {}/{}'.format(epoch + 1, config.num_epochs))

        # sample from controller
        arcs, _ = controller.sample(False)
        model.set_arcs(arcs)

        # print sample
        config.logger.info("- Model architecture:")
        config.logger.info("- Normal cell:    {}".format(arcs[0]))
        config.logger.info("- Reduction cell: {}".format(arcs[1]))

        # train child model
        model.train_epoch(train_set, epoch)
        
        # evaluate child model
        valid_loss, valid_acc = model.evaluate(valid_set)
        config.logger.info("- Evaluation:")
        config.logger.info("- loss: {}".format(valid_loss))
        config.logger.info("- acc: {}".format(valid_acc))

        # save models
        if valid_acc >= best_acc:
            best_acc = valid_acc
            config.logger.info("- New best scores!")
            model.save_model()
            controller.save_model()
        
        # train controller
        if epoch % config.train_contro_every == 0 and epoch > 50:
            prog = Progbar(target=config.contro_train_epochs) # progress bar for visualization
            config.logger.info('Controller training ({}):'.format(config.contro_num_epochs))

            avg_valid_acc = 0.
            for i in range(config.contro_train_epochs):
                # sample for the first time
                arcs, (logits_1, logits_2) = controller.sample(True)
                model.set_arcs(arcs)
                
                # evaluate child model
                _, valid_acc = model.evaluate(valid_set)
                avg_valid_acc += valid_acc
                assert valid_acc >= 0.0, "Valid accuarcy should be positive"

                arc_1 = torch.tensor(arcs[0], dtype=torch.long).to(device)
                arc_2 = torch.tensor(arcs[1], dtype=torch.long).to(device)
                loss = controller.fit(valid_acc, [(arc_1, logits_1), (arc_2, logits_2)])

                prog.update(i + 1, [("loss", loss)])
            
            avg_valid_acc /= config.contro_train_epochs
            config.logger.info("- baseline: {}".format(controller.criterion.baseline))
            config.logger.info("- average acc: {}".format(avg_valid_acc))


if __name__ == "__main__":
    main()
