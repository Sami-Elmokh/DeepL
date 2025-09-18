# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
import csv
import torch.optim as opt
import torchvision.models as m

# Local imports
from . import data
from . import models
from . import optim
from . import utils




def train(config):

    use_cuda = torch.cuda.is_available()
    print("cuda", use_cuda)
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    
    if "wandb" in config["logging"]:
        wandb.login(key="d4caf58e4bc882d64ca04237c26e75b4725dcde1")

        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info("Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader = data.get_dataloaders(True,
        data_config, use_cuda)


    # Build the model
    logging.info("= Model")

    model_config = config["model"]

    model = m.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, model_config["num_classes"])

    # model = models.build_model(model_config, model_config["cin"] , model_config["num_classes"])

    
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"])

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logging.info("= copy")

    logdir = pathlib.Path(logdir)
    with open(logdir / "config-sample.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    logging.info("= summary")

    input_size = next(iter(train_loader))[0].shape
    logging.info("= summary text")

    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}"
    )
    logging.info("= open summary")

    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)

    logging.info("= wandb")

    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    logging.info("= Checkpoint")
    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3, verbose = 1)
    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )
    for e in range(config["nepochs"]):
        # Train 1 epoch
        logging.info("= Train")
        #logging.info("epoch is -----", e)
        print("current epoch is", e)

        train_loss = utils.train(model, train_loader, loss, optimizer, device, lr_scheduler)
        # Test
        test_loss = utils.test(model, valid_loader, loss, device)

        updated = model_checkpoint.update(test_loss)
        logging.info(
            "[%d/%d] Test loss : %.3f %s"
            % (
                e,
                config["nepochs"],
                test_loss,
                "[>> BETTER <<]" if updated else "",
            )
        )


        # Update the dashboard
        metrics = {"train_CE": train_loss, "test_CE": test_loss}
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)
    torch.save(model.state_dict(),'model_parameters.pth')


def test(config, simple=True):
    use_cuda = torch.cuda.is_available()
    print("cuda", use_cuda)
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    
    # if "wandb" in config["logging"]:
    #     wandb.login(key="d4caf58e4bc882d64ca04237c26e75b4725dcde1")

    #     wandb_config = config["logging"]["wandb"]
    #     wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
    #     wandb_log = wandb.log
    #     wandb_log(config)
    #     logging.info("Will be recording in wandb run name : {wandb.run.name}")
    # else:
    #     wandb_log = None

    # # Build the dataloaders
    # logging.info("= Building the dataloaders")
    # data_config = config["data"]

    # test_dataloader
    data_config = config["data"]

    test_loader= data.get_test_dataloaders(data_config,use_cuda)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # # Build the model
    model_config = config["model"]  
    if simple:
        model = m.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, model_config["num_classes"])
    
    else:
        model = models.build_model(model_config, model_config["cin"] , model_config["num_classes"])

    # Load the trained parameters (weights) into the model from the checkpoint file
    model.load_state_dict(torch.load('/usr/users/sdi-labworks-2023-2024/sdi-labworks-2023-2024_34/deep-learning-project/logs/ResNet50_39/best_model.pt'))

    with open('/usr/users/sdi-labworks-2023-2024/sdi-labworks-2023-2024_34/deep-learning-project/logs/ResNet50_39/submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Id','Predicted'])
        print("inside loop")
        # Iterate over data and make predictions
        for input, observation_id in test_loader:
            with torch.no_grad():
                output = model(input.float())

            # Get top 30 indices for each prediction
            top_k_values, top_k_indices = torch.topk(output, k=30, dim=1)

            # Convert indices to space-separated string
            s_pred = " ".join(map(str, top_k_indices.squeeze().tolist()))

            writer.writerow([observation_id.item(), s_pred])






    #raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
