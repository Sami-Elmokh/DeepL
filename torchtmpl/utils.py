# coding: utf-8

# Standard imports
import os

# External imports
import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
import numpy as np
import torch.optim as opt


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


def top_30_error_rate(outputs, targets):
    # if outputs has a size of (batch_size, number_classes)
    # and targets has size (batch_size,1) where the second dimension indicates the class
    n_batch = outputs.shape[0]
    sum_e_i = 0
    # for each observation i in the batch
    for i in range(n_batch):
        # extract the 30 classes with highest probability
        values, top_30 = torch.topk(outputs[i], k=30, dim=1)
        # if none of them is the ground truth label then ei=1, else, it is 0
        e_i = 1
        for j in range(30):
            if top_30[j] == targets[i]:
                e_i = 0
        sum_e_i += e_i
    return (sum_e_i/n_batch)


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False


def train(model, loader, f_loss, optimizer, device, lr_scheduler ,dynamic_display=True):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    optimizer -- A torch.optim.Optimzer object
    device    -- A torch.device
    Returns :
    The averaged train metrics computed over a sliding window
    """

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0
    for i, (inputs, targets) in (pbar := tqdm.tqdm(enumerate(loader))):
        print ("batch",i)
        inputs, targets = inputs.to(device), targets.to(device)
        input_nan = torch.isnan(inputs).any().item()

        # standardize batch
        #inputs = (inputs-inputs.mean())/inputs.std()

        # Compute the forward propagation
        if input_nan:
            torch.where(torch.isnan(inputs),torch.tensor(0.0),inputs)
        input_nan = torch.isnan(inputs).any().item()

        outputs = model(inputs.float())
        print("max_value inputs = ",torch.max(inputs))
        print ("shape", outputs.shape)
        print("max_value outputs = ",torch.max(outputs))

        # epsilon= 1e-8
        pbar.set_description(f"memory allocated : {torch.cuda.memory_allocated():.2f}")
        pbar.set_description(f"memory cached : {torch.cuda.memory_cached():.2f}")

        output_nan = torch.isnan(outputs).any().item()

        # outputs = F.log_softmax(outputs+epsilon, dim=1)
        # output_softmax_nan = torch.isnan(outputs).any().item()

        #outputs = F.log_softmax(outputs+epsilon, dim=1)

        # if len(targets.shape) > 1:
        #         targets = torch.argmax(targets, dim=1)
                
        loss = f_loss(outputs, targets)

        loss_penalty = loss
        # for j in model.parameters():
        #     l2_norm = (torch.sum(j ** 2)).to(device=device)
        #     loss_penalty += 0.1 * l2_norm

        
        # Backward and optimize
        optimizer.zero_grad()
        loss_penalty.backward()
        optimizer.step()
        # grad_clip=0.5
        # if grad_clip is not None:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update the metrics
        # We here consider the loss is batch normalized
        #pbar.set_description(f"loss : {loss:.2f}")
        #print("printing", loss.item())
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]
        pbar.set_description(f"Train loss : {loss.item():.2f}")
        pbar.set_description(f"Input Nan test : {input_nan}")
        pbar.set_description(f"Output Nan test : {output_nan}")
        #pbar.set_description(f"Output Softmax Nan test : {output_softmax_nan}")




        # Add evaluation metric
        #top_30_error = top_30_error_rate(outputs, targets)
    lr_scheduler.step(loss.item()/len(loader))   #Setting up lr decay  

    return total_loss / num_samples#, top_30_error)


def test(model, loader, f_loss, device):
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    total_loss = 0
    num_samples = 0
    for (inputs, targets) in loader:

        inputs, targets = inputs.to(device), targets.to(device)

        # standardize batch
        #inputs = (inputs-inputs.mean())/inputs.std()

        # Compute the forward propagation
        input_nan = torch.isnan(inputs).any().item()

        # standardize batch
        #inputs = (inputs-inputs.mean())/inputs.std()

        # Compute the forward propagation
        if input_nan:
            torch.where(torch.isnan(inputs),torch.tensor(0.0),inputs)
        input_nan = torch.isnan(inputs).any().item()

        outputs = model(inputs.float())

        loss = f_loss(outputs, targets)


        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]

        # Add evaluation metric
        #top_30_error = top_30_error_rate(outputs, targets)

    return total_loss / num_samples#, top_30_error
