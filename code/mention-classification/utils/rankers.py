"""
Description: This module contains the implementation of the ReconstructionLossRanker class.

The code is adapted from the source code for the the paper:

  Kaufman, A. R. (2024). Selecting More Informative Training Sets with Fewer Observations. Political Analysis, 32(1), 133–139. doi: 10.1017/pan.2023.19

The source code is available via Harvard dataverse: https://doi.org/10.7910/DVN/4ROL8S
"""

import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random

from typing import List, Union

from mention_classification.utils import log

def reconstruction_loss(output, input):
  return 0.5 * torch.square(torch.norm(input-output, p='fro')) / input.size()[1]

class ReconstructionLossRanker(nn.Module):
  """
  Autoencoder model with a single hidden layer for sparse reconstruction of a dataset

  The model takes in a dataset and attempts to reconstruct the entire dataset.
   This is done with a weight matrix while applying sparsity regularization on the weights.
  Regularization limits the amount of data that can be used for reconstruction and effectively prevents that the model learns an identity function).

  The learned model allows identifying the "most relevant" observations that can reconstruct the entire dataset.

  Args:
    hdim (int): The number of hidden units in the model.

  Attributes:
    hdim (int): The number of hidden units in the model.
    weights (torch.nn.Parameter): The weight matrix of the model.

  Methods:
    forward(input): Forward pass of the model.
  """
  def __init__(
      self,
      hdim: int,
      learning_rate: float = 0.005,
      num_epochs: float = 20000,
      log_n_steps: int = 500,
      regularization_coefficient: float = 0.01,
      device: Union[str, torch.device] = 'cpu',
      seed: int = 42,
    ):
    super(ReconstructionLossRanker, self).__init__()
    self.hdim = hdim

    # set hyperparameters
    self.learning_rate = learning_rate
    self.num_epochs = num_epochs
    self.log_n_steps = log_n_steps
    self.regularization_coefficient = regularization_coefficient

    self.device = torch.device(device) if isinstance(device, str) else device

    # set seed
    self.seed = seed
    random.seed(self.seed)
    np.random.seed(self.seed)
    torch.manual_seed(self.seed)

    # init weights
    w = torch.zeros((hdim, hdim), device=self.device)
    torch.nn.init.xavier_normal_(w)
    w.requires_grad = True
    self.weights = nn.Parameter(w)

  def forward(self, input):
    x = torch.matmul(input, self.weights)
    return x

  def fit(
      self,
      data: Union[List[List[float]], pd.DataFrame, np.ndarray],
      verbose: bool = True
    ):
    """
    Fit the model to the data.

    Args:
      data (Union[List[List[float]], pd.DataFrame, np.ndarray]): The dataset to fit the model to.
      learning_rate (float): The learning rate for the optimizer.
      num_epochs (float): The number of epochs to train the model.
      regularization_coefficient (float): The coefficient for the regularization term.
    """
    if isinstance(data, pd.DataFrame):
      data = torch.tensor(data.values, device=self.device)
    if isinstance(data, np.ndarray):
      data = torch.tensor(data, device=self.device)
    if isinstance(data, list):
      data = torch.tensor(data, device=self.device)

    # prepare the data
    data = data.t()

    # prepare training
    optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    self.loss_list_ = []
    prev_eval_loss = 99999

    # train and evaluate
    for epoch in range(self.num_epochs):
      # turn on training mode
      self.train()

      # forward pass
      output = self(data.float())
      reg21 = torch.sum(torch.norm(self.weights, p=2, dim=1))
      loss = reconstruction_loss(output, data) + self.regularization_coefficient * reg21

      # backward pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # log
      if verbose and self.log_n_steps > 0 and epoch % self.log_n_steps == 0:
        log(f'epoch [{epoch + 1}/{self.num_epochs}], loss: {loss.item():.4f}')

      # evaluate
      self.eval()
      eval_loss = F.l1_loss(self(data.float()), data.float()).item()
      self.loss_list_.append(eval_loss)
      if verbose and self.log_n_steps > 0 and epoch % self.log_n_steps == 0:
        #log('Eval loss: ', eval_loss)
        pass

      # stop of update stoping criterion
      if eval_loss > prev_eval_loss:
        if verbose: log(f'Final loss after {epoch+1} epochs: {eval_loss:.4f}')
        break
      else:
        prev_eval_loss = eval_loss

    # get list of row indexes (in original data) in descending order of importance for reconstruction
    self.importance_indexes_ = torch.argsort(torch.norm(self.weights, p=2, dim=1), descending=True).cpu().numpy()

    return self.importance_indexes_, self.loss_list_

  def transform(
      self,
      data: Union[List[List[float]], pd.DataFrame, np.ndarray],
      k: Union[int, None]=None
    ):
    if isinstance(data, np.ndarray):
      n = data.shape[0]
    else:
      n = len(data)
    if k is None:
      k = n

    assert k <= n, 'k must be less than or equal to the number of columns in the data'

    if isinstance(data, pd.DataFrame):
      data = data.iloc[self.importance_indexes_[:k], :]
    if isinstance(data, np.ndarray):
      data = data[self.importance_indexes_[:k]]
    if isinstance(data, list):
      data = [data[i] for i in self.importance_indexes_[:k]]

    return data

  def fit_transform(
      self,
      data: Union[List[List[float]], pd.DataFrame, np.ndarray],
      k: Union[int, None]=None
    ):
    if isinstance(data, np.ndarray):
      n = data.shape[0]
    else:
      n = len(data)
    if k is None:
      k = n

    assert k <= n, 'k must be less than or equal to the number of columns in the data'

    self.fit(data)

    return self.transform(data, k=k)
