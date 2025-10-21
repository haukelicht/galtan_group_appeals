import numpy as np
from torch import nn

import torch
from setfit import SetFitModel
from setfit import Trainer, TrainingArguments

from datasets import Dataset
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PredictionOutput
from transformers import EarlyStoppingCallback
try:
    import optuna
except:
    pass

from tqdm import tqdm, trange
import warnings

from dataclasses import dataclass
from copy import deepcopy
from typing import List, Dict, Optional, Union, Callable, Any, Tuple


class SetFitModelWithEarlyStopping(SetFitModel):
    """
    A SetFit model with early stopping functionality.
    """
    def fit(
        self,
        x_train: List[str],
        y_train: Union[List[int], List[List[int]]],
        x_eval: Optional[List[str]] = None,
        y_eval: Optional[Union[List[int], List[List[int]]]] = None,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        body_learning_rate: Optional[float] = None,
        head_learning_rate: Optional[float] = None,
        end_to_end: bool = False,
        l2_weight: Optional[float] = None,
        max_length: Optional[int] = None,
        show_progress_bar: bool = True,
        # Early stopping parameters
        early_stopping_patience: Optional[int] = None,
        early_stopping_threshold: float = 0.0,
        metric_for_best_model: str = "loss",
        compute_metrics: Optional[callable] = None,
        greater_is_better: bool = False,
        load_best_model_at_end: bool = True,
    ) -> None:
        """Train the classifier head, only used if a differentiable PyTorch head is used.

        Args:
            x_train (`List[str]`): A list of training sentences.
            y_train (`Union[List[int], List[List[int]]]`): A list of labels corresponding to the training sentences.
            num_epochs (`int`): The number of epochs to train for.
            x_eval (`List[str]`, *optional*): A list of validation sentences.
            y_eval (`Union[List[int], List[List[int]]]`, *optional*): A list of labels corresponding to the validation sentences.
            batch_size (`int`, *optional*): The batch size to use.
            body_learning_rate (`float`, *optional*): The learning rate for the `SentenceTransformer` body
                in the `AdamW` optimizer. Disregarded if `end_to_end=False`.
            head_learning_rate (`float`, *optional*): The learning rate for the differentiable torch head
                in the `AdamW` optimizer.
            end_to_end (`bool`, defaults to `False`): If True, train the entire model end-to-end.
                Otherwise, freeze the `SentenceTransformer` body and only train the head.
            l2_weight (`float`, *optional*): The l2 weight for both the model body and head
                in the `AdamW` optimizer.
            max_length (`int`, *optional*): The maximum token length a tokenizer can generate. If not provided,
                the maximum length for the `SentenceTransformer` body is used.
            show_progress_bar (`bool`, defaults to `True`): Whether to display a progress bar for the training
                epochs and iterations.
            early_stopping_patience (`int`, *optional*): Number of epochs with no improvement 
                after which training will be stopped. Only used if validation set is provided.
            early_stopping_threshold (`float`, defaults to 0.0): Minimum change to qualify as improvement.
            metric_for_best_model (`str`, defaults to "loss"): Metric to monitor for early stopping.
            greater_is_better (`bool`, defaults to False): Whether higher values of the metric indicate better models.
        """
        if self.has_differentiable_head:  # train with pyTorch
            self.model_body.train()
            self.model_head.train()
            if not end_to_end:
                self.freeze("body")

            dataloader = self._prepare_dataloader(x_train, y_train, batch_size, max_length)
            if x_eval is not None and y_eval is not None:
                eval_dataloader = self._prepare_dataloader(x_eval, y_eval, batch_size, max_length)
            criterion = self.model_head.get_loss_fn()
            optimizer = self._prepare_optimizer(head_learning_rate, body_learning_rate, l2_weight)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

            # Early stopping setup
            best_metric = float("inf") if not greater_is_better else float("-inf")
            best_model_state = None
            no_improvement_count = 0

            for epoch_idx in trange(num_epochs, desc="Epoch", disable=not show_progress_bar):
                # Training loop - existing code
                epoch_loss = 0.0
                num_batches = 0
                
                for batch in tqdm(dataloader, desc="Iteration", disable=not show_progress_bar, leave=False):
                    features, labels = batch
                    optimizer.zero_grad()

                    # to model's device
                    features = {k: v.to(self.device) for k, v in features.items()}
                    labels = labels.to(self.device)

                    outputs = self.model_body(features)
                    if self.normalize_embeddings:
                        outputs["sentence_embedding"] = nn.functional.normalize(
                            outputs["sentence_embedding"], p=2, dim=1
                        )
                    outputs = self.model_head(outputs)
                    logits = outputs["logits"]

                    loss: torch.Tensor = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                # Calculate epoch average loss
                epoch_avg_loss = epoch_loss / max(1, num_batches)
                
                # Validation and early stopping logic
                if x_eval is not None and y_eval is not None and early_stopping_patience is not None:
        
                    val_metric = self._compute_eval_metric(eval_dataloader, metric_for_best_model, compute_metrics, epoch_avg_loss, show_progress_bar)
                    
                    improved = (greater_is_better and val_metric > best_metric + early_stopping_threshold) or \
                               (not greater_is_better and val_metric < best_metric - early_stopping_threshold)
                               
                    if improved:
                        best_metric = val_metric
                        # Save model state
                        best_model_state = {"body": None, "head": None}
                        if end_to_end:
                            best_model_state["body"] = {k: v.cpu() for k, v in self.model_body.state_dict().items()}
                        best_model_state["head"] = {k: v.cpu() for k, v in self.model_head.state_dict().items()}
                        
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    # Early stopping check
                    if no_improvement_count >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch_idx + 1} epochs")
                        break
                
                scheduler.step()
            
            # Load best model if early stopping occurred
            if load_best_model_at_end and best_model_state is not None:
                print(f"Loading best model")
                if best_model_state["body"] is not None:
                    self.model_body.load_state_dict(
                        {k: v.to(self.model_body.device) for k, v in best_model_state["body"].items()}
                    )
                self.model_head.load_state_dict(
                    {k: v.to(self.model_head.device) for k, v in best_model_state["head"].items()}
                )

            if not end_to_end:
                self.unfreeze("body")
        else:  # train with sklearn
            if early_stopping_patience:
                warnings.warn(
                    "Early stopping not supported for non-differentiable head",
                    NotImplementedError,
                    stacklevel=2
                )
            
            embeddings = self.model_body.encode(x_train, normalize_embeddings=self.normalize_embeddings)
            self.model_head.fit(embeddings, y_train)
            if self.labels is None and self.multi_target_strategy is None:
                # Try to set the labels based on the head classes, if they exist
                # This can fail in various ways, so we catch all exceptions
                try:
                    classes = self.model_head.classes_
                    if classes.dtype.char == "U":
                        self.labels = classes.tolist()
                except Exception:
                    pass
    
    def _compute_eval_metric(self, eval_dataloader, metric_name, compute_metrics, train_loss: float, show_progress_bar=True):
        """
        Compute the validation metric for early stopping.
        """
        self.model_body.eval()
        self.model_head.eval()

        val_loss = 0.0
        num_batches = 0
        all_logits = []
        all_labels = []
        
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not show_progress_bar, leave=False):
            features, labels = batch
            all_labels.extend(self._output_type_conversion(labels, as_numpy=True))
            with torch.no_grad():
                features = {k: v.to(self.device) for k, v in features.items()}
                labels = labels.to(self.device)

                outputs = self.model_body(features)
                if self.normalize_embeddings:
                    outputs["sentence_embedding"] = nn.functional.normalize(
                        outputs["sentence_embedding"], p=2, dim=1
                    )
                outputs = self.model_head(outputs)
                logits = outputs["logits"]

                loss: torch.Tensor = self.model_head.get_loss_fn()(logits, labels)
                val_loss += loss.item()
                num_batches += 1
                logits = self._output_type_conversion(logits, as_numpy=True)
                all_logits.append(logits)
        val_avg_loss = val_loss / max(1, num_batches)
        
        val_metric = val_avg_loss
        metrics = {'training loss': train_loss, 'validation loss': val_avg_loss}
        if compute_metrics is not None:
            logits = np.concatenate(all_logits, axis=0)
            p = PredictionOutput(predictions=logits, label_ids=np.array(all_labels), metrics=None)
            metrics.update(compute_metrics(p))
            if metric_name in metrics:
                val_metric = metrics[metric_name]
            else:
                warnings.warn(
                    f"metric \"{metric_name}\" not in output of `compute_metrics` function. "
                    f"Using validation loss as fallback",
                    stacklevel=2,
                )
        # report # TODO: make use of transformers eval loop reporting utils
        print(metrics)
            
        return val_metric

@dataclass
class EarlyStoppingTrainingArguments(TrainingArguments):
    metric_for_best_model: Optional[Tuple[str, str]] = ('embedding_loss', 'loss')
    greater_is_better: Tuple[bool, bool] = (False, False)

class EarlyStoppingTrainer(Trainer):

    def __init__(
        self,
        model: Optional["SetFitModel"] = None,
        args: Optional[TrainingArguments] = None,
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        model_init: Optional[Callable[[], "SetFitModel"]] = None,
        metric: Union[str, Callable[["Dataset", "Dataset"], Dict[str, float]]] = "accuracy",
        metric_kwargs: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        compute_metrics: Optional[Callable] = None,
    ) -> None:
        old_callbacks = deepcopy(callbacks)
        
        # remove duplicate early stopping callback, from sentence transformer trainer, if any
        if sum(isinstance(c, EarlyStoppingCallback) for c in callbacks) > 1:
            new_callbacks = []
            surplus_eac = False
            while len(callbacks)>0:
                cb = callbacks.pop()
                if not isinstance(cb, EarlyStoppingCallback):
                    new_callbacks.append(cb)
                elif not surplus_eac:
                    new_callbacks.append(cb)
                    surplus_eac = True
                else:
                    pass
            new_callbacks
        else:
            new_callbacks = old_callbacks
        
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            metric=metric,
            metric_kwargs=metric_kwargs,
            callbacks=new_callbacks,
            column_mapping=column_mapping,
        )

        self._callbacks = old_callbacks
        self._compute_metrics = compute_metrics

    @property
    def args(self) -> Union[TrainingArguments, EarlyStoppingTrainingArguments]:
        return self._args

    @args.setter
    def args(self, args: Union[TrainingArguments, EarlyStoppingTrainingArguments]) -> None:
        self._args = args
        if hasattr(self, "st_trainer"):
            if isinstance(args, EarlyStoppingTrainingArguments):
                if isinstance(args.metric_for_best_model, tuple):
                    args.metric_for_best_model = args.metric_for_best_model[0]
                if isinstance(args.greater_is_better, tuple):
                    args.greater_is_better = args.greater_is_better[0]
            self.st_trainer.setfit_args = args

    @property
    def model(self) -> Union["SetFitModel", "SetFitModelWithEarlyStopping"]:
        return self._model

    @model.setter
    def model(self, model: Union["SetFitModel", "SetFitModelWithEarlyStopping"]) -> None:
        self._model = model
        if hasattr(self, "st_trainer"):
            self.st_trainer.setfit_model = model

        
    def train(
        self,
        args: Optional[TrainingArguments] = None,
        trial: Optional[Union["optuna.Trial", Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Main training entry point.

        Args:
            args (`TrainingArguments`, *optional*):
                Temporarily change the training arguments for this training call.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        if len(kwargs):
            warnings.warn(
                f"`{self.__class__.__name__}.train` does not accept keyword arguments anymore. "
                f"Please provide training arguments via a `TrainingArguments` instance to the `{self.__class__.__name__}` "
                f"initialisation or the `{self.__class__.__name__}.train` method.",
                DeprecationWarning,
                stacklevel=2,
            )

        if trial:  # Trial and model initialization
            self._hp_search_setup(trial)  # sets trainer parameters and initializes model

        args = args or self.args or EarlyStoppingTrainingArguments() # NOTE: using subclassed train args here 

        if self.train_dataset is None:
            raise ValueError(
                f"Training requires a `train_dataset` given to the `{self.__class__.__name__}` initialization."
            )

        train_parameters = self.dataset_to_parameters(self.train_dataset)
        full_parameters = (
            train_parameters + self.dataset_to_parameters(self.eval_dataset) if self.eval_dataset else train_parameters
        )

        self.train_embeddings(*full_parameters, args=args)
        self.train_classifier(*full_parameters, args=args) # NOTE: also pass eval x and y to classifier training
    
    def train_embeddings(
        self,
        x_train: List[str],
        y_train: Optional[Union[List[int], List[List[int]]]] = None,
        x_eval: Optional[List[str]] = None,
        y_eval: Optional[Union[List[int], List[List[int]]]] = None,
        args: Optional[TrainingArguments] = None,
    ):
        args = deepcopy(args)
        if isinstance(args.metric_for_best_model, tuple):
            args.metric_for_best_model = args.metric_for_best_model[0]
        if isinstance(args.greater_is_better, tuple):
            args.greater_is_better = args.greater_is_better[0]
        super().train_embeddings(x_train, y_train, x_eval, y_eval, args)

    def train_classifier(
        self,
        x_train: List[str],
        y_train: Optional[Union[List[int], List[List[int]]]] = None,
        x_eval: Optional[List[str]] = None,
        y_eval: Optional[Union[List[int], List[List[int]]]] = None,
        args: Optional[TrainingArguments] = None,
    ):
        args = deepcopy(args)
        if isinstance(self.model, SetFitModelWithEarlyStopping):
            early_stopping_kwargs = {}
            
            if isinstance(args, EarlyStoppingTrainingArguments):
                early_stopping_kwargs["metric_for_best_model"] = args.metric_for_best_model[1] if isinstance(args.metric_for_best_model, tuple) else "loss"
                early_stopping_kwargs["greater_is_better"] = args.greater_is_better[1] if isinstance(args.greater_is_better, tuple) else False 
                if early_stopping_kwargs["metric_for_best_model"]=="loss" and early_stopping_kwargs["greater_is_better"]:
                    # TODO: warn that EarlyStoppingTrainingArguments.greater_is_better[1] should be false when using "loss"
                    pass
                early_stopping_kwargs["load_best_model_at_end"] = args.load_best_model_at_end
            else:
                early_stopping_kwargs["metric_for_best_model"] = "loss"
                early_stopping_kwargs["greater_is_better"] = False

            if any(isinstance(c, EarlyStoppingCallback) for c in self._callbacks):
                cb = [c for c in self._callbacks if isinstance(c, EarlyStoppingCallback)][-1]
                early_stopping_kwargs["early_stopping_patience"] = cb.early_stopping_patience
                early_stopping_kwargs["early_stopping_threshold"] = cb.early_stopping_threshold
            if isinstance(self._compute_metrics, Callable):
                early_stopping_kwargs["compute_metrics"] = self._compute_metrics
            
            self.model.fit(
                x_train,
                y_train,
                x_eval,
                y_eval,
                num_epochs=args.classifier_num_epochs,
                batch_size=args.classifier_batch_size,
                body_learning_rate=args.body_classifier_learning_rate,
                head_learning_rate=args.head_learning_rate,
                l2_weight=args.l2_weight,
                max_length=args.max_length,
                show_progress_bar=args.show_progress_bar,
                end_to_end=args.end_to_end,
                # early stopping args
                **early_stopping_kwargs
            )

