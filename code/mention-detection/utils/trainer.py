import os
import json
import shutil
from typing import Union, Callable, Union, Callable
from datasets import Dataset
import torch
from transformers import (
    PreTrainedTokenizer, 
    DefaultDataCollator, 
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    Trainer,
)

class WriteValidationResultsCallback(TrainerCallback):
    """Trainer callback to write validation set results to disk while training"""
    def __init__(self, path='validation_results.jsonl', overwrite=True):
        super().__init__()
        self.path = path
        if overwrite:
            with open(self.path, 'w') as f:
                f.write('')
    
    def on_evaluate(self, args, state, control, **kwargs):
        validation_results = state.log_history[-1]
        with open(self.path, "a") as f:
            f.write(json.dumps(validation_results) + "\n")

def train_and_test(
    experiment_name: str,
    experiment_results_path: str,
    run_id: Union[None, str],
    model_init: Callable,
    tokenizer: PreTrainedTokenizer,
    data_collator: DefaultDataCollator,
    train_dat: Dataset,
    dev_dat: Union[None, Dataset],
    test_dat: Union[None, Dataset],
    compute_metrics: Callable,
    metric: str,
    epochs: int = TrainingArguments.num_train_epochs,
    learning_rate: float = TrainingArguments.learning_rate,
    train_batch_size: int = TrainingArguments.per_device_train_batch_size,
    gradient_accumulation_steps: int = TrainingArguments.gradient_accumulation_steps,
    fp16_training: bool = True,
    eval_batch_size: int = TrainingArguments.per_device_eval_batch_size,
    weight_decay: float = TrainingArguments.weight_decay,
    early_stopping: bool = True,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.03,
    seed: int = 42,
):
    """
    Fine-tune and evaluate a Transformer model.

    Args:
        experiment_name (str): 
            Name of the experiment. Used for creating directories for saving results.
        experiment_results_path (str): 
            Base path where experiment results will be saved.
        run_id (Union[None, str]): 
            Optional unique identifier for the run. If None, an identifier will be generated.
        model_init (Callable): 
            A function that initializes the model to be trained.
        tokenizer (PreTrainedTokenizer): 
            Tokenizer for preprocessing text data.
        data_collator (DefaultDataCollator): 
            Data collator that batches data samples.
        train_dat (Dataset): 
            The dataset used for training.
        dev_dat (Union[None, Dataset]): 
            The dataset used for validation. If None, validation is skipped.
        test_dat (Union[None, Dataset]): 
            The dataset used for testing. If None, testing is skipped.
        compute_metrics (Callable): 
            Function to compute metrics based on predictions and true labels.
        metric (str): 
            Name of the metric to be used for evaluation.
        epochs (int): 
            Number of training epochs. Defaults to TrainingArguments.num_train_epochs.
        learning_rate (float): 
            Learning rate for the optimizer. Defaults to TrainingArguments.learning_rate.
        train_batch_size (int): 
            Batch size for training. Defaults to TrainingArguments.per_device_train_batch_size.
        gradient_accumulation_steps (int): 
            Number of steps to accumulate gradients before updating model parameters. Defaults to TrainingArguments.gradient_accumulation_steps.
        fp16_training (bool): 
            Whether to use mixed precision training. Defaults to True.
        eval_batch_size (int): 
            Batch size for evaluation. Defaults to TrainingArguments.per_device_eval_batch_size.
        weight_decay (float): 
            Weight decay for the optimizer. Defaults to TrainingArguments.weight_decay.
        early_stopping (bool): 
            Whether to use early stopping. Defaults to True.
        early_stopping_patience (int): 
            Number of evaluations with no improvement after which training will be stopped. Defaults to 3.
        early_stopping_threshold (float): 
            Minimum change in the monitored metric to qualify as an improvement. Defaults to 0.03.
        seed (int): 
            Random seed for reproducibility. Defaults to 42.
        
    Returns:
        Trainer: 
            Trainer object used for training.
        str: 
            Path to the best model checkpoint.
        dict: 
            Evaluation results on the test set.
    """
    results_path = os.path.join(experiment_results_path, experiment_name)
    os.makedirs(results_path, exist_ok=True)

    output_path = os.path.join(results_path, 'checkpoints')
    logs_path = os.path.join(results_path, 'logs')

    # note: the following training options depend on the availability of a dev set and will be disabled if none is provided
    #  - evaluating after each epoch
    #  - early stopping
    #  - saving at most 2 models during training
    #  - saving the best model at the end
    #  - saving the dev results

    training_args = TrainingArguments(
        # hyperparameters
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=weight_decay,
        optim='adamw_torch',
        # how to select "best" model
        do_eval=dev_dat is not None,
        metric_for_best_model=metric,
        load_best_model_at_end=True,
        # when to evaluate
        evaluation_strategy='epoch',
        # when to save
        save_strategy='epoch',
        save_total_limit=2 if dev_dat is not None else None, # don't save all model checkpoints
        # where to store results
        output_dir=output_path,
        overwrite_output_dir=True,
        # logging
        logging_dir=logs_path,
        logging_strategy='epoch',
        # efficiency
        fp16=fp16_training if torch.cuda.is_available() else False,
        fp16_full_eval=False,
        # reproducibility
        seed=seed,
        data_seed=seed,
        full_determinism=True
    )

    # build callbacks
    callbacks = []
    if early_stopping:
        if dev_dat is None:
            raise ValueError('Early stopping requires a dev data set')
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold))
    if dev_dat:
        fn = run_id+'-dev_results.jsonl' if run_id else 'dev_results.jsonl'
        fp = os.path.join(results_path, fn)
        callbacks.append(WriteValidationResultsCallback(path=fp))

    # train
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dat,
        eval_dataset=dev_dat if dev_dat is not None else None,
        tokenizer=tokenizer,
        data_collator=data_collator if data_collator is not None else None,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    print('Training ...')
    _ = trainer.train()

    # save best model to results folder
    # CAVEAT: this is not the "best" model if no dev_dat is provided
    dest = run_id+'-best_model' if run_id else 'best_model'
    dest = os.path.join(results_path, dest)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    trainer.save_model(dest)
    # save tokenizer to best_model folder
    tokenizer.save_pretrained(dest)

    # evaluate
    if test_dat:
        print('Evaluating ...')
        res = trainer.evaluate(test_dat, metric_key_prefix='test')
        print(res)
        fn = run_id+'-test_results.json' if run_id else 'test_results.json'
        fp = os.path.join(results_path, fn)
        with open(fp, 'w') as file:
            json.dump(res, file)
    else:
      res = None

    # finally: clean up
    if os.path.exists(output_path):
        # TODO: reconsider this when dev_dat is None (in this case, no best model will be copied and deliting the output path would delete any model checkpoints)
        shutil.rmtree(output_path)
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)

    return trainer, dest, res
