import time
import gc
from datetime import datetime
import torch
from model import BertHSLN
from trainer import SentenceClassificationTrainer
from utils import get_device, ResultWriter, log
from task import pubmed_task  # Assuming pubmed_task is defined elsewhere

# Configuration
config = {
    "bert_model": "bert-base-uncased",
    "bert_trainable": False,
    "model": BertHSLN.__name__,
    "dropout": 0.5,
    "word_lstm_hs": 758,
    "att_pooling_dim_ctx": 200,
    "att_pooling_num_ctx": 15,
    "lr": 3e-05,
    "lr_epoch_decay": 0.9,
    "batch_size": 32,
    "max_seq_length": 128,
    "max_epochs": 5,
    "early_stopping": 5,
}

MAX_DOCS = -1

def create_task(create_func):
    return create_func(train_batch_size=config["batch_size"], max_docs=MAX_DOCS)

task = create_task(pubmed_task)

save_best_models = True
device = get_device(0)

timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
run = f"{timestamp}_{task.task_name}_baseline"
run_results = f'results/{run}'

# Create output directory
os.makedirs(run_results, exist_ok=True)

# Preload data
task.get_folds()

restarts = 1 if task.num_folds == 1 else 1
for restart in range(restarts):
    for fold_num, fold in enumerate(task.get_folds()):
        start = time.time()
        result_writer = ResultWriter(f"{run_results}/{restart}_{fold_num}_results.jsonl")
        result_writer.log(f"Fold {fold_num} of {task.num_folds}")
        result_writer.log(f"Starting training {restart} for fold {fold_num}... ")

        trainer = SentenceClassificationTrainer(device, config, task, result_writer)
        best_model = trainer.run_training_for_fold(fold_num, fold, return_best_model=save_best_models)
        if best_model is not None:
            model_path = os.path.join(run_results, f'{restart}_{fold_num}_model.pt')
            result_writer.log(f"saving best model to {model_path}")
            torch.save(best_model.state_dict(), model_path)

        result_writer.log(f"finished training {restart} for fold {fold_num}: {time.time() - start}")

        gc.collect()

log("Training finished.")

