import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


class SentenceClassificationTrainer:
    def __init__(self, device, config, task, result_writer):
        self.device = device
        self.config = config
        self.task = task
        self.result_writer = result_writer

        # Initialize model
        self.model = BertHSLN(
            bert_model_name=config["bert_model"],
            hidden_dim=config["word_lstm_hs"],
            num_labels=task.num_labels,
            dropout=config["dropout"]
        ).to(device)

        # Optimizer and loss function
        self.optimizer = Adam(self.model.parameters(), lr=config["lr"])

    def run_training_for_fold(self, fold_num, fold_data, return_best_model=False):
        # Training loop
        best_model = None
        best_val_loss = float("inf")

        for epoch in range(self.config["max_epochs"]):
            self.model.train()

            for batch in fold_data:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(input_ids, attention_mask)

                # Calculate loss (CRF loss)
                loss = self.model.crf(output, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

            # Evaluate after every epoch
            val_loss = self.evaluate(fold_data)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model

        return best_model

    def evaluate(self, fold_data):
        # Evaluate the model on validation data and return the loss
        self.model.eval()
        val_loss = 0
        for batch in fold_data:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():
                output = self.model(input_ids, attention_mask)
                loss = self.model.crf(output, labels)
                val_loss += loss.item()

        return val_loss / len(fold_data)
