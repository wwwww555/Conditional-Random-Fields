#模型定义
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class BertHSLN(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_labels, dropout=0.5):
        super(BertHSLN, self).__init__()

        # BERT model (using LEGAL-BERT or other models)
        self.bert = BertModel.from_pretrained(bert_model_name)

        # BiLSTM layer
        self.bilstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                              hidden_size=hidden_dim,
                              bidirectional=True,
                              batch_first=True)

        # Attention mechanism
        self.attn = Attention(hidden_dim)

        # CRF layer for sequence tagging
        self.crf = CRF(num_labels, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        # Get BERT embeddings
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

        # Apply BiLSTM
        lstm_out, _ = self.bilstm(last_hidden_state)

        # Apply Attention
        attn_out = self.attn(lstm_out)  # Get the sentence representation after attention pooling

        # Use CRF to decode the output
        output = self.crf.decode(attn_out)
        return output


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)  # BiLSTM is bidirectional, hence *2

    def forward(self, lstm_out):
        # Compute attention weights
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        weighted_sum = torch.sum(attn_weights * lstm_out, dim=1)
        return weighted_sum
