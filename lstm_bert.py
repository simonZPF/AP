import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from config import PATTERN


class LSTM_PET(nn.Module):
    def __init__(self, model_name="hfl/chinese-roberta-wwm-ext"):
        super(LSTM_PET, self).__init__()
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.bert = self.model.bert
        self.cls = self.model.cls
        self.embeddings = self.bert.embeddings
        self.encoder = self.bert.encoder
        self.lstm = nn.LSTM(self.model.config.hidden_size, self.model.config.hidden_size, 1, batch_first=True)
        self.cell = nn.LSTMCell(self.model.config.hidden_size, self.model.config.hidden_size)

    def forward(self, input_ids, position_ids):
        embedding_output = self.embeddings(input_ids)  # [batch, len, 768]
        lstm_out = self.lstm(embedding_output)
        seq_output = []
        state, hx, cx = lstm_out[0].permute(1, 0, 2), lstm_out[1][0][0], lstm_out[1][1][0]
        for i in range(5):
            hx, cx = self.cell(state[-(6-i)], (hx, cx))
            seq_output.append(hx)
        seq_out = torch.stack(seq_output)
        first_p, end_p = torch.split(embedding_output, (1+PATTERN, embedding_output.shape[1] - 1-PATTERN), 1)
        final_p = torch.cat((first_p, seq_out.permute(1, 0, 2), end_p), 1)
        final_encoder = self.encoder(final_p,attention_mask=torch.zeros(final_p.shape[1]).cuda())
        outputs = self.cls(final_encoder[0])
        return MaskedLMOutput(
            loss=None,
            logits=outputs,
            hidden_states=final_encoder.hidden_states,
            attentions=final_encoder.attentions,
        )
