import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from config import PATTERN


def SequenceMask(X, X_len, value=-1e6):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float)[None, :] >= X_len[:, None]
    X[mask] = value
    return X


def masked_softmax(X, valid_length):
    # X: 3-D tensor, valid_length: 1-D or 2-D tensor
    softmax = nn.Softmax(dim=-1)
    if valid_length is None:
        return softmax(X)
    else:
        shape = X.shape
        if valid_length.dim() == 1:
            try:
                valid_length = torch.FloatTensor(valid_length.numpy().repeat(shape[1], axis=0))  # [2,2,3,3]
            except:
                valid_length = torch.FloatTensor(valid_length.cpu().numpy().repeat(shape[1], axis=0))  # [2,2,3,3]
        else:
            valid_length = valid_length.reshape((-1,))
        # fill masked elements with a large negative, whose exp is 0
        X = SequenceMask(X.reshape((-1, shape[-1])), valid_length)

        return softmax(X).reshape(shape)


class AdditiveAttention(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # `queries` 的形状：(`batch_size`，查询的个数，1，`num_hidden`)
        # `key` 的形状：(`batch_size`，1，“键－值”对的个数，`num_hiddens`)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # `self.w_v` 仅有一个输出，因此从形状中移除最后那个维度。
        # `scores` 的形状：(`batch_size`，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # `values` 的形状：(`batch_size`，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class AttentionDecoder(nn.Module):
    def __init__(self, num_hiddens, num_layers,
                 dropout=0.0, steps=5, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.rnn = nn.GRU(num_hiddens * 2, num_hiddens, num_layers, dropout=dropout)
        self.steps = steps

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # `outputs`的形状为 (`batch_size`，`num_steps`，`num_hiddens`).
        # `hidden_state`的形状为 (`num_layers`，`batch_size`，`num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, state):
        # `enc_outputs`的形状为 (`batch_size`, `num_steps`, `num_hiddens`).
        # `hidden_state`的形状为 (`num_layers`, `batch_size`,`num_hiddens`)
        enc_outputs, hidden_state, = state
        enc_valid_lens = None
        outputs, self._attention_weights = [], []
        out = torch.unsqueeze(enc_outputs.permute(1, 0, 2)[-1], dim=1)
        for _ in range(self.steps):
            # `query`的形状为 (`batch_size`, 1, `num_hiddens`)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # `context`的形状为 (`batch_size`, 1, `num_hiddens`)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结

            x = torch.cat((context, out), dim=-1)
            # 将 `x` 变形为 (1, `batch_size`, `embed_size` + `num_hiddens`)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            out = out.permute(1, 0, 2)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        return torch.cat(outputs, dim=1)

    @property
    def attention_weights(self):
        return self._attention_weights


class GRU_ATT_PET(nn.Module):
    def __init__(self, model_name, num_layers=2, dropout=0.1, steps=2):
        super(GRU_ATT_PET, self).__init__()
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.bert = self.model.bert
        self.cls = self.model.cls
        self.embeddings = self.bert.embeddings
        self.encoder = self.bert.encoder
        hidden_size = self.model.config.hidden_size
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers,
                          dropout=dropout, batch_first=True)
        self.seq2seq = AttentionDecoder(hidden_size, num_layers, dropout=dropout, steps=steps)

    def forward(self, input_ids, position_ids):
        embedding_output = self.embeddings(input_ids)  # [batch, num_steps, num_hidden]
        encoder = self.rnn(embedding_output)
        seq_out = self.seq2seq(encoder)
        first_p, end_p = torch.split(embedding_output, (1 + PATTERN, embedding_output.shape[1] - 1 - PATTERN), 1)
        final_p = torch.cat((first_p, seq_out, end_p), 1)
        final_encoder = self.encoder(final_p, attention_mask=torch.zeros(final_p.shape[1]).cuda())
        outputs = self.cls(final_encoder[0])
        return MaskedLMOutput(
            loss=None,
            logits=outputs,
            hidden_states=final_encoder.hidden_states,
            attentions=final_encoder.attentions,
        )
