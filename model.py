import math
from os import terminal_size
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from graph import TypeRelationGraph
from torch_geometric.nn import GATConv
from candidate_types import Candidate

# import constants
from constants import *


class KaFSP(nn.Module):
    def __init__(self, vocabs):
        super(KaFSP, self).__init__()
        self.vocabs = vocabs
        self.encoder = Encoder(vocabs[INPUT], DEVICE)
        self.decoder = Decoder(vocabs[LOGICAL_FORM], DEVICE)
        self.ner = NerNet(len(vocabs[NER]))
        self.coref = CorefNet(len(vocabs[COREF]))
        self.graph = TypeRelationGraph(vocabs[GRAPH]).data
        self.graph_net = GraphNet(len(vocabs[GRAPH]))
        self.candidates = Candidate(vocabs[TYPE_REL_RECOGNIZE])
        # self.candidate_types = CandidateType()
        # self.candidate_relations = CandidateRelation()
        self.type_rel_rec = TypeRelRec()
        # self.type_recognize = TypeRecognize()
        # self.relation_recognize = RelationRecognize()
        # self.type_position = PositionPredict(len(vocabs[TYPE_POSITION]), self.candidate_types)

    def forward(self, src_tokens, trg_tokens):
        encoder_out = self.encoder(src_tokens)
        ner_out, ner_h = self.ner(encoder_out)
        coref_out = self.coref(torch.cat([encoder_out, ner_h], dim=-1))
        decoder_out, decoder_h = self.decoder(
            src_tokens, trg_tokens, encoder_out)
        encoder_ctx = encoder_out[:, -1:, :]
        # type_out = self.type_recognize(encoder_ctx, self.candidate_types.type_tensor)
        # relation_out = self.relation_recognize(encoder_ctx, self.candidate_relations.relation_tensor)
        # undefine_tokens = torch.zeros([relation_out.shape[0], 2]).to(DEVICE)
        # classifier_out = torch.cat([type_out, relation_out], dim=-1)
        # classifier_out = torch.cat([undefine_tokens, classifier_out], dim=-1)
        classifier_out = self.type_rel_rec(encoder_ctx, self.candidates.can_tensor)

        graph_out = self.graph_net(encoder_ctx, classifier_out, decoder_h, self.graph)

        # type_pos_out = self.type_position(decoder_h, type_out)
        # print(type_out)
        # print(type_pos_out)
        # print('='*20)

        return {
            LOGICAL_FORM: decoder_out,
            NER: ner_out,
            COREF: coref_out,
            GRAPH: graph_out,
            TYPE_REL_RECOGNIZE: classifier_out
            # TYPE_RECOGNIZE: type_out,
            # RELATION_RECOGNIZE: relation_out
        }

    def _predict_encoder(self, src_tensor):
        with torch.no_grad():
            encoder_out = self.encoder(src_tensor)
            ner_out, ner_h = self.ner(encoder_out)
            coref_out = self.coref(torch.cat([encoder_out, ner_h], dim=-1))
            encoder_ctx = encoder_out[:, -1:, :]
            classifier_out = self.type_rel_rec(encoder_ctx, self.candidates.can_tensor)
            # type_out = self.type_recognize(encoder_ctx, self.candidate_types.type_tensor)
            # relation_out = self.relation_recognize(encoder_ctx, self.candidate_relations.relation_tensor)

        return {
            ENCODER_OUT: encoder_out,
            NER: ner_out,
            COREF: coref_out,
            TYPE_REL_RECOGNIZE: classifier_out
            # TYPE_RECOGNIZE: type_out,
            # RELATION_RECOGNIZE: relation_out
        }

    def _predict_decoder(self, src_tokens, trg_tokens, encoder_out):
        with torch.no_grad():
            decoder_out, decoder_h = self.decoder(
                src_tokens, trg_tokens, encoder_out)
            encoder_ctx = encoder_out[:, -1:, :]
            classifier_out = self.type_rel_rec(encoder_ctx, self.candidates.can_tensor)
            # graph_out = self.graph_net(encoder_ctx, decoder_h, self.graph)
            # type_out = self.type_recognize(encoder_ctx, self.candidate_types.type_tensor)
            # relation_out = self.relation_recognize(encoder_ctx, self.candidate_relations.relation_tensor)
            # undefine_tokens = torch.zeros([relation_out.shape[0], 2]).to(DEVICE)
            # classifier_out = torch.cat([type_out, relation_out], dim=-1)
            # classifier_out = torch.cat([undefine_tokens, classifier_out], dim=-1)
            graph_out = self.graph_net(encoder_ctx, classifier_out, decoder_h, self.graph)
            # type_pos_out = self.type_position(decoder_h, type_out)

            return {
                DECODER_OUT: decoder_out,
                GRAPH: graph_out,
            }


class LstmFlatten(nn.Module):
    def forward(self, x):
        return x[0].squeeze(1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(-1, x.shape[-1])


class NerNet(nn.Module):
    def __init__(self, tags, dropout=args.dropout):
        super(NerNet, self).__init__()
        self.ner_lstm = nn.Sequential(
            nn.LSTM(input_size=args.emb_dim,
                    hidden_size=args.emb_dim, batch_first=True),
            LstmFlatten(),
            nn.LeakyReLU()
        )
        self.ner_linear = nn.Sequential(
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(args.emb_dim, tags)
        )
        # self.crf = CRF(tags, batch_first=True)

    def forward(self, x):
        h = self.ner_lstm(x)
        y = self.ner_linear(h)
        return y, h

        # loss = self.crf(y, labels)
        # loss = -1 * loss
        # print(loss)
        # return loss, h


class CorefNet(nn.Module):
    def __init__(self, tags, dropout=args.dropout):
        super(CorefNet, self).__init__()
        self.seq_net = nn.Sequential(
            nn.Linear(args.emb_dim*2, args.emb_dim),
            nn.LeakyReLU(),
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(args.emb_dim, tags)
        )

    def forward(self, x):
        return self.seq_net(x)


class GraphNet(nn.Module):
    def __init__(self, num_nodes):
        super(GraphNet, self).__init__()
        self.gat = GATConv(args.bert_dim, args.emb_dim,
                           heads=args.graph_heads, dropout=args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.get_prob = nn.Linear((args.emb_dim*args.graph_heads)+args.emb_dim, args.emb_dim*args.graph_heads)
        self.linear_out = nn.Linear(
            (args.emb_dim*args.graph_heads)+args.emb_dim, args.emb_dim)
        self.score = nn.Linear(args.emb_dim, 1)
        self.context_net = nn.Sequential(
            nn.Linear(args.emb_dim*2, args.emb_dim),
            nn.LeakyReLU(),
            Flatten(),
            nn.Dropout(args.dropout),
            nn.Linear(args.emb_dim, num_nodes)
        )

    def forward(self, encoder_ctx, classifier_out, decoder_h, graph):
        g = self.gat(graph.x, graph.edge_index)
        g = self.dropout(g) # [1850, 600]
        classifier_out = classifier_out.unsqueeze(-1).repeat(1,1,args.emb_dim) # [50, 1850, 300]
        g = torch.cat([classifier_out, g.unsqueeze(0).repeat(classifier_out.shape[0],1,1)], dim=-1) # [batch_size, 1850, 900]
        g = self.get_prob(g) # [batch_size, 1850, 600]
        g = self.linear_out(torch.cat([encoder_ctx.repeat(1, graph.x.shape[0], 1), g], dim=-1))
        g = Flatten()(self.score(
            g).squeeze(-1).unsqueeze(1).repeat(1, decoder_h.shape[1], 1))
        x = self.context_net(
            torch.cat([encoder_ctx.expand(decoder_h.shape), decoder_h], dim=-1))
        # y = x*g
        # print(y.size())
        return x * g


class Encoder(nn.Module):
    def __init__(self, vocabulary, device, embed_dim=args.emb_dim, layers=args.layers,
                 heads=args.heads, pf_dim=args.pf_dim, dropout=args.dropout, max_positions=args.max_positions):
        super().__init__()
        input_dim = len(vocabulary)
        self.padding_idx = vocabulary.stoi[PAD_TOKEN]
        self.dropout = dropout
        self.device = device

        input_dim, embed_dim = vocabulary.vectors.size()
        self.scale = math.sqrt(embed_dim)
        self.embed_tokens = nn.Embedding(input_dim, embed_dim)
        self.embed_tokens.weight.data.copy_(vocabulary.vectors)
        self.embed_positions = PositionalEmbedding(
            embed_dim, dropout, max_positions)

        self.layers = nn.ModuleList(
            [EncoderLayer(embed_dim, heads, pf_dim, dropout, device) for _ in range(layers)])

    def forward(self, src_tokens):
        src_mask = (src_tokens != self.padding_idx).unsqueeze(1).unsqueeze(2)

        x = self.embed_tokens(src_tokens) * self.scale
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, pf_dim, dropout, device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadedAttention(
            embed_dim, heads, dropout, device)
        self.pos_ff = PositionwiseFeedforward(embed_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_tokens, src_mask):
        x = self.layer_norm(
            src_tokens + self.dropout(self.self_attn(src_tokens, src_tokens, src_tokens, src_mask)))
        x = self.layer_norm(x + self.dropout(self.pos_ff(x)))

        return x


class Decoder(nn.Module):
    def __init__(self, vocabulary, device, embed_dim=args.emb_dim, layers=args.layers,
                 heads=args.heads, pf_dim=args.pf_dim, dropout=args.dropout, max_positions=args.max_positions):
        super().__init__()

        output_dim = len(vocabulary)
        self.pad_id = vocabulary.stoi[PAD_TOKEN]
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.device = device
        self.max_positions = max_positions

        self.scale = math.sqrt(embed_dim)
        self.embed_tokens = nn.Embedding(output_dim, embed_dim)
        self.embed_positions = PositionalEmbedding(
            embed_dim, dropout, max_positions)

        self.layers = nn.ModuleList(
            [DecoderLayer(embed_dim, heads, pf_dim, dropout, device) for _ in range(layers)])

        self.linear_out = nn.Linear(embed_dim, output_dim)

    def make_masks(self, src_tokens, trg_tokens):
        src_mask = (src_tokens != self.pad_id).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg_tokens != self.pad_id).unsqueeze(1).unsqueeze(3)
        trg_len = trg_tokens.shape[1]
        trg_sub_mask = torch.tril(torch.ones(
            (trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return src_mask, trg_mask

    def forward(self, src_tokens, trg_tokens, encoder_out):
        src_mask, trg_mask = self.make_masks(src_tokens, trg_tokens)

        x = self.embed_tokens(trg_tokens) * self.scale
        x += self.embed_positions(trg_tokens)
        h = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            h = layer(h, encoder_out, trg_mask, src_mask)

        x = h.contiguous().view(-1, h.shape[-1])
        x = self.linear_out(x)

        return x, h


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadedAttention(
            embed_dim, heads, dropout, device)
        self.src_attn = MultiHeadedAttention(embed_dim, heads, dropout, device)
        self.pos_ff = PositionwiseFeedforward(embed_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embed_trg, embed_src, trg_mask, src_mask):
        x = self.layer_norm(
            embed_trg + self.dropout(self.self_attn(embed_trg, embed_trg, embed_trg, trg_mask)))
        x = self.layer_norm(
            x + self.dropout(self.src_attn(x, embed_src, embed_src, src_mask)))
        x = self.layer_norm(x + self.dropout(self.pos_ff(x)))

        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout, device):
        super().__init__()
        assert embed_dim % heads == 0
        self.attn_dim = embed_dim // heads
        self.heads = heads
        self.dropout = dropout

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.attn_dim])).to(device)

        self.linear_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)

        # (batch, heads, sent_len, attn_dim)
        Q = Q.view(batch_size, -1, self.heads,
                   self.attn_dim).permute(0, 2, 1, 3)
        # (batch, heads, sent_len, attn_dim)
        K = K.view(batch_size, -1, self.heads,
                   self.attn_dim).permute(0, 2, 1, 3)
        # (batch, heads, sent_len, attn_dim)
        V = V.view(batch_size, -1, self.heads,
                   self.attn_dim).permute(0, 2, 1, 3)

        # (batch, heads, sent_len, sent_len)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # (batch, heads, sent_len, sent_len)
        attention = F.softmax(energy, dim=-1)
        attention = F.dropout(attention, p=self.dropout,
                              training=self.training)

        x = torch.matmul(attention, V)  # (batch, heads, sent_len, attn_dim)
        # (batch, sent_len, heads, attn_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        # (batch, sent_len, embed_dim)
        x = x.view(batch_size, -1, self.heads * (self.attn_dim))
        x = self.linear_out(x)

        return x


class PositionwiseFeedforward(nn.Module):
    def __init__(self, embed_dim, pf_dim, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, pf_dim)
        self.linear_2 = nn.Linear(pf_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        x = torch.relu(self.linear_1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.linear_2(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        pos_embed = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2)
                             * -(math.log(10000.0) / d_model))
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer('pos_embed', pos_embed)

    def forward(self, x):
        return Variable(self.pos_embed[:, :x.size(1)], requires_grad=False)

class TypeRelRec(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(args.bert_dim, args.emb_dim)
    def forward(self, encoder_ctx, candidate):
        linear_out = self.linear(candidate).permute(1, 0)
        x = torch.matmul(encoder_ctx, linear_out)[:, -1:, :]
        y = torch.squeeze(x)
        return y

