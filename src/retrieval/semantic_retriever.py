import torch
import faiss
import numpy as np
import transformers

from torch  import nn
from torch  import Tensor as T
from typing import Tuple, List

from transformers import BertConfig
from transformers import BertModel
from transformers import BertTokenizer

from src.utils import read_file
from src.utils import read_json

from src.retrieval.abs_retriever import AbsRetriever

class HFBertEntityEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = nn.Linear(config.hidden_size * 2, project_dim) if project_dim != 0 else None
        self.ent_s_idx = 21128
        self.ent_e_idx = 21129
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, pretrained: bool = True, **kwargs
    ) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)
        else:
            return HFBertEntityEncoder(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:

        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # HF >4.0 version support
        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ):
            sequence_output = out.last_hidden_state
            pooled_output = None
            hidden_states = out.hidden_states

        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out
        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        if isinstance(representation_token_pos, int):
            ent_s_pos = torch.where(input_ids == self.ent_s_idx)
            ent_e_pos = torch.where(input_ids == self.ent_e_idx)

            ent_s_embeds = sequence_output[ent_s_pos]
            ent_e_embeds = sequence_output[ent_e_pos]

            ent_embeds = torch.cat([ent_s_embeds, ent_e_embeds], dim=-1)
            pooled_output = ent_embeds
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert representation_token_pos.size(0) == bsz, "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack([sequence_output[i, representation_token_pos[i, 1], :] for i in range(bsz)])

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    # TODO: make a super class for all encoders
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

class SemanticRetriever(AbsRetriever):
    def __init__(self, model_path, entity_path, entity_content_path, entity_vectors_path):
        # self.contexts = self._load_entity(entity_path)
        self.contents   = read_json(entity_content_path)
        self.contexts   = [data['entity'] for data in self.contents]
        self.vectors    = np.load(entity_vectors_path)
        self.rank_index = self._build_rank_index(list(range(len(self.contents))))

        # load dpr model
        tag = 'Amian/bert-base-chinese-entity'
        self.model = HFBertEntityEncoder.init_encoder(
            tag, projection_dim=768,
        )
        self.tokenizer = BertTokenizer.from_pretrained(tag)
        
        state_dict = torch.load(model_path, map_location='cpu')['model_dict']
        
        question_model_state = self._get_model(state_dict, 'question_model')
        # ctx_model_state      = self._get_model(state_dict, 'ctx_model')
        self.model.load_state_dict(question_model_state)

    def _load_entity(self, entity_path):
        contexts         = []
        contexts = read_file(entity_path)
        contexts = [[e[0], self.encode(e[0])] for e in contexts]
        return contexts

    def _build_rank_index(self, subset_index, d=768):
        rank_index = faiss.IndexFlatIP(d)
        for i in subset_index:
            embedding = self.vectors[i].reshape(1, -1)
            rank_index.add(embedding)
        return rank_index

    def _get_model(self, state_dict, keyword):
        model_state = {}
        for k in state_dict:
            if keyword in k:
                key = k.replace(f'{keyword}.', '')
                model_state[key] = state_dict[k]
        return model_state

    def encode(self, text, position):
        start, end = position
        masked = text[:start] + '<E_START>' + '[MASK]' * (end - start) + '<E_END>' + text[end:]
        inputs = self.tokenizer(masked, return_tensors="pt", padding=True)
        return inputs

    def similarity(self, query, topk):
        sequence_output, pooled_output, hidden_states = self.model(**query)
        embedding = pooled_output.detach().numpy()
        D, I = self.rank_index.search(embedding, topk)
        # normalize
        D = D - np.min(D)
        D = D / np.max(D)
        score = []
        # return score
        for i in range(D.shape[1]):
            entity = self.contexts[I[0][i]]
            score.append([D[0][i], entity])
        return score

    def retrieve_one_step(self, text: str, span: List[str], topk: int=10) -> List[str]:
        entity, type, position = span
        query  = self.encode(text, position)
        result = self.similarity(query, topk)
        return sorted(result, reverse=True)
        
    def retrieve(self, texts: List[str], spans: List[str], topk: int=10) -> List[str]:
        results = []
        for text, span in zip(texts, spans):
            result = self.retrieve_one_step(text, span, topk)
            results.append(result)
        return results

if __name__ == '__main__':
    model_path  = "/share/nas165/amian/experiments/nlp/DPR/outputs/2023-03-07/16-45-46/output/dpr_biencoder.39"
    entity_path = ""
    entity_content_path = "/share/nas165/amian/experiments/speech/AISHELL-NER/dump/2023_02_27__15_58_45_test/aishell_ner_ctx.json"
    entity_vectors_path = "/share/nas165/amian/experiments/speech/AISHELL-NER/dump/2023_15_03__14_36_47/embeds.npy"

    retriever = SemanticRetriever(model_path, entity_path, entity_content_path, entity_vectors_path)

    text = "许玮拎日前传闻阮經天八年情变"
    span = [["许玮拎", "", [0, 3]], ["阮經天", "", [7, 10]]]

    results = retriever.retrieve([text for _ in range(len(span))], span)
    print(results)