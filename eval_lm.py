import json
import os.path
from abc import ABC

from lm_eval.base import BaseLM
from lm_eval import evaluator, tasks
from modeling import ToyTransformer, global_config, AttentionBackend
from tokenization import TRIETokenizer
import torch
from typing import *
import bisect


# noinspection DuplicatedCode
def generate(model, tokenizer, prompt, temperature, top_p, rep_penalty,
             max_new_tokens=20, total_tokens=None,
             end_tokens=None,
             enable_kv_cache=True):
    model.eval()

    feed_tokens = tokenizer.encode(prompt) if isinstance(prompt, str) else prompt

    all_tokens = feed_tokens.copy()
    if total_tokens is not None:
        max_new_tokens = max(0, total_tokens - len(feed_tokens))

    with torch.no_grad():
        kv_cache = None
        for _ in range(max_new_tokens):
            logits, kv_cache = model.forward(
                torch.tensor([feed_tokens if enable_kv_cache else all_tokens]).to(model.device),
                kv_cache=kv_cache)
            logits = logits[0][-1].cpu()
            if not enable_kv_cache:
                kv_cache = None

            # apply repetition penalty
            logits_rep = torch.gather(logits, 0, torch.tensor(all_tokens))
            logits_rep = torch.where(logits_rep < 0, logits_rep * rep_penalty, logits_rep / rep_penalty)
            logits.scatter_(0, torch.tensor(all_tokens), logits_rep)

            if top_p > 0.0:
                # apply temperature
                logits /= max(temperature, 1e-6)

                probs = torch.softmax(logits, dim=0)

                # apply top-p
                ordered_probs, ordered_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(ordered_probs, dim=0).tolist()
                top_p_index = bisect.bisect_right(cum_probs, top_p) + 1
                ordered_probs, ordered_indices = ordered_probs[:top_p_index], ordered_indices[:top_p_index]
                sampled_index = ordered_indices[torch.multinomial(ordered_probs, num_samples=1).item()].item()
            else:
                sampled_index = torch.argmax(logits).item()

            all_tokens.append(sampled_index)
            feed_tokens = [sampled_index]

            if end_tokens is not None and sampled_index in end_tokens:
                break

    return all_tokens


class ToyTransformerEvalAdapter(BaseLM):
    def __init__(self, model: ToyTransformer, tokenizer: TRIETokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        global_config['attn_backend'] = AttentionBackend.Naive

    @property
    def eot_token_id(self):
        return self.tokenizer.encode('</s>')

    @property
    def max_length(self):
        return 1024

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 4

    @property
    def device(self):
        return self.model.device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]):
        return self.tokenizer.decode(tokens)

    def _model_generate(self, context, max_length, eos_token_id):
        assert context.shape[0] == 1
        context = context[0].tolist()
        tokens = generate(self.model, self.tokenizer, context, 1.0, 0.00, 1.0, total_tokens=max_length, end_tokens=[eos_token_id])
        return torch.tensor([tokens])

    def _model_call(self, inputs):
        return self.model.forward(inputs)[0]


def evaluate(model_path, eval_session_name, eval_task_list):
    os.makedirs(f'eval_results/{eval_session_name}', exist_ok=True)

    device = torch.device('cuda')
    dtype = torch.float32

    tokenizer = TRIETokenizer('llama_vocab_pruned_32k.json')
    model = ToyTransformer(tokenizer.get_vocab_size(), 12, 12, 768, 1024, device, dtype)
    load_result = model.load_state_dict(torch.load(model_path))
    print(load_result)
    adapter = ToyTransformerEvalAdapter(model, tokenizer)

    for eval_name, eval_fewshot in eval_task_list:
        if os.path.exists(f'eval_results/{eval_session_name}/{eval_name}.json'):
            print('Skip', eval_name, 'since it\'s already done')
            continue
        else:
            print(f'Running {eval_name}...')
        results = evaluator.evaluate(adapter, tasks.get_task_dict([eval_name]), False, eval_fewshot)

        with open(f'eval_results/{eval_session_name}/{eval_name}.json', 'w') as file:
            json.dump(results, file, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    evaluate('checkpoints/train-round3-135M/checkpoint-done/model.pt', 'pt_final', eval_task_list=[
        ('arc_challenge', 25),
        ('hellaswag', 10),
        ('truthfulqa_mc', 0),
        ('winogrande', 5),
        # ('gsm8k', 5),
        # ('drop', 3),
    ])
