import json
from typing import *
import re
from math import ceil

standalone_vocab = '~!@#$%^&*()-=[]{}:;"\'|\\/<>,.? \n\r\t'

with open('llama_vocab_original.json', 'r') as file:
    vocab: Dict[str, int] = json.load(file)

# prune reserved vocabs and leading space vocabs
vocab = {s: i for s, i in vocab.items() if i > 258 and not s.startswith('▁')}
# prune standalone vocabs
vocab = {s: i for s, i in vocab.items() if not any(j in s and j != s for j in standalone_vocab)}
# prune non-alphabet vocabs
vocab = {s: i for s, i in vocab.items() if re.fullmatch('[a-zA-Z]+', s)}
# prune single ascii vocabs
vocab = {s: i for s, i in vocab.items() if len(s.encode('utf-8')) > 1}
# prune pure upper case vocabs
vocab = {s: i for s, i in vocab.items() if not s.isupper()}

llama_vocab = [v.encode('utf-8') for v in vocab.keys()]
llama_vocab.sort()

bytes_vocab = [i.to_bytes(1, 'little') for i in range(256)]

reserved_vocab = ['<s>', '</s>', '<pad>']

final_vocab = bytes_vocab + reserved_vocab + llama_vocab

vocab_padding = (ceil(len(final_vocab) / 512) * 512) - len(final_vocab)

final_vocab += [f'<reserved_{i}>' for i in range(vocab_padding)]

final_vocab = [v if isinstance(v, bytes) else v.encode('utf-8') for v in final_vocab]

final_vocab = [{'id': i, 'bytes': list(v), 'repr': str(v)[2:-1]} for i, v in enumerate(final_vocab)]
with open('llama_vocab_pruned.json', 'w') as file:
    json.dump(final_vocab, file, indent=2, ensure_ascii=False)