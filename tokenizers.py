import time
from typing import *
import re
import json
import numpy as np


def sample_vocab(tokens: Iterable[str], vocab_size: Optional[int] = None,
                 vocab_coverage: Optional[float] = None) -> List[str]:
    assert (vocab_size is not None and vocab_coverage is None) or \
           (vocab_size is None and vocab_coverage is not None), "vocab_size [or] vocab_coverage need specified"

    token_count = {}
    for c in tokens:
        token_count[c] = token_count.get(c, 0) + 1

    if vocab_size is not None:
        token_count = list(token_count.items())
        token_count.sort(key=lambda i: i[1], reverse=True)
        vocab = [c[0] for c in token_count[:vocab_size]]
    else:
        total_count = sum(token_count.values())
        token_freq = [(c, i / total_count) for c, i in token_count.items()]
        token_freq.sort(key=lambda i: i[1], reverse=True)
        freq_sum = 0.0
        split = 0
        for split in range(len(token_freq)):
            freq_sum += token_freq[split][1]
            if freq_sum >= vocab_coverage:
                break
        vocab = [c[0] for c in token_freq[:split + 1]]
    return vocab


class CharTokenizer:
    def __init__(self, corpus: str, vocab_size: Optional[int] = None, vocab_coverage: Optional[float] = None,
                 reserved_vocab: Optional[List[str]] = None, unk_literal: str = '<unk>'):
        if reserved_vocab is not None:
            assert len(reserved_vocab) == len(set(reserved_vocab)), 'no duplicate is allowed in reserved vocab'
            assert unk_literal not in reserved_vocab, f'unk literal "{unk_literal}" cannot be in reserved vocab'
        else:
            reserved_vocab = []
        vocab = reserved_vocab.copy() if reserved_vocab is not None else []
        vocab += sample_vocab(corpus, vocab_size - len(vocab) - 1, vocab_coverage)
        self.s2i = {s: i + 1 for i, s in enumerate(vocab)}
        self.s2i[unk_literal] = 0
        self.i2s = {i: s for s, i in self.s2i.items()}
        self.special_vocab = set(reserved_vocab + [unk_literal])
        self.unk_literal = unk_literal

    def encode(self, text: str) -> List[int]:
        cursor, ids = 0, []
        while cursor < len(text):
            for s in self.special_vocab:
                if text[cursor:].startswith(s):
                    ids.append(self.s2i[s])
                    cursor += len(s)
                    break
            else:
                ids.append(self.s2i.get(text[cursor], self.s2i.get(self.unk_literal)))
                cursor += 1
        return ids

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.i2s[i] for i in ids)

    def get_vocab_mapping(self):
        return self.s2i


class WordTokenizer:
    def __init__(self, corpus: str, vocab_size: Optional[int] = None, vocab_coverage: Optional[float] = None,
                 reserved_vocab: Optional[List[str]] = None, unk_literal: str = '<unk>'):
        if reserved_vocab is not None:
            assert len(reserved_vocab) == len(set(reserved_vocab)), 'no duplicate is allowed in reserved vocab'
            assert unk_literal not in reserved_vocab, f'unk literal "{unk_literal}" cannot be in reserved vocab'
        else:
            reserved_vocab = []
        vocab = reserved_vocab.copy() if reserved_vocab is not None else []

        tokens = (c[0] if c[0] != '' else c[1] for c in re.finditer(r'(\w+)|(\W)', corpus))
        vocab += sample_vocab(tokens, vocab_size - len(vocab) - 1, vocab_coverage)

        self.s2i = {s: i + 1 for i, s in enumerate(vocab)}
        self.s2i[unk_literal] = 0
        self.i2s = {i: s for s, i in self.s2i.items()}
        self.special_vocab = set(reserved_vocab + [unk_literal])
        self.unk_literal = unk_literal

    def encode(self, text: str) -> List[int]:
        specials = '|'.join(f'{i}' for i in self.special_vocab)
        tokens = (c[0] if c[0] != '' else c[1] for c in re.finditer(rf'({specials}|\w+)|(\W)', text))
        return [self.s2i.get(t, self.s2i[self.unk_literal]) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.i2s[i] for i in ids)

    def get_vocab_mapping(self):
        return self.s2i

    def get_vocab_size(self):
        return len(self.s2i)

    def eval_vocab_coverage(self, corpus: str):
        encoded = self.encode(corpus)
        return 1 - (len([i for i in encoded if i == 0]) / len(encoded))


class TRIETokenizer:
    @staticmethod
    def split_bytes(data: bytes):
        return [b'%c' % i for i in data]

    def __init__(self, vocab_file: str):
        self.nodes = [[b'', -1, -1, {}, 0]]  # node value, parent index, token id, children, child_max_len
        with open(vocab_file, 'r') as file:
            vocabs = json.load(file)
        vocabs.sort(key=lambda i: len(i['bytes']))
        for entry in vocabs:
            self.add_vocab(bytes(entry['bytes']), entry['id'])

        self.id_to_bytes = {i['id']: i['bytes'] for i in vocabs}

    def add_vocab(self, vocab_bytes: bytes, vocab_id: int):
        full_vocab_bytes = vocab_bytes
        cur_node, cur_node_idx = self.nodes[0], 0
        while len(vocab_bytes) > 0:
            max_match = 0
            for i in range(1, cur_node[4] + 1):
                if vocab_bytes[:i] in cur_node[3]:
                    max_match = i
            if max_match > 0:
                cur_node_idx = cur_node[3][vocab_bytes[:max_match]]
                cur_node = self.nodes[cur_node_idx]
                vocab_bytes = vocab_bytes[max_match:]
            else:
                child_idx = len(self.nodes)
                self.nodes.append([full_vocab_bytes, cur_node_idx, vocab_id, {}, 0])
                cur_node[3][vocab_bytes] = child_idx
                cur_node[4] = max(cur_node[4], len(vocab_bytes))
                break

    def attempt_match(self, match_bytes: bytes):
        start_length = len(match_bytes)

        cur_node = self.nodes[0]
        while cur_node is not None:
            match_length, max_length = 0, min(cur_node[4], len(match_bytes))
            for i in range(1, max_length + 1):
                if match_bytes[:i] in cur_node[3]:
                    match_length = i
            if match_length == 0:
                return start_length - len(match_bytes), cur_node[2]
            else:
                cur_node = self.nodes[cur_node[3][match_bytes[:match_length]]]
                match_bytes = match_bytes[match_length:]
        return -1, -1

    def encode(self, text: str):
        text_bytes = text.encode('utf-8')
        tokens, length = [], 0
        while length < len(text_bytes):
            offset, token_id = self.attempt_match(text_bytes[length:])
            assert offset >= 0
            tokens.append(token_id)
            length += offset
        return tokens

    def decode(self, token_ids: List[int]):
        return bytes([t for i in token_ids for t in self.id_to_bytes[i]]).decode('utf-8')


tokenizer = TRIETokenizer('llama_vocab_pruned.json')
print(len(tokenizer.nodes))
with open('corpus/TinyStoriesV2-GPT4-valid.txt', 'r') as file:
    s = time.time()
    text = file.read()[:100000]
    encoded = tokenizer.encode(text)
    print(len(encoded))
    e = time.time()
    print(f'{100000 / (e - s):.2f} cps')
