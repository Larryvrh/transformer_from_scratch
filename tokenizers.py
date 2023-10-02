from typing import *
import re


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


# class CharTokenizer:
#     def __init__(self, corpus: str, vocab_size: Optional[int] = None, vocab_coverage: Optional[float] = None,
#                  reserved_vocab: Optional[List[str]] = None, unk_literal: str = '<unk>'):
#         if reserved_vocab is not None:
#             assert len(reserved_vocab) == len(set(reserved_vocab)), 'no duplicate is allowed in reserved vocab'
#             assert unk_literal not in reserved_vocab, f'unk literal "{unk_literal}" cannot be in reserved vocab'
#         else:
#             reserved_vocab = []
#         vocab = reserved_vocab.copy() if reserved_vocab is not None else []
#         vocab += sample_vocab(corpus, vocab_size - len(vocab) - 1, vocab_coverage)
#         self.s2i = {s: i + 1 for i, s in enumerate(vocab)}
#         self.s2i[unk_literal] = 0
#         self.i2s = {i: s for s, i in self.s2i.items()}
#         self.special_vocab = set(reserved_vocab + [unk_literal])
#         self.unk_literal = unk_literal
#
#     def encode(self, text: str) -> List[int]:
#         cursor, ids = 0, []
#         while cursor < len(text):
#             for s in self.special_vocab:
#                 if text[cursor:].startswith(s):
#                     ids.append(self.s2i[s])
#                     cursor += len(s)
#                     break
#             else:
#                 ids.append(self.s2i.get(text[cursor], self.s2i.get(self.unk_literal)))
#                 cursor += 1
#         return ids
#
#     def decode(self, ids: List[int]) -> str:
#         return ''.join(self.i2s[i] for i in ids)
#
#     def get_vocab_mapping(self):
#         return self.s2i


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
