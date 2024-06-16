
TinyModel is a 44M parameter model trained on TinyStories for mechanistic interpretability.

It has 4 layers, uses ReLU activations, and has no layernorm.

It was trained for 3 epochs on a [preprocessed version](https://huggingface.co/datasets/noanabeshima/TinyStoriesV2) of [TinyStoriesV2](https://huggingface.co/datasets/roneneldan/TinyStories).


```
from tiny_model import TinyModel, tokenizer

lm = TinyModel()

# for inference
tok_ids, attn_mask = tokenizer(['Once upon a time', 'In the forest'])
logprobs = lm(tok_ids)

# or
lm.generate('Once upon a time, Ada was happily walking through a magical forest with')

# To decode tok_ids you can use
tokenizer.decode(tok_ids)
```

Tokenization is done as follows:
- the top-10K most frequent tokens using the GPT-NeoX tokenizer are selected and sorted by frequency.
- To tokenize a document, first tokenize with the GPT-NeoX tokenizer. Then replace tokens not in the top 10K tokens with a special \[UNK\] token id. All token ids are then mapped to be between 1 and 10K, roughly sorted from most frequent to least.
- Finally, prepend the document with a [BEGIN] token id.

