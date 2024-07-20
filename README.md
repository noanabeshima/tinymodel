# [TinyModel](https://github.com/noanabeshima/tiny_model)

TinyModel is a 4 layer, 44M parameter model trained on [TinyStories V2](https://arxiv.org/abs/2305.07759) for mechanistic interpretability. It uses ReLU activations and no layernorms. It comes with trained SAEs and transcoders.

It can be installed with `pip install tinymodel` for Python 3.11 and higher.

This library is in an alpha state, it probably has some bugs. Please let me know if you find any or you're having any trouble with the library, I can be emailed at my full name @ gmail.com or messaged on twitter. You can also add GitHub issues.


```
from tiny_model import TinyModel, tokenizer

lm = TinyModel()

# for inference
tok_ids, attn_mask = tokenizer(['Once upon a time', 'In the forest'])
logprobs = lm(tok_ids)

# Get SAE/transcoder acts
# See 'SAEs/Transcoders' section for more information.
feature_acts = lm['M1N123'](tok_ids)
all_feat_acts = lm['M2'](tok_ids)

# Generation
lm.generate('Once upon a time, Ada was happily walking through a magical forest with')

# To decode tok_ids you can use
tokenizer.decode(tok_ids)
```

It was trained for 3 epochs on a [preprocessed version of TinyStoriesV2](https://huggingface.co/datasets/noanabeshima/TinyStoriesV2). Pre-tokenized dataset [here](https://huggingface.co/datasets/noanabeshima/TinyModelTokIds). I recommend using this dataset for getting SAE/transcoder activations.



# SAEs/transcoders
Some sparse SAEs/transcoders are provided along with the model.

For example, `acts = lm['M2N100'](tok_ids)`

To get sparse acts, choose which part of the transformer block you want to look at (currently [sparse MLP](https://www.lesswrong.com/posts/MXabwqMwo3rkGqEW8/sparse-mlp-distillation)/[transcoder](https://www.alignmentforum.org/posts/YmkjnWtZGLbHRbzrP/transcoders-enable-fine-grained-interpretable-circuit) and SAEs on attention out are available, under the tags `'M'` and `'A'` respectively). Residual stream and MLP out SAEs exist, they just haven't been added yet, bug me on e.g. Twitter if you want this to happen fast.

Then, add the layer. A sparse MLP at layer 2 would be `'M2'`.
Finally, optionally add a particular neuron. For example `'M0N10000'`.

# Tokenization
Tokenization is done as follows:
- the top-10K most frequent tokens using the GPT-NeoX tokenizer are selected and sorted by frequency.
- To tokenize a document, first tokenize with the GPT-NeoX tokenizer. Then replace tokens not in the top 10K tokens with a special \[UNK\] token id. All token ids are then mapped to be between 1 and 10K, roughly sorted from most frequent to least.
- Finally, prepend the document with a [BEGIN] token id.

