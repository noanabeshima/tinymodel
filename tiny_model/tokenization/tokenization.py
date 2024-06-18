from transformers import AutoTokenizer
from unidecode import unidecode
import torch
import os
import numpy as np
import re
from typing import Union, List
from tqdm import tqdm


current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
neo_tokenizer = AutoTokenizer.from_pretrained(
    "roneneldan/TinyStories",
    padding=True,
    truncation=True,
    add_special_tokens=True,
    max_length=2048,
)
neo_tokenizer.model_max_length = 2048
neo_tokenizer.add_special_tokens(
    {
        "bos_token": "[BEGIN]",
        "eos_token": "[END]",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
    },
)
neo_tok_ids_to_ts = torch.load(f"{current_dir}/neo_tok_ids_to_ts.pt")
ts_tok_ids_to_neo = torch.load(f"{current_dir}/ts_tok_ids_to_neo.pt")


def clean_text(text):
    # Convert from unicode to ascii to make tokenization better; don't split up quotation marks into multiple tokens e.g.
    text = unidecode(text)

    # Replace multiple newlines with single newline
    text = re.sub(r'\n\n+', '\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    return text

def clean_story(story):
    # Convert from unicode to ascii to make tokenization better; don't split up quotation marks into multiple tokens e.g.
    story = unidecode(story)

    # lots of spaces at beginning of documents
    story = story.strip()

    # lots of trailing spaces at the end of lines
    story_lines = story.split("\n")
    story_lines = [line.strip() for line in story_lines]
    story = "\n".join(story_lines)

    # Double newline is rare in train data while single newline is common.
    while "\n\n" in story:
        story = story.replace("\n\n", "\n")

    # Double spaces are rare
    while "  " in story:
        story = story.replace("  ", " ")

    return story


def enc(stories, padding=True, return_attn_mask=False, max_length=256, add_begin=False):
    if add_begin is True and isinstance(max_length, int) and max_length > 0:
        max_length = max_length - 1
    if isinstance(stories, str):
        stories = [stories]
    stories = [
        story
        for story in stories
        if ("â" not in story)
        and ("€" not in story)
        and ("»" not in story)
        and ("«" not in story)
    ]
    stories = [clean_text(story) for story in stories]

    # Start with the TinyStories tokenizer, the GPTNeo tokenizer.
    out = neo_tokenizer(
        stories,
        max_length=max_length,
        return_tensors="pt",
        padding=padding,
        truncation=True,
    )
    input_ids, attn_mask = out["input_ids"], out["attention_mask"]

    # Replace tokens not in the top-10k most frequent tokens in the train dataset with the [UNK] special token.
    # Something like 1~6% of documents have at least one [UNK] token
    # All non-[UNK] tokens appear at least 100 times in the train dataset
    # I think that in the original TinyStories dataset these were just dropped instead of being replaced with an [UNK] token.
    unk_mask = ~torch.isin(input_ids, ts_tok_ids_to_neo) * attn_mask.bool()
    input_ids[unk_mask] = neo_tokenizer.unk_token_id

    # Replace the first [PAD] token with an [END] token
    eos_idx = attn_mask.argmin(dim=1)
    for i, eos_i in enumerate(eos_idx):
        if eos_i != 0:
            input_ids[i, eos_i] = neo_tokenizer.eos_token_id

    # Add a [BEGIN] token to the beginning of every story. The model is trained with this.
    if add_begin is True:
        input_ids = torch.cat(
            (torch.full((input_ids.shape[0], 1), neo_tokenizer.bos_token_id), input_ids),
            dim=1,
        )
        attn_mask = torch.cat(
            (torch.ones(attn_mask.shape[0], 1, dtype=torch.int), attn_mask), dim=1
        )

    # Convert from GPTNeo tok ids to custom tinystories tok ids, the most common Neo tok ids and some special tokens
    input_ids = neo_tok_ids_to_ts[input_ids]

    if return_attn_mask:
        return input_ids, attn_mask
    else:
        return input_ids


def dec(ts_tok_ids):
    if type(ts_tok_ids) in {torch.Tensor, np.ndarray} and np.prod(ts_tok_ids.shape) == 1:
        ts_tok_ids = int(ts_tok_ids.item())
    if isinstance(ts_tok_ids, int):
        ts_tok_ids = [ts_tok_ids]
    if isinstance(ts_tok_ids, list):
        ts_tok_ids = torch.tensor(ts_tok_ids)
    ts_tok_ids = ts_tok_ids.cpu()
    if not isinstance(ts_tok_ids, torch.Tensor):
        ts_tok_ids = torch.tensor(ts_tok_ids, dtype=torch.int32)
    neo_tok_ids = ts_tok_ids_to_neo[ts_tok_ids]
    if len(neo_tok_ids.shape) == 1:
        return neo_tokenizer.decode(neo_tok_ids)
    else:
        return neo_tokenizer.batch_decode(neo_tok_ids)



def tok_see(tok_ids: Union[str, torch.Tensor, list[int]], printout=True, symbolic_spaces=True, symbolic_newlines=True):
    if isinstance(tok_ids, str):
        tok_ids = enc(tok_ids, add_begin=False, max_length=2048)
    if isinstance(tok_ids, np.ndarray):
        tok_ids = torch.tensor(tok_ids)
    if isinstance(tok_ids, torch.Tensor):
        tok_ids = tok_ids.squeeze()
        if len(tok_ids.shape) == 0:
            tok_ids = tok_ids[None]
        assert len(tok_ids.shape) == 1, tok_ids.shape
    # toks = [dec(tok_id).replace(' ', '⋅').replace('\n', '↵') for tok_id in tok_ids]
    toks = [dec(tok_id) for tok_id in tok_ids]
    if symbolic_newlines:
        toks = [tok.replace('\n', '↵') for tok in toks]
    if symbolic_spaces:
        toks = [tok.replace(' ', '⋅') for tok in toks]
    if printout:
        print(toks)
    return toks


def get_tok_strs(tok_ids=None, symbolic_spaces=False, symbolic_newlines=False):
    if tok_ids is None:
        return get_tok_strs(torch.arange(0, 10_000)[None]).flatten()
    # tok_ids is a tok_id tensor
    res = []
    for doc in tqdm(tok_ids):
        doc_tok_strs = tok_see(doc, printout=False, symbolic_spaces=symbolic_spaces, symbolic_newlines=symbolic_newlines)
        res.append(doc_tok_strs)

    res = np.array(res)
    
    return res


class Tokenizer:
    def __init__(self):
        pass

    def encode(self, s: str):
        assert isinstance(s, str)
        
        return enc(s, add_begin=False)[0].tolist()

    def decode(self, tok_ids: Union[list, torch.Tensor, int]):
        if isinstance(tok_ids, int):
            tok_ids = [tok_ids]
        assert isinstance(tok_ids, list) or isinstance(tok_ids, torch.Tensor)
        
        return dec(tok_ids)

    def __call__(self, docs: List[str], padding=True, return_attn_mask=True, max_length=256, add_begin=True):
        return enc(docs, padding=padding, return_attn_mask=return_attn_mask, max_length=max_length, add_begin=add_begin)


tokenizer = Tokenizer()