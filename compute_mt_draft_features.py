#!/usr/bin/env python3
"""
Compute surprisal and attention features on MT draft (GPT-4o output) text
for PE conditions in the Yiyu study.

Output: attention-norm-yiyu-positive-dur/mt_draft_features.tsv
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from attention_util import src_seq_att, tgt_seq_att

lang_code = {"en": "eng_Latn", "zh": "zho_Hans"}



def get_rel_ids(tokenized_ids, target_ids, special_token_offset):
    """Convert word-level token positions to subword-level positions.

    tokenized_ids: list of 1-d tensors, each holding subword ids for one word
    target_ids: list of word-level indices
    special_token_offset: 1 for NLLB (lang tag), 0 for mGPT (no leading token)
    """
    new_target_ids = []
    for tid in target_ids:
        prefix_len = sum(tokids.size(0) for tokids in tokenized_ids[:tid])
        for i in range(len(tokenized_ids[tid])):
            new_target_ids.append(prefix_len + i + special_token_offset)
    return new_target_ids


def tokenize_nllb(tokenizer, words, lang):
    """Tokenize words for NLLB as [lang_tag] + subwords + [eos].

    Returns (full_ids, rel_ids) where rel_ids are subword-level positions
    for each word, offset by +1 for the lang tag.
    """
    subword_ids = [
        tokenizer(w, return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"].view(-1)
        for w in words
    ]
    lang_id = torch.tensor([tokenizer.convert_tokens_to_ids(lang_code[lang])])
    eos_id = torch.tensor([tokenizer.eos_token_id])
    full_ids = torch.cat([lang_id] + subword_ids + [eos_id]).view(-1).to(dtype=torch.long)
    rel_ids = get_rel_ids(subword_ids, list(range(len(words))), 1)
    return full_ids, rel_ids


def tokenize_mgpt(tokenizer, words):
    """Tokenize words for mGPT as bare subwords with no special tokens.

    Returns (full_ids, rel_ids) where rel_ids are subword-level positions
    with offset 0.
    """
    subword_ids = [
        tokenizer(w, return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"].view(-1)
        for w in words
    ]
    full_ids = torch.cat(subword_ids).view(-1).to(dtype=torch.long)
    rel_ids = get_rel_ids(subword_ids, list(range(len(words))), 0)
    return full_ids, rel_ids


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draft_json", help="Private tokenized MT draft JSON")
    with open(parser.parse_args().draft_json, encoding="utf-8") as stream:
        MT_DRAFT = json.load(stream)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    nllb_checkpoint = "facebook/nllb-200-distilled-600M"
    print(f"Loading NLLB: {nllb_checkpoint}", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(nllb_checkpoint, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(nllb_checkpoint)

    lm_checkpoint = "sberbank-ai/mGPT"
    print(f"Loading mGPT: {lm_checkpoint}", flush=True)
    tokenizer_lm = GPT2Tokenizer.from_pretrained(lm_checkpoint)
    model_lm = GPT2LMHeadModel.from_pretrained(lm_checkpoint)

    model = model.to(device)
    model_lm = model_lm.to(device)
    model.eval()
    model_lm.eval()

    xentropy = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")

    outdir = "attention-norm-yiyu-positive-dur"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "mt_draft_features.tsv")

    header = [
        "condition", "tgt_segid",
        "mt_draft_tgt_surp", "mt_draft_xent", "mt_draft_xmi",
        "mt_draft_eos_xatt", "mt_draft_y2srcseq_ent", "mt_draft_y2y",
        "mt_draft_y2cont", "mt_draft_y_ent",
        "mt_draft_x2cont", "mt_draft_cont2x", "mt_draft_x_ent",
        "mt_draft_x2x", "mt_draft_x2eos", "mt_draft_tgtseq2x",
    ]

    with open(outpath, "w") as outf:
        outf.write("\t".join(header) + "\n")

        with torch.no_grad():
            for condition, data in MT_DRAFT.items():
                src_lang = data["src_lang"]
                tgt_lang = data["tgt_lang"]

                for seg_idx, (src_words, tgt_words) in enumerate(
                    zip(data["source_segments"], data["target_segments"])
                ):
                    src_tok, src_pos = tokenize_nllb(tokenizer, src_words, src_lang)
                    tgt_tok, tgt_pos = tokenize_nllb(tokenizer, tgt_words, tgt_lang)

                    src_tok_lm, src_pos_lm = tokenize_mgpt(tokenizer_lm, src_words)
                    tgt_tok_lm, tgt_pos_lm = tokenize_mgpt(tokenizer_lm, tgt_words)

                    src_tok = src_tok.unsqueeze(0).to(device)
                    tgt_tok = tgt_tok.unsqueeze(0).to(device)
                    src_tok_lm = src_tok_lm.unsqueeze(0).to(device)
                    tgt_tok_lm = tgt_tok_lm.unsqueeze(0).to(device)
                    src_mask = torch.ones_like(src_tok).to(device)

                    output = model(
                        input_ids=src_tok,
                        attention_mask=src_mask,
                        labels=tgt_tok,
                    )
                    logits = output.logits

                    enc_att = torch.cat(
                        [layer.unsqueeze(0) for layer in output.encoder_attentions], dim=0
                    )
                    enc_att = enc_att.permute(1, 0, 2, 3, 4).squeeze(0)

                    dec_att = torch.cat(
                        [layer.unsqueeze(0) for layer in output.decoder_attentions], dim=0
                    )
                    dec_att = dec_att.permute(1, 0, 2, 3, 4).squeeze(0)

                    x_att = torch.cat(
                        [layer.unsqueeze(0) for layer in output.cross_attentions], dim=0
                    )
                    x_att = x_att.permute(1, 0, 2, 3, 4).squeeze(0)

                    src_output_lm = model_lm(input_ids=src_tok_lm)
                    tgt_output_lm = model_lm(input_ids=tgt_tok_lm)
                    tgt_oologits = tgt_output_lm.logits

                    src_pos_t = torch.tensor(src_pos, dtype=torch.long, device=device)
                    tgt_pos_t = torch.tensor(tgt_pos, dtype=torch.long, device=device)
                    tgt_pos_lm_t = torch.tensor(tgt_pos_lm, dtype=torch.long, device=device)

                    logit_s = logits.squeeze(0)
                    tgt_tids_s = tgt_tok.squeeze(0)
                    xent = xentropy(logit_s[tgt_pos_t], tgt_tids_s[tgt_pos_t])

                    tgt_oologit_s = tgt_oologits.squeeze(0)
                    tgt_tids_lm_s = tgt_tok_lm.squeeze(0)
                    tgt_surp = xentropy(tgt_oologit_s[tgt_pos_lm_t], tgt_tids_lm_s[tgt_pos_lm_t])

                    nllb_tgt_len = tgt_pos_t.size(0)
                    lm_tgt_len = tgt_pos_lm_t.size(0)

                    xent_val = (xent / nllb_tgt_len).item()
                    tgt_surp_val = (tgt_surp / lm_tgt_len).item()
                    xmi_val = tgt_surp_val - xent_val

                    x2cont, cont2x, x_ent, x2x, x2eos, tgtseq2x = src_seq_att(
                        enc_att, x_att, src_pos_t
                    )
                    y2srcseq_ent, y2eos, y2cont, y2y, y_ent = tgt_seq_att(
                        x_att, dec_att, tgt_pos_t
                    )

                    row = [
                        condition,
                        str(seg_idx + 1),
                        str(tgt_surp_val),
                        str(xent_val),
                        str(xmi_val),
                        str(y2eos.item()),
                        str(y2srcseq_ent.item()),
                        str(y2y.item()),
                        str(y2cont.item()),
                        str(y_ent.item()),
                        str(x2cont.item()),
                        str(cont2x.item()),
                        str(x_ent.item()),
                        str(x2x.item()),
                        str(x2eos.item()),
                        str(tgtseq2x.item()),
                    ]
                    outf.write("\t".join(row) + "\n")
                    outf.flush()

                    print(
                        f"  {condition} tgt_segid={seg_idx+1}: "
                        f"tgt_surp={tgt_surp_val:.4f} xent={xent_val:.4f} xmi={xmi_val:.4f}",
                        flush=True,
                    )

    print(f"\nOutput written to {outpath}", flush=True)


if __name__ == "__main__":
    main()
