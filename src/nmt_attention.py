import argparse
import os, json, copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from data_util import LineByLineCRITTDatasetByNLLBmGPT, to_device, lang_code, extension_by_level
from attention_util import dummy_attention_by_batch, src_seq_att, tgt_seq_att
from data_correction import filtered_studies
import numpy as np
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")


def main():
    thread_count = int(os.environ.get("NMT_COGNITIVE_THREADS", "8"))
    torch.set_num_threads(thread_count)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, min(2, thread_count // 2)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--session_file", default=None, type=str, required=True,
                        help="session file in json")
    parser.add_argument("--source_table_dir",  default=None, type=str, required=True,
                        help="directory of st tables")
    parser.add_argument("--outputf", default=None, type=str,
                        help="output file")
    parser.add_argument("--src_level", default="token", type=str,
                        help="source segmentation level")
    parser.add_argument("--tgt_level", default="segment", type=str,
                        help="target segmentation level")
    parser.add_argument('--labels',
                        default=['Dur', 'FixS', 'FixT', 'TrtS', 'TrtT'],
                        type=str, nargs='*')
    parser.add_argument("--bsz", default=16, type=int, help="Batch size")
    parser.add_argument("--max", action='store_true', help="use max over heads instead of mean")
    parser.add_argument("--num_workers", default=1, type=int,
                        help="number of workers")
    parser.add_argument("--cpu", action='store_true', help="cpu instead of cuda")
    parser.add_argument("--translation_only", action='store_true', help="use translation data only")
    parser.add_argument("--dummy_attention", action='store_true', help="compute uniform attentional feature")
    parser.add_argument("--normalize", action='store_true', help="normalize attentional feature")
    args = parser.parse_args()
    print(f"Using torch CPU threads: {torch.get_num_threads()}", flush=True)

    model_checkpoint = "facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    lm_model_checkpoint = "sberbank-ai/mGPT"
    tokenizer_lm = GPT2Tokenizer.from_pretrained(lm_model_checkpoint)
    model_lm = GPT2LMHeadModel.from_pretrained(lm_model_checkpoint)

    xentropy = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    nlayers, nheads, vocab_sz = model.config.encoder_layers, model.config.encoder_attention_heads, model.config.vocab_size

    device = torch.device("cpu")
    if not args.cpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            model_lm = model_lm.to(device)
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
            model = model.to(device)
            model_lm = model_lm.to(device)
    model.eval()
    model_lm.eval()
    print(f"Using device: {device}", flush=True)

    with open(args.session_file, "r") as f:
        corpora_sessions = json.load(f)

    if args.translation_only:
        corpora_sessions = {corpus: [sess for sess in sessions if sess['session_id'].split('_')[1].startswith('T')]
                            for corpus, sessions in corpora_sessions.items()}


    for study, fsess in filtered_studies.items():
        if fsess == "all":
            corpora_sessions = {corpus: [sess for sess in sessions if sess['study'] != study]
                                for corpus, sessions in corpora_sessions.items()}
        else:
            for ses in fsess:
                corpora_sessions = {corpus: [sess for sess in sessions if sess['study'] != study and sess['session_id'] != ses]
                                    for corpus, sessions in corpora_sessions.items()}


    outf = open(args.outputf, "w")

    header = ["src_lang", "tgt_lang", "study", "session_id", "src_tokens", "tgt_tokens", "src_segid", "tgt_segid",
              "src_token_ids", "tgt_token_ids", "src_posq", "tgt_posq", "xent", "xmi", "src_surp", "tgt_surp"]

    # src_tok2cont_cov (), src_tok2cont_ent, src_cont2tok_cov, src_tok2eos, tgtseq2srctok_cov
    # tgttok2srcseq_var, tgttok2srcseq_ent, tok2eos_xatt, tok2cont_datt_cov, tok2cont_datt_ent

    header += ['x2cont', 'cont2x', 'x_ent', 'x2x', 'x2eos', 'tgtseq2x', 'y2srcseq_ent', 'eos_xatt', 'y2cont', 'y2y', 'y_ent']

    header += args.labels # + src/tgt POS
    outf.write('\t'.join(header) + '\n')

    for corpus, sessions in corpora_sessions.items():
        for session in sessions:
            # st tables because there are inconsistencies in ag tables (trados/CREATIVE)
            fname = f"{session['session_id']}.{extension_by_level(args.src_level, args.tgt_level)}"
            src_table_file = os.path.join(args.source_table_dir, session['study'], f"{session['study']}-tables", fname)
            if not os.path.exists(src_table_file):
                src_table_file = os.path.join(args.source_table_dir, fname)

            if os.path.exists(src_table_file):
                print(f"Processing: {session['study']}/{session['session_id']}", flush=True)
                dataset = LineByLineCRITTDatasetByNLLBmGPT(tokenizer, tokenizer_lm, session, src_table_file, args.labels,
                                                       src_level=args.src_level, tgt_level=args.tgt_level)

                dataloader = DataLoader(dataset, args.bsz, collate_fn=dataset.collate_encode_fn,
                                        num_workers=args.num_workers)

                lines = []
                session_meta = []
                session_metric_tensors = []
                for bid, batch in enumerate(dataloader):
                    with torch.no_grad():
                        if batch is None:
                            continue
                        src_tokens, src_token_ids, tgt_tokens, tgt_token_ids, src_posqs, tgt_posqs, src_segids, tgt_segids,\
                            src_tokenized, tgt_tokenized, src_token_ids_lm, tgt_token_ids_lm, src_tokenized_lm, tgt_tokenized_lm, ys = batch

                        src_tokenized, tgt_tokenized = to_device(src_tokenized, device), to_device(tgt_tokenized, device)
                        src_tokenized_lm, tgt_tokenized_lm = to_device(src_tokenized_lm, device), to_device(tgt_tokenized_lm, device)

                        output = model(input_ids=src_tokenized["input_ids"], attention_mask=src_tokenized['attention_mask'], labels=tgt_tokenized["input_ids"])
                        logits = output.logits

                        # attention: (bsz, nlayers, nheads, tgt_len, src_len)
                        enc_att = torch.cat([layer.unsqueeze(0) for layer in output.encoder_attentions], 0).permute(
                            1, 0, 2, 3, 4)
                        dec_att = torch.cat([layer.unsqueeze(0) for layer in output.decoder_attentions], 0).permute(
                            1, 0, 2, 3, 4)
                        x_att = torch.cat([layer.unsqueeze(0) for layer in output.cross_attentions],
                                          0).permute(1, 0, 2, 3, 4)

                        # lm output
                        src_output_lm = model_lm(input_ids=src_tokenized_lm["input_ids"], attention_mask=src_tokenized_lm['attention_mask'])
                        tgt_output_lm = model_lm(input_ids=tgt_tokenized_lm["input_ids"], attention_mask=tgt_tokenized_lm['attention_mask'])
                        tgt_oologits = tgt_output_lm.logits
                        src_oologits = src_output_lm.logits


                        for i, (src_ids, src_ids_lm, src_tids_lm, tgt_ids, tgt_tids, tgt_ids_lm, tgt_tids_lm, \
                                src_toks, tgt_toks, src_posq, tgt_posq, src_segid, tgt_segid, \
                                xatt, eatt, datt, logit, src_oologit, tgt_oologit, y) in enumerate(zip(
                            src_token_ids, src_token_ids_lm, src_tokenized_lm["input_ids"], tgt_token_ids, tgt_tokenized["input_ids"],
                            tgt_token_ids_lm, tgt_tokenized_lm["input_ids"], src_tokens, tgt_tokens, src_posqs, tgt_posqs,
                            src_segids, tgt_segids, x_att, enc_att, dec_att, logits, src_oologits, tgt_oologits, ys)):
                            src_ids_list = list(src_ids)
                            src_ids_lm_list = list(src_ids_lm)
                            tgt_ids_list = list(tgt_ids) if tgt_ids is not None else None
                            tgt_ids_lm_list = list(tgt_ids_lm) if tgt_ids is not None else None

                            src_ids = torch.as_tensor(src_ids_list, dtype=torch.long, device=eatt.device)
                            src_ids_lm = torch.as_tensor(src_ids_lm_list, dtype=torch.long, device=src_oologit.device)
                            if tgt_ids is not None:
                                tgt_ids = torch.as_tensor(tgt_ids_list, dtype=torch.long, device=datt.device)
                                tgt_ids_lm = torch.as_tensor(tgt_ids_lm_list, dtype=torch.long, device=tgt_oologit.device)


                            x2cont, cont2x, x_ent, x2x, x2eos, tgtseq2x = \
                                src_seq_att(eatt, xatt, src_ids, dummy=args.dummy_attention, normalize=args.normalize)

                            y2srcseq_ent, y2eos, y2cont, y2y, y_ent = \
                                tgt_seq_att(xatt, datt, tgt_ids, dummy=args.dummy_attention, normalize=args.normalize)


                            src_surprisal = xentropy(src_oologit[src_ids_lm], src_tids_lm[src_ids_lm])
                            if args.normalize:
                                src_len = len(src_ids_lm)
                                src_surprisal /= src_len


                            if tgt_ids is not None and tgt_ids.size(0) > 0:
                                # cross entropy loss
                                xent = xentropy(logit[tgt_ids], tgt_tids[tgt_ids])

                                # cross mutual info
                                output_only_xent = xentropy(tgt_oologit[tgt_ids_lm], tgt_tids_lm[tgt_ids_lm])
                                xmi = (output_only_xent - xent)

                                tgt_surprisal = output_only_xent

                                if args.normalize:
                                    tgt_len = tgt_ids.size(0)
                                    xent /= tgt_len
                                    xmi /= tgt_len
                                    tgt_surprisal /= len(tgt_ids_lm)


                            else:
                                xent = torch.tensor(float("nan"), device=device)
                                xmi = torch.tensor(float("nan"), device=device)
                                tgt_surprisal = torch.tensor(float("nan"), device=device)

                            session_metric_tensors.append(torch.stack([
                                xent,
                                xmi,
                                src_surprisal,
                                tgt_surprisal,
                                x2cont,
                                cont2x,
                                x_ent,
                                x2x,
                                x2eos,
                                tgtseq2x,
                                y2srcseq_ent,
                                y2eos,
                                y2cont,
                                y2y,
                                y_ent,
                            ]))

                            str_src_ids = '+'.join(list(map(str, src_ids_list)))
                            str_tgt_ids = '+'.join(list(map(str, tgt_ids_list))) if tgt_ids_list is not None else "nan"
                            session_meta.append([
                                dataset.src_lang,
                                dataset.tgt_lang,
                                session['study'],
                                session['session_id'],
                                src_toks,
                                tgt_toks,
                                str(src_segid),
                                str(tgt_segid),
                                str_src_ids,
                                str_tgt_ids,
                                str(src_posq),
                                str(tgt_posq),
                                list(map(str, y.tolist())),
                            ])

                if session_metric_tensors:
                    session_metric_values = torch.stack(session_metric_tensors).detach().cpu().tolist()
                    for meta, metric_values in zip(session_meta, session_metric_values):
                        xent_val, xmi_val, src_surprisal_val, tgt_surprisal_val = metric_values[:4]
                        att_values = metric_values[4:]
                        y_values = meta.pop()
                        line = meta + [
                            str(xent_val),
                            str(xmi_val),
                            str(src_surprisal_val),
                            str(tgt_surprisal_val),
                        ]
                        line += [str(val) for val in att_values]
                        line += y_values
                        lines.append('\t'.join(line))

                outf.write('\n'.join(lines) + '\n')
                outf.flush()
                print(f"Completed: {session['session_id']} ({len(lines)} rows)", flush=True)
    outf.close()

if __name__ == "__main__":
    main()
