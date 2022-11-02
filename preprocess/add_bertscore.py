import os
import regex as re

from datasets import load_from_disk
import argparse
from bert_score.scorer import BERTScorer


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Oracles for dataset')

    parser.add_argument('--dataset', default='xsum_pegasus')
    parser.add_argument('--data_dir', default='/nlp/projects/faithsum')
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--device', default=0, type=int)

    args = parser.parse_args()
    in_dir = os.path.join(args.data_dir, args.dataset)

    dataset = load_from_disk(in_dir)
    bs = BERTScorer(
        device=args.device, lang='en', batch_size=args.batch_size, use_fast_tokenizer=True
    )

    def get_sents(source):
        tps = re.split(r'(<s\d+>)', source)
        source_sents = []
        for tp_idx, tp in enumerate(tps):
            if re.match(r'(<s\d+>)', tp) is not None:
                source_sents.append(tps[tp_idx + 1].strip())
        return source_sents

    def remove_oracle(source_sents, remove_idxs):
        return [s for i, s in enumerate(source_sents) if i not in remove_idxs]

    def keep_oracle(source_sents, keep_idxs):
        return [s for i, s in enumerate(source_sents) if i  in keep_idxs]

    def add_bert_score(batch):
        oracle_idxs = batch['oracle_idxs']
        source = batch['source_annotated']
        source_sents = [get_sents(x) for x in source]
        non_oracle = [remove_oracle(ss, oracle_idxs) for ss, oi in zip(source_sents, oracle_idxs)]
        oracle = [keep_oracle(ss, oi) for ss, oi in zip(source_sents, oracle_idxs)]

        source_str = [' '.join(x) for x in source_sents]
        oracle_str = [' '.join(x) for x in oracle]
        non_oracle_str = [' '.join(x) for x in non_oracle]
        ref_str = batch['summary']

        # BertScore on the oracle <-> reference
        # BertScore on the non-oracle <-> reference
        # BertScore on the full <-> reference
        src_p, src_r, src_f = bs.score(ref_str, source_str, batch_size=args.batch_size)
        oracle_p, oracle_r, oracle_f = bs.score(ref_str, oracle_str, batch_size=args.batch_size)
        non_p, non_r, non_f = bs.score(ref_str, non_oracle_str, batch_size=args.batch_size)

        # Store these values
        bs_stats = {
            'source_bs_precision': src_p.cpu().numpy().tolist(),
            'source_bs_recall': src_r.cpu().numpy().tolist(),
            'source_bs_f1': src_f.cpu().numpy().tolist(),
            'oracle_bs_precision': oracle_p.cpu().numpy().tolist(),
            'oracle_bs_recall': oracle_r.cpu().numpy().tolist(),
            'oracle_bs_f1': oracle_f.cpu().numpy().tolist(),
            'non_oracle_bs_precision': non_p.cpu().numpy().tolist(),
            'non_oracle_bs_recall': non_r.cpu().numpy().tolist(),
            'non_oracle_bs_f1': non_f.cpu().numpy().tolist(),
            'oracle_focus_score': (oracle_p - non_p).cpu().numpy().tolist()
        }

        return bs_stats


    dataset = dataset.map(
        add_bert_score,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc="Adding BertScore",
    )

    print(f'Saving back to {in_dir}')
    dataset.save_to_disk(in_dir)
