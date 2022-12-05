import torch
import numpy as np


def infer_hf_model(args):
    if args.hf_model is None:
        # Infer it
        if args.dataset == 'samsum':
            args.hf_model = 'lidiya/bart-large-xsum-samsum'
        elif args.dataset == 'cnn_dailymail':
            args.hf_model = 'facebook/bart-large-cnn'
        else:
            raise Exception('Unknown Yet.')


def corrupt_oracle_indicators(batch, has_bos=True, full_random=False):
    ids = []
    batch_size = len(batch['oracle_labels'])
    for batch_idx in range(batch_size):
        # p = batch['priority'][batch_idx]
        # trunc_idx = min(len(p), oracle_mask_k)
        cls_locations = batch['cls_mask'][batch_idx].unsqueeze(0)
        prev_mask = batch['attention_mask'][batch_idx].unsqueeze(0)
        idx_to_keep = batch['oracle_labels'][batch_idx]

        other_sents = [int(x) for x in np.arange(cls_locations.cpu().numpy().sum()) if x not in idx_to_keep.cpu().numpy()]

        if full_random:
            num_to_replace = min(len(other_sents), len(idx_to_keep))
            idx_to_keep = np.sort(np.random.choice(other_sents, size=(num_to_replace, ), replace=False))
        else:
            if len(other_sents) >= 1:
                other_sent = np.random.choice(other_sents)
                # coin_flip = np.random.random()
                # this_sent = np.random.choice(list(range(len(idx_to_keep))))

                idx_to_keep[np.random.randint(len(idx_to_keep))] = other_sent
                idx_to_keep = torch.sort(idx_to_keep)[0]

                # if coin_flip < 0.33:  # Swap
                #     idx_to_keep[np.random.randint(len(idx_to_keep))] = other_sent
                #     idx_to_keep = torch.sort(idx_to_keep)[0]
                # elif coin_flip < 0.66:  # Add
                #     idx_to_keep = idx_to_keep.cpu().numpy().tolist()
                #     idx_to_keep.append(other_sent)
                # else:  # Delete
                #     idx_to_keep = idx_to_keep.cpu().numpy().tolist()
                #     idx_to_keep.pop(this_sent)
        updated_mask = sentence_indicators(cls_locations, idx_to_keep, prev_mask, has_bos=has_bos)
        ids.append(updated_mask)
    return torch.cat(ids)


def implement_oracle_indicators(batch, has_bos=True):
    ids = []
    batch_size = len(batch['oracle_labels'])
    for batch_idx in range(batch_size):
        # p = batch['priority'][batch_idx]
        # trunc_idx = min(len(p), oracle_mask_k)
        cls_locations = batch['cls_mask'][batch_idx].unsqueeze(0)
        prev_mask = batch['attention_mask'][batch_idx].unsqueeze(0)
        idx_to_keep = batch['oracle_labels'][batch_idx]
        updated_mask = sentence_indicators(cls_locations, idx_to_keep, prev_mask, has_bos=has_bos)
        ids.append(updated_mask)
    return torch.cat(ids)


def sentence_indicators(cls_mask, sent_idx_to_mask, prev_mask, has_bos=True):
    """
    :param cls_mask: indications of where sentences tokens are
    :param sent_idx_to_mask: which sentences to mask (the sentence order, not location in cls_mask)
    :param prev_mask: attention mask you are updating to mask all sentences NOT in sent_idx_to_mask
    :return:
    """
    sent_mask = torch.zeros_like(cls_mask, device=cls_mask.device).long()
    sent_locs = cls_mask.nonzero()[:, 1]
    max_seq_len = cls_mask.size()[1]
    num_sents = len(sent_locs)
    for sent_idx, sent_loc in enumerate(sent_locs):
        sent_loc = sent_loc.item()
        end_loc = sent_locs[sent_idx + 1].item() if sent_idx + 1 < num_sents else max_seq_len
        if sent_idx in sent_idx_to_mask:
            sent_mask[0, sent_loc:end_loc] = 2
        else:
            sent_mask[0, sent_loc:end_loc] = 1
    if has_bos:
        sent_mask[:, 0] = 2  # Always focus on the BOS token
    sent_mask.masked_fill_(prev_mask == 0, 0)
    return sent_mask


def implement_oracle_masking(batch):
    updated_attention_masks = []
    batch_size = len(batch['oracle_labels'])
    for batch_idx in range(batch_size):
        # p = batch['priority'][batch_idx]
        # trunc_idx = min(len(p), oracle_mask_k)
        cls_locations = batch['cls_mask'][batch_idx].unsqueeze(0)
        prev_mask = batch['attention_mask'][batch_idx].unsqueeze(0)
        # idx_to_keep = p[:trunc_idx]
        idx_to_keep = batch['oracle_labels'][batch_idx]
        # Masks everything but the sentences specified by idx_to_keep
        # cls_locations is True if there's a <s{idx}> tag in the spot, 0 otherwise.
        # Used for sentence boundary detection in the method.
        updated_mask = sentence_mask(cls_locations, idx_to_keep, prev_mask)
        updated_attention_masks.append(updated_mask)
    return torch.cat(updated_attention_masks)


def sentence_mask(cls_mask, sent_idx_to_mask, prev_mask):
    """
    :param cls_mask: indications of where sentences tokens are
    :param sent_idx_to_mask: which sentences to mask (the sentence order, not location in cls_mask)
    :param prev_mask: attention mask you are updating to mask all sentences NOT in sent_idx_to_mask
    :return:
    """
    sent_mask = torch.zeros_like(cls_mask, device=cls_mask.device).long()
    sent_locs = cls_mask.nonzero()[:, 1]
    max_seq_len = cls_mask.size()[1]
    num_sents = len(sent_locs)
    for sent_idx, sent_loc in enumerate(sent_locs):
        sent_loc = sent_loc.item()
        end_loc = sent_locs[sent_idx + 1].item() if sent_idx + 1 < num_sents else max_seq_len
        if sent_idx in sent_idx_to_mask:
            sent_mask[0, sent_loc:end_loc] = 1
    sent_mask[:, 0] = 1  # Always attend to the BOS token
    sent_mask.masked_fill_(prev_mask == 0, 0)
    return sent_mask
