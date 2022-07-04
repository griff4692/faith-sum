# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 22:26:39 2022

@author: Mert
"""

from datasets import load_dataset

import nltk
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    BartForSequenceClassification,
    BartForConditionalGeneration,
    BartTokenizer
)

import re
from tqdm import tqdm


def expectation_maximization(extractor, abstractor, batch, theta):

    ##############################EXTRACTOR####################################
    ###########################################################################

    extractor_logits = extractor(input_ids=batch['input_ids']).logits  # p(s^êxt|D)
    extractor_logits *= theta
    encoded_sentences = []
    sentence_masks = []
    texts = tokenizer.batch_decode(batch['input_ids'])
    for text in texts:
        text = text[3:-4]  # remove bos and eos
        encoded_sentences.append(
            tokenizer.batch_encode_plus(
                nltk.sent_tokenize(text),
                max_length=max_sentence_size,
                padding='max_length',
                truncation=True
            )
        )

        encoded_sentences[-1] = {
            key: torch.LongTensor(
                value
            ).to(device) for key, value in encoded_sentences[-1].items()
        }
        sentence_mask = torch.zeros(max_sentence_size)
        sentence_mask[:encoded_sentences[-1]['input_ids'].size()[0]] = 1
        sentence_masks.append(sentence_mask.unsqueeze(0).long())
    sentence_masks = torch.cat(sentence_masks, 0).to(device)

    extractor_logits.masked_fill_(sentence_masks == 0, -1e20)
    extractor_logits = extractor_logits - extractor_logits.logsumexp(-1).view(-1, 1)

    ##############################ABSTRACTOR###################################
    ###########################################################################

    abstractor_logits = []
    loss_fct = nn.CrossEntropyLoss(
        weight=None,
        size_average=None,
        ignore_index=abstractor.config.pad_token_id,
        reduce=None,
        reduction='none',
        label_smoothing=0.0
        )
    negative_elbo = []
    marginal_likelihood = []
    logits = []
    for i in range(len(sentence_masks)):
        labels = batch['labels'][i]
        abstractor_logits.append(
            abstractor(
                input_ids=encoded_sentences[i]['input_ids'],
                attention_mask=encoded_sentences[i]['attention_mask'],
                labels=labels.unsqueeze(0).repeat_interleave(
                    encoded_sentences[i]['input_ids'].size(0),
                    0
                )
            ).logits
        )  # p(w|w<i,s_i)
        abstractor_logits[i] = abstractor_logits[i] - abstractor_logits[
            i].logsumexp(-1).unsqueeze(-1)
        labels = labels.unsqueeze(0).repeat_interleave(
            abstractor_logits[i].size(0),
            0
        )
        loglikelihood = - loss_fct(
            abstractor_logits[i].transpose(-1, 1),
            labels
        )  # p(w_i | s_i, w_<i)
        loglikelihood += extractor_logits[i][:sentence_masks[i].sum()].view(-1, 1)  # p(S|D, \theta)
        loglikelihood.masked_fill_(labels == abstractor.config.pad_token_id, 0)
        posterior = torch.exp(
            loglikelihood - loglikelihood.masked_fill(
                labels == abstractor.config.pad_token_id,
                -1e20
            ).logsumexp(0)
        ).masked_fill(labels == abstractor.config.pad_token_id, 0).detach()  # E-step
        negative_elbo.append(
            - torch.sum(loglikelihood * posterior, 0).sum(0)
        )  # M-step
        marginal_likelihood.append(
            loglikelihood.logsumexp(0).masked_fill(
                labels[0] == abstractor.config.pad_token_id, 0
            )
        )
        logits.append(
            (abstractor_logits[i] + extractor_logits[i][
                :sentence_masks[i].sum()].view(-1, 1, 1)
                ).logsumexp(0)
            )
    marginal_likelihood = torch.stack(marginal_likelihood)
    marginal_likelihood = marginal_likelihood.sum() / torch.sum(marginal_likelihood < 0.)
    negative_elbo = torch.stack(negative_elbo).mean()
    logits = torch.stack(logits)
    return negative_elbo, marginal_likelihood.item(), logits


def preprocess(dataset):
    dataset['article'] = [re.sub('\(.*?\)', '', t) for t in dataset['article']]
    dataset['article'] = [t.replace('--', '') for t in dataset['article']]
    return dataset


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, source_len, summ_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_len = source_len
        self.summ_len = summ_len
        self.text = self.dataset['article']
        self.summary = self.dataset['highlights']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        summary = str(self.summary[i])
        summary = ' '.join(summary.split())
        text = str(self.text[i])
        text = ' '.join(text.split())
        source = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.text_len,
            return_tensors='pt',
            pad_to_max_length=True
        )  # Each source sequence is encoded and padded to max length in batches
        target = self.tokenizer.batch_encode_plus(
            [summary],
            max_length=self.summ_len,
            return_tensors='pt',
            pad_to_max_length=True
        )  # Each target sequence is encoded and padded to max lenght in batches

        source_ids = source['input_ids'].squeeze()
        source_masks = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        # target_masks = target['attention_mask'].squeeze()

        return {
            'input_ids': source_ids.to(torch.long),
            'attention_mask': source_masks.to(torch.long),
            'labels': target_ids.to(torch.long),
        }


##################################FLAGS#######################################
##############################################################################

debug = False

torch.autograd.set_detect_anomaly(debug)

hf_model = 'facebook/bart-base'
device = 'cpu'
max_sentence_size = 50
batch_size = 2
lr = 1e-5
epochs = 200
theta = 1  # encourage extractor sparsity

################################TOKENIZER#####################################
##############################################################################

tokenizer = BartTokenizer.from_pretrained(
    pretrained_model_name_or_path=hf_model
)
##################################DATA########################################
##############################################################################

dataset = load_dataset('cnn_dailymail', '3.0.0')

# As we can observe, dataset is too large so for now we will consider just 8k rows for training and 4k rows for validation
train_dataset = dataset['train'][:8000]
val_dataset = dataset['validation'][:4000]

train_dataset = preprocess(train_dataset)
val_dataset = preprocess(val_dataset)

tokenizer = BartTokenizer.from_pretrained(
    pretrained_model_name_or_path=hf_model
)

train_dataset = CustomDataset(
    train_dataset,
    tokenizer,
    270,
    160
)

val_dataset = CustomDataset(
    val_dataset,
    tokenizer,
    270,
    160
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=0
)

##################################MODEL#######################################
##############################################################################

extractor = BartForSequenceClassification.from_pretrained(
    hf_model,
    num_labels=max_sentence_size
).to(device)

abstractor = BartForConditionalGeneration.from_pretrained(
    hf_model,
).to(device)

optimizer = torch.optim.Adam(
    list(extractor.parameters()) + list(abstractor.parameters()),
    lr=lr
)

###############################TRAINING#######################################
##############################################################################

i = 0

train_total = len(train_loader)
val_total = len(val_loader)

for epoch in range(epochs):

    abstractor.train()
    extractor.train()
    train_losses = []
    train_marginal_loglikelihoods = []

    for batch in tqdm(train_loader, total=train_total):
        batch = {key: value.to(device) for (key, value) in batch.items()}

        optimizer.zero_grad()

        negative_elbo, marginal_loglikelihood, _ = expectation_maximization(
            extractor,
            abstractor,
            batch,
            theta,
        )

        negative_elbo.backward()
        optimizer.step()
        train_losses.append(negative_elbo.item())
        train_marginal_loglikelihoods.append(marginal_loglikelihood)
        i+=1
        if i%200 == 0:
            print(
                '\nEpoch {} mean train ELBO:{}'.format(
                    epoch, np.mean(train_losses)
                )
            )
            print(
                'Epoch {} mean train LL:{}'.format(
                    epoch, np.mean(train_marginal_loglikelihoods)
                )
            )
            print(
                'Epoch {} mean train BCE:{}'.format(
                    epoch, -np.mean(train_marginal_loglikelihoods)
                )
            )
            train_losses = []
            train_marginal_loglikelihoods = []
    with torch.no_grad():
        abstractor.eval()
        extractor.eval()
        val_losses = []
        val_marginal_loglikelihoods = []
        print('Validation...')
        for batch in tqdm(val_loader, total=val_total):
            batch = {key: value.to(device) for (key, value) in batch.items()}
            negative_elbo, marginal_loglikelihood, logits = expectation_maximization(
                extractor,
                abstractor,
                batch,
                theta,
            )
            val_losses.append(negative_elbo.item())
            val_marginal_loglikelihoods.append(marginal_loglikelihood)

        print(
            '\nEpoch {} mean val ELBO:{}'.format(
                epoch, np.mean(val_losses)
            )
        )
        print(
            'Epoch {} mean val LL:{}'.format(
                epoch, np.mean(val_marginal_loglikelihoods)
            )
        )
        print(
            'Epoch {} mean val BCE:{}'.format(
                epoch, -np.mean(val_marginal_loglikelihoods)
            )
        )
        print(
            ''.join(
                tokenizer.batch_decode(logits.max(-1)[1])
                )
            )  # example vanilla generation

##############################################################################
##############################################################################