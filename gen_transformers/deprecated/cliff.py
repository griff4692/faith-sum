# if self.hparams.is_cliff:
#     # START OF CLIFF
#     import itertools
# 
#     pos_idxs = [0, 1]
#     pos_contrasts = list(itertools.combinations(pos_idxs, 2))
#     # Heavily borrowed from CLIFF Github
#     # https://github.com/ShuyangCao/cliff_summ/blob/8913d92f85457e030d77dc5dfa255bea7e226dc4/models/pegasus/contrastive_trainer.py
#     decoder_states = contrast_outputs.decoder_hidden_states[-1]
#     decoder_proj = self.sent_bart.contrast_projection(decoder_states)
#     decoder_proj_mask = cand_labels == -100
#
#     decoder_proj.masked_fill_(decoder_proj_mask.unsqueeze(-1), 0)
#     decoder_pooled = decoder_proj.sum(dim=1) / (cand_labels != -100).sum(dim=-1, keepdim=True)
#     states_norm = decoder_pooled / decoder_pooled.norm(dim=-1, keepdim=True)
#     cosine_sim = torch.matmul(states_norm, states_norm.transpose(0, 1))
#     inverted_identity = 1 - torch.eye(len(cosine_sim), device=self.device)
#     cosine_sim_exp = cosine_sim.exp() * inverted_identity
#     denom = cosine_sim_exp.sum(dim=1)
#     contrast_nll = 0.0
#     for a, b in pos_contrasts:
#         exp_sim = cosine_sim_exp[a, b]
#         contrast_nll = contrast_nll - torch.log(exp_sim / denom[a])
#     cliff_loss = contrast_nll / len(pos_contrasts)
#     # END OF CLIFF
