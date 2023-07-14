# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain BERT"""

from functools import partial

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import BertModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed

from pytorch_memlab import MemReporter

def is_rank_0():
    # if torch.distributed.get_rank() == 0:
    if torch.cuda.current_device() == 0:
        return True
    else:
        return False

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building BERT model ...')

    args = get_args()
    vocab_size = 1024
    #vocab_size = args.vocab_size
    ##TODO : Hack for synthetic datareader
    args.padded_vocab_size = vocab_size ##Hack
    args.custom_token_counting = True
    num_tokentypes = 2 if args.bert_binary_head else 0

    with deepspeed.zero.Init(
            data_parallel_group=mpu.get_data_parallel_group(),
            remote_device=None if args.remote_device == 'none' else args.remote_device,
            config_dict_or_path=args.deepspeed_config,
            enabled=args.zero_stage == 3,
            mpu=mpu
    ):
        model = BertModel(
            num_tokentypes=num_tokentypes,
            add_binary_head=args.bert_binary_head,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process)

    if is_rank_0():
        print(model)

    # print(f"number of params: {sum([p.numel() for p in model.parameters()])}")
    # p_list = []
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         p_list.append([name, param.numel()])
    # p_list.sort(key=lambda x : x[1], reverse=True)
    # if is_rank_0():
    #     print(p_list)
    #     print('\n')

    return model


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    local_world_size = 1 if not mpu.sequence_parallel_is_initialized() else mpu.get_sequence_parallel_world_size()
    local_rank = torch.distributed.get_rank() if not mpu.sequence_parallel_is_initialized() \
               else mpu.get_sequence_parallel_rank()

    seq_length = data_b['text'].size(1)
    sub_seq_length = seq_length // local_world_size
    sub_seq_start = local_rank * sub_seq_length
    sub_seq_end = (local_rank+1) * sub_seq_length

    # Unpack.
    tokens = data_b['text'][:, sub_seq_start:sub_seq_end].long()
    types = data_b['types'][:, sub_seq_start:sub_seq_end].long()
    sentence_order = data_b['is_random'].long()

    # loss_mask = data_b['loss_mask'][:, sub_seq_start:sub_seq_end].float()
    # lm_labels = data_b['labels'][:, sub_seq_start:sub_seq_end].long()
    # padding_mask = data_b['padding_mask'][:, sub_seq_start:sub_seq_end].long()

    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


def loss_func(loss_mask, sentence_order, output_tensor):
    lm_loss_, sop_logits = output_tensor

    lm_loss_ = lm_loss_.float()
    lm_loss = torch.sum(lm_loss_.view(-1))
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    if sop_logits is not None:
        sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                   sentence_order.view(-1),
                                   ignore_index=-1)
        sop_loss = sop_loss.float()
        loss = lm_loss + sop_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss, sop_loss])
        return loss, {'lm loss': averaged_losses[0],
                      'sop loss': averaged_losses[1]}

    else:
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss])
        return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output_tensor = model(tokens, padding_mask, tokentype_ids=types,
                        lm_labels=lm_labels)

    # mem_reporter = MemReporter(model)
    # if is_rank_0() and args.curr_iteration == 0:
    #     mem_reporter.report(device=torch.device(f'cuda:{torch.cuda.current_device()}'), loc='After iter')

    return output_tensor, partial(loss_func, loss_mask, sentence_order)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        binary_head=args.bert_binary_head)
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    train_valid_test_datasets_provider = None

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step, args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
