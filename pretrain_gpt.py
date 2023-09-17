# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributed as ptdist

from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.dist import setup_torch
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator import get_accelerator

import os
import time
import math
import subprocess
import logging
import wandb
from pathlib import Path
import socket


log = logging.getLogger(__name__)
HERE = Path(os.path.abspath(__file__)).parent

setup_torch(
    backend='deepspeed',
    port='5432',
)
log.info(f'Hello from {ptdist.get_rank()} / {ptdist.get_world_size()}')


def get_rank() -> int:
    return ptdist.get_rank()


def is_first_rank():
    return get_rank() == 0


# WBRUN = None
if is_first_rank():
    tensorboard_dir = os.environ.get('TENSORBOARD_DIR', None)
    if tensorboard_dir is not None:
        log.info(f'Patching tensorboard from {tensorboard_dir}')
        wandb.tensorboard.patch(root_logdir=tensorboard_dir)
    # wbrun_id = wandb.util.generate_id()
    current_time = time.time()
    local_time = time.localtime(current_time)
    # os.environ['WANDB_RUN_GROUP'] = f'experiment-{generate_id()}'
    # seq_len = os.environ.get('SEQ_LEN', 1)
    # seq_len = int(seq_len) // 1024
    # global_batch = os.environ.get('GLOBAL_BATCH', 1)
    # mp_size = os.environ.get('MPSIZE', 1)
    # pp_size = os.environ.get('PPSIZE', 1)
    # world_size = ptdist.get_world_size()
    wandb.init(
        project='Megatron-DS-Benchmarking',
        sync_tensorboard=True,
        dir=tensorboard_dir,
        resume='allow',
        # dir=os.getcwd(),
        # sync_tensorboard=True,
        # group=f'experiment-{generate_id()}'
        # group='long-seq-ANL0',
        # name=f'{seq_len}kseq-len-no-seq-parallel-{global_batch}global-batch-{world_size}GPUs-{mp_size}MP-{pp_size}PP-{local_time.tm_hour}-{local_time.tm_min}'
        # name=f'{seq_len}kseq-len-new-{global_batch}global-batch-{world_size}GPUs-{mp_size}MP-{pp_size}PP-{local_time.tm_hour}-{local_time.tm_min}'
    )
    assert wandb.run is not None
    # assert WBRUN is not None and WBRUN is wandb.run
    wandb.run.log_code(HERE.as_posix())  # type:ignore
    # wandb.run.config.update({'current_time': current_time})
    wandb.run.config.update({'current_time': current_time})
    model_size = os.environ.get('MODEL_SIZE', None)
    if model_size is not None:
        wandb.run.config.update({'MODEL_SIZE': model_size})
    if wandb.run is not None:
        wandb.run.config.update({'world_size': ptdist.get_world_size()})
        env = dict(os.environ)
        _ = env.pop('LS_COLORS', None)
        wandb.run.config.update({'env': env})
        hostname = socket.gethostbyaddr(socket.gethostname())[0]
        if hostname.startswith('theta'):
            wandb.run.config.update({'machine': 'ThetaGPU'})
        elif hostname.startswith('x3'):
            wandb.run.config.update({'machine': 'Polaris'})
        elif hostname.startswith('x1'):
            wandb.run.config.update({'machine': 'Sunspot'})
        elif hostname.startswith('nid'):
            wandb.run.config.update({'machine': 'Perlmutter'})
        else:
            wandb.run.config.update({'machine': hostname})


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)
    args = get_args()
    if wandb.run is not None:
        wandb.run.config.update(vars(args))

    with deepspeed.zero.Init(
            data_parallel_group=mpu.get_data_parallel_group(),
            remote_device=None if args.remote_device == 'none' else args.remote_device,
            config_dict_or_path=args.deepspeed_config,
            enabled=args.zero_stage == 3,
            mpu=mpu
    ):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Precompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(
                torch.ones(
                    (1, args.seq_length, args.seq_length),
                    device=get_accelerator().current_device_name()
                )
            ).view(1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)
        else:
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    # if is_first_rank() and WBRUN is not None and WBRUN is wandb.run:
    if wandb.run is not None:
        wandb.run.watch(
            model,
            log='all',
            log_graph=True,
        )

    see_memory_usage(f"After Building Model", force=True)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if wandb.run is not None:
        wandb.run.config.update({'num_params': num_params})
    print_rank_0('\n ------------------------ ')
    print_rank_0(f'num of parameters {num_params}')
    print_rank_0('------------------------\n ')

    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss
    )

    # for sequence parallel
    world_rank = torch.distributed.get_rank()
    local_world_size = 1 if not mpu.sequence_parallel_is_initialized() else mpu.get_sequence_parallel_world_size()
    local_rank = world_rank % local_world_size
    seq_length = tokens.size(1)

    assert seq_length % local_world_size == 0
    sub_seq_length = seq_length // local_world_size
    sub_seq_start = local_rank * sub_seq_length
    sub_seq_end = (local_rank + 1) * sub_seq_length

    # Unpack.
    # tokens = data_b['text'][:, sub_seq_start:sub_seq_end].long()
    # types = data_b['types'][:, sub_seq_start:sub_seq_end].long()
    # sentence_order = data_b['is_random'].long()
    # loss_mask = data_b['loss_mask'][:, sub_seq_start:sub_seq_end].float()
    # lm_labels = data_b['labels'][:, sub_seq_start:sub_seq_end].long()
    # #padding_mask = data_b['padding_mask'].long() 
    # ##SAGE check
    # padding_mask = data_b['padding_mask'][:, sub_seq_start:sub_seq_end].long()

    # tokens = tokens[:, sub_seq_start:sub_seq_end]
    # labels = labels[:, sub_seq_start:sub_seq_end]
    # loss_mask = loss_mask[:, sub_seq_start:sub_seq_end].float()
    # # attention_mask = attention_mask[:, :, sub_seq_start:sub_seq_end, sub_seq_start:sub_seq_end].long()
    # position_ids = position_ids[:, sub_seq_start:sub_seq_end]

    tokens = tokens[:, sub_seq_start:sub_seq_end]
    position_ids = position_ids[:, sub_seq_start:sub_seq_end]

    return tokens, labels, loss_mask, attention_mask, position_ids


def data_post_process(data, data_sampler_state_dict):
    args = get_args()
    if args.data_efficiency_curriculum_learning:
        if 'seqlen_truncate' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_truncate'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_truncate']
            if current_seqlen < args.seq_length:
                data['text'] = data['text'][:, :(current_seqlen+1)].contiguous()
        elif 'seqlen_reshape' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_reshape'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_reshape']
            if current_seqlen < args.seq_length:
                orig_num_token = torch.numel(data['text'])
                reshape_len = (data['text'].size()[1] // (current_seqlen+1)) * (current_seqlen+1)
                data['text'] = torch.cat((data['text'][:, :reshape_len].contiguous().view(-1, current_seqlen+1),
                    data['text'][:, -(current_seqlen+1):]), 0).contiguous()
                num_row = math.ceil(orig_num_token / (current_seqlen+1))
                num_row = min(num_row, data['text'].size()[0])
                if num_row > 1 and num_row % 2 != 0:
                    num_row -= 1
                data['text'] = data['text'][:num_row, :].contiguous()
        else:
            args.data_efficiency_curriculum_learning_seqlen_type = None
    return data


def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    if args.curriculum_learning_legacy and args.curriculum_seqlen < tokens.size()[1]:
        # seqlen-based curriculum learning
        # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        if labels is not None:
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, moe_loss, mos_loss, output_tensor):
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    if is_first_rank() and wandb.run is not None:
        wandb.run.log({'averaged_loss/averaged_loss': averaged_loss[0]})

    if args.mos or args.kd:
        # assert max(args.num_experts) >= 1
        loss = loss + moe_loss + mos_loss
        if args.mos:
            return loss, {'total loss': loss, 'lm loss': averaged_loss[0], 'moe loss': moe_loss, 'mos loss': mos_loss}
        elif args.kd:
            return loss, {'total loss': loss, 'lm loss': averaged_loss[0], 'moe loss': moe_loss, 'kd loss': mos_loss}
        print_rank_0('>>> total loss: {}, lm loss {}, kd loss {}'.format(loss, averaged_loss[0], mos_loss))
    else:
        if max(args.num_experts) <= 1:
            return loss, {'lm loss': averaged_loss[0]}
        else:
            loss = loss + moe_loss
            return loss, {'lm loss': averaged_loss[0], 'moe loss': moe_loss}


def calculate_mos_loss(args, stu_output, teacher_model, tokens, position_ids, attention_mask):
    mos_loss = 0
    alpha = args.kd_alpha_ce
    beta = args.kd_beta_ce
    kd_temp = args.kd_temp
    if teacher_model:
        with torch.no_grad():
            if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
                assert args.curriculum_seqlen is not None
                curriculum_seqlen = args.curriculum_seqlen
                tokens = tokens[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
                # No need to truncate labels as we do not need it for the teacher logits
            tea_output, *tea_other_losses = teacher_model(tokens, position_ids, attention_mask)
            assert stu_output.size() == tea_output.size(), 'teacher and student output should match in size. Student: {}, Teacher: {}, CL seq length {}'.format(stu_output.size(), tea_output.size(), args.curriculum_seqlen)

        student_logits = F.log_softmax(stu_output / kd_temp, dim=2)
        tea_logits = F.softmax(tea_output / kd_temp, dim=2) # The target logits is expected to be probabilities. If we use log_softmax, then we need to set target_log to true when initializing the KLDivLoss.

        mos_loss = kd_temp * kd_temp * nn.KLDivLoss(reduction='batchmean')(student_logits, tea_logits)

        mos_loss = mos_loss.div(args.seq_length) * beta
    return mos_loss


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()
    # Get the batch.
    t0 = time.time()
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()
    if is_first_rank() and wandb.run is not None:
        wandb.run.log({'timers/batch-generator': time.time() - t0})

    if args.data_efficiency_curriculum_learning:
        args.curriculum_seqlen = tokens.size()[1]
        if hasattr(args, 'data_efficiency_curriculum_learning_seqlen_type') and \
            args.data_efficiency_curriculum_learning_seqlen_type == 'seqlen_reshape':
            args.data_efficiency_curriculum_learning_numel = torch.numel(tokens)

    if args.mos or args.kd:
        # The forward func can return either the loss or the logits, depending on whether passing in the labels or not.
        stu_output, *other_losses = model(tokens, position_ids, attention_mask)
        if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
            assert args.curriculum_seqlen is not None
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        output_tensor = mpu.vocab_parallel_cross_entropy(stu_output.contiguous().float(), labels)
    else:
        output_tensor, *other_losses = model(tokens, position_ids, attention_mask,
                                            labels=labels)
    if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    moe_losses = []
    for moe_loss in other_losses:
        if moe_loss is not None:
            moe_losses.append(moe_loss)
    moe_loss = sum(moe_losses) * args.moe_loss_coeff

    mos_loss = 0
    if args.mos or args.kd:
        assert model.training
        if args.teacher_forward and args.teacher_model is not None:
            mos_loss = calculate_mos_loss(args, stu_output,
                args.teacher_model[0], tokens, position_ids, attention_mask)
    
    if is_first_rank() and wandb.run is not None:
        wandb.run.log({'timers/forward_step': time.time() - t0})
    # Output_tensor stores the standard loss, loos_func calculates the total loss.
    return output_tensor, partial(loss_func, loss_mask, moe_loss, mos_loss)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path)
    print_rank_0("> finished creating GPT datasets ...")
    return train_ds, valid_ds, test_ds


def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


def git_ds_info():
    from deepspeed.env_report import main as ds_report
    if is_first_rank():
        ds_report()

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print_rank_0(
        '**** '
        'Git info for Megatron: '
        f'git_hash={git_hash} '
        f'git_branch={git_branch} '
        '****'
    )


def main():
    import deepspeed.comm as dist
    git_ds_info()
    t0 = time.time()
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        data_post_process=data_post_process,
        # wbrun=WBRUN
    )
    dist.log_summary()
    # if is_first_rank() and WBRUN is not None and WBRUN is wandb.run:
    if is_first_rank() and wandb.run is not None:
        wandb.run.log({'pretrain_time/pretrain_time': time.time() - t0})
        wandb.run.finish()


if __name__ == "__main__":
    main()
