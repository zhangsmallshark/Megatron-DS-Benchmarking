import torch
from megatron.core import mpu, tensor_parallel
from megatron import get_args

class SynDataSet():
    
    def __init__(self, batch_size, vocab_size, seq_length):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.step = 0

    def generate(self):
        tokens = torch.randint(low=0, high=self.vocab_size, size=(
            self.batch_size,
            self.seq_length,
        ))
        types = torch.randint(low=0, high=2, size=(
            self.batch_size,
            self.seq_length,
        ))
        sentence_order = torch.randint(low=0, high=2, size=(self.batch_size,))
        loss_mask = torch.randint(low=0, high=2, size=(
            self.batch_size,
            self.seq_length,
        ))
        lm_labels = torch.randint(low=0, high=self.vocab_size, size=(self.batch_size, self.seq_length))
        padding_mask = torch.randint(low=0, high=2, size=(self.batch_size, self.seq_length))
        return dict(text=tokens,
                    types=types,
                    is_random=sentence_order,
                    loss_mask=loss_mask,
                    labels=lm_labels,
                    padding_mask=padding_mask)

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()

def build_syn_train_valid_test_data_provider():
    trainloader, valloader, testloader = None, None, None
    args = get_args()
    #per_gpu_batch = args.global_batch_size //parallel_group_size
    per_gpu_batch = args.global_batch_size // mpu.get_data_parallel_world_size()
    #per_gpu_batch = args.micro_batch_size
    vocab_size = 1024
    ##TODO : before here
    #args.padded_vocab_size = vocab_size ##Hack
    #print('SAGE TRAINING BUILD SYNC DATA args ', args)
    trainloader = SynDataSet(batch_size=per_gpu_batch,
                                 vocab_size=vocab_size,
                                 seq_length = args.seq_length)

    valloader = SynDataSet(batch_size=per_gpu_batch,
                                 vocab_size=vocab_size,
                                 seq_length = args.seq_length)

    testloader = SynDataSet(batch_size=per_gpu_batch,
                                 vocab_size=vocab_size,
                                 seq_length = args.seq_length)

    #return train_data_iterator, valid_data_iterator, test_data_iterator
    do_train = trainloader is not None and args.train_iters > 0
    do_valid = valloader is not None and args.eval_iters > 0
    do_test = testloader is not None and args.eval_iters > 0
    ##SAGE to do; broadcast?
    '''
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()
    '''
    args.do_train = do_train
    args.do_valid = do_valid
    args.do_test = do_test
    return trainloader, valloader, testloader

