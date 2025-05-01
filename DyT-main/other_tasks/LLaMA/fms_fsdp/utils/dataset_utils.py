import os
import glob
import torch
import pyarrow as pa

import torch

class DistributedDataset(torch.utils.data.IterableDataset):
    def __init__(self, root_dir, rank, world_size, batch_size, seq_length, bos_token, eos_token):
        super().__init__()
        self.root_dir = root_dir
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.bos_token = bos_token
        self.eos_token = eos_token

        # the root directory should contain files named rank_0.arrow, rank_1.arrow, ...
        # the total number of files should be divisible by the world size
        # each rank will read num_files_per_rank files such that rank 0 reads rank_0.arrow, rank_{0 + world_size}.arrow, ...
        num_files = len(glob.glob(os.path.join(self.root_dir, "rank_*.arrow")))
        assert num_files % self.world_size == 0
        self.num_files_per_rank = num_files // self.world_size
        self.readers = [
            pa.ipc.open_file(pa.memory_map(
                os.path.join(self.root_dir, f"rank_{self.rank + i * self.world_size}.arrow")
            )) 
            for i in range(self.num_files_per_rank)
        ]

        # state variables
        self.buffer = []
        self.current_reader_idx = 0
        self.current_batch_idx = 0
        
    def __iter__(self):
        for reader_idx in range(self.current_reader_idx, len(self.readers)):
            self.current_reader_idx = reader_idx
            
            reader = self.readers[reader_idx]
            for batch_idx in range(self.current_batch_idx, reader.num_record_batches):
                self.current_batch_idx = batch_idx

                sample = reader.get_batch(batch_idx)['input_ids'].to_pylist()
                self.buffer += [self.bos_token] + sample + [self.eos_token]

                while len(self.buffer) >= self.batch_size * self.seq_length + 1:
                    yield torch.LongTensor(self.buffer[:self.batch_size * self.seq_length]).reshape(self.batch_size, self.seq_length), \
                          torch.LongTensor(self.buffer[1:self.batch_size * self.seq_length + 1]).reshape(self.batch_size, self.seq_length)
                    self.buffer = self.buffer[self.batch_size * self.seq_length:]
            
            self.current_batch_idx = 0

    def state_dict(self):
        """Return a dictionary containing the state of the dataset."""
        return {
            'buffer': self.buffer,
            'current_reader_idx': self.current_reader_idx,
            'current_batch_idx': self.current_batch_idx,
        }

    def load_state_dict(self, state_dict):
        """Load the state of the dataset."""
        self.buffer = state_dict['buffer']
        self.current_reader_idx = state_dict['current_reader_idx']
        self.current_batch_idx = state_dict['current_batch_idx']
