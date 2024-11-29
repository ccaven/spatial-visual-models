
class Vocabulary:
    def __init__(self):
        self.str_to_idx = dict()
        self.idx_to_str = list()
        self.range_starts = dict()
        self.range_lens = dict()
    
    def __len__(self) -> int:
        return len(self.idx_to_str)

    def add_token(self, label: str):
        next_idx = len(self.idx_to_str)
        self.idx_to_str.append(label)
        self.str_to_idx[label] = next_idx
    
    def get_range_len(self, label: str):
        return self.range_lens[label]

    def get_range_start(self, label: str):
        return self.range_starts[label]

    def get_token(self, label: str):
        return self.str_to_idx[label]
    
    def add_token_range(self, label: str, quantity: int):
        self.range_starts[label] = len(self.idx_to_str)
        self.range_lens[label] = quantity
        for i in range(quantity):
            self.add_token(f"{label}_{i}")
    
    def get_token_in_range(self, label: int, idx: int):
        #assert label in self.range_starts, f"Requested range {label} not in ranges"
        #assert idx < self.range_lens[label], f"Requested index {idx} for range {label} out of bounds {self.range_lens[label]}"
        return self.range_starts[label] + idx

    def get_index_in_range(self, idx: int):
        for label in self.range_starts.keys():
            start = self.range_starts[label]
            length = self.range_lens[label]
            if idx >= start and idx < start + length:
                return idx - start
        raise IndexError(f"Requested index {idx} not inside a range.")
