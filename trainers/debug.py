import traceback
from torch.utils.data._utils.collate import default_collate

def debug_collate(batch):
    """
    A custom collate that first checks if any item is None
    or if any dict key is None, then calls default_collate.
    """
    for i, b in enumerate(batch):
        if b is None:
            print(f"[DEBUG] Batch item {i} is None. Full batch = {batch}")
            traceback.print_stack()
            raise ValueError("Found None in batch item")

        if isinstance(b, dict):
            # Check each key in the dict
            for k, v in b.items():
                if v is None:
                    print(f"[DEBUG] Batch item {i}, key '{k}' is None")
                    print(f"Full item = {b}")
                    traceback.print_stack()
                    raise ValueError("Found None in dictionary item")

    # If no issues, proceed with normal collate
    return default_collate(batch)
