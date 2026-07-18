import numba
import os
from tqdm import tqdm

@numba.njit(cache=True)
def query_prefix_sum(arr_prefix_sum, start, stop):
    """Returns the sum of elements in the range [start, stop) of arr."""
    return arr_prefix_sum[stop - 1] - arr_prefix_sum[start - 1] if start > 0 else arr_prefix_sum[stop - 1]

def compact_logging_enabled() -> bool:
    return os.environ.get("LNQ_COMPACT_LOG", "0") == "1"

def get_progress_bar(total: int, desc: str):
    return tqdm(
        total=total,
        desc=desc,
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
        leave=False if compact_logging_enabled() else True,
        mininterval=5.0 if compact_logging_enabled() else 0.1,
        disable=False,
    )
