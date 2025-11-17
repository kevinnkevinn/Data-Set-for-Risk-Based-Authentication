"""
I/O utilities: chunked CSV reader and simple memory monitor.
"""
from typing import Iterator, Optional, List
import pandas as pd
import psutil, time

def iter_csv_chunks(path: str, chunksize: int = 100_000, usecols: Optional[List[str]] = None) -> Iterator[pd.DataFrame]:
    """Stream CSV in chunks to save memory."""
    for df in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
        yield df

def monitor_memory(note: str = "") -> None:
    vm = psutil.virtual_memory()
    print(f"[MEM] {note} used={vm.used/1e9:.2f}GB avail={vm.available/1e9:.2f}GB percent={vm.percent}%")

def consolidate_chunks(path: str, limit_rows: int = 500_000) -> pd.DataFrame:
    """Load limited rows from chunk iterator."""
    dfs = []
    total = 0
    for chunk in iter_csv_chunks(path):
        dfs.append(chunk)
        total += len(chunk)
        if total >= limit_rows:
            break
    return pd.concat(dfs, ignore_index=True)
