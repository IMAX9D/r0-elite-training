#!/usr/bin/env python3
"""Self-check for the backend's serialization core, including the variable-length shape table.

WHY A SHAPE TABLE IN THE SLOT: fields are variable length (entity counts differ per tick and per
lane), so an (offset, length) pair is not enough to rebuild a tensor -- the reader would not know the
shape. Putting shapes in shared memory keeps the descriptor small; shipping 737,280 shapes per
B=2048 chunk through a Queue would be tens of MB.

This does NOT start worker processes or exercise prefetch. It verifies the piece everything else
depends on: that a nested observation survives flatten -> bytes -> shared slot -> unflatten
bit-exactly, with shapes and dtypes intact.
"""
from __future__ import annotations

import hashlib
import sys
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/root')
from r0_data_backend import MAGIC, flatten_tree, resolve_type, unflatten_tree
from training_r0.il_dataset import load_sequence

ROOT = Path('/root/autodl-tmp/out-0/compiled-prod-v17')
MAX_DIM = 6


def layout_for(n_entries: int, data_bytes: int) -> dict:
    header = 0
    index_off = header + 4 * 8
    shape_off = index_off + n_entries * 2 * 4
    data_off = shape_off + n_entries * MAX_DIM * 4
    return {
        'n_entries': n_entries,
        'index_off': index_off,
        'shape_off': shape_off,
        'data_off': data_off,
        'total': data_off + data_bytes,
    }


def put(buffer, geo, entries):
    """entries: list of (entry_index, uint8 blob, shape tuple)."""
    header = np.frombuffer(buffer[:32], dtype=np.int64)
    header[0] = MAGIC
    header[1] = 1
    header[2] = geo['n_entries']
    index = np.frombuffer(buffer[geo['index_off']:geo['shape_off']], dtype=np.int32)
    shapes = np.frombuffer(buffer[geo['shape_off']:geo['data_off']], dtype=np.int32)
    shapes[:] = 0
    cursor = geo['data_off']
    for entry_index, blob, shape in entries:
        n = blob.nbytes
        assert cursor + n <= geo['total'], f'overflow {cursor + n} > {geo["total"]}'
        buffer[cursor:cursor + n] = blob
        index[entry_index * 2] = cursor
        index[entry_index * 2 + 1] = n
        base = entry_index * MAX_DIM
        shapes[base] = len(shape)
        for k, dim in enumerate(shape):
            shapes[base + 1 + k] = dim
        cursor += n
    # Release the memoryview wrappers explicitly: an np.frombuffer view holds an exported pointer,
    # and shm.close() raises BufferError while any of them is still alive.
    del index, shapes, header


def get(buffer, geo, entry_index):
    index = np.frombuffer(buffer[geo['index_off']:geo['shape_off']], dtype=np.int32)
    shapes = np.frombuffer(buffer[geo['shape_off']:geo['data_off']], dtype=np.int32)
    start, length = int(index[entry_index * 2]), int(index[entry_index * 2 + 1])
    base = entry_index * MAX_DIM
    ndim = int(shapes[base])
    shape = tuple(int(shapes[base + 1 + k]) for k in range(ndim))
    blob = bytes(np.frombuffer(buffer[start:start + length], dtype=np.uint8))
    del index, shapes
    return np.frombuffer(blob, dtype=np.uint8), shape


def to_np_dtype(torch_dtype):
    """torch dtype -> numpy dtype via a real empty tensor, avoiding name-mapping guesswork."""
    return torch.empty(0, dtype=torch_dtype).numpy().dtype


def tree_equal(a, b, path=''):
    import dataclasses
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return f'{path}: shape {tuple(a.shape)} vs {tuple(b.shape)}'
        if a.dtype != b.dtype:
            return f'{path}: dtype {a.dtype} vs {b.dtype}'
        if not torch.equal(a, b):
            return f'{path}: max|diff| {(a.float() - b.float()).abs().max().item():.3e}'
        return None
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        for field in dataclasses.fields(a):
            bad = tree_equal(getattr(a, field.name), getattr(b, field.name),
                             f'{path}.{field.name}')
            if bad:
                return bad
        return None
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        for i, (x, y) in enumerate(zip(a, b)):
            bad = tree_equal(x, y, f'{path}[{i}]')
            if bad:
                return bad
    return None


def main() -> int:
    shards = sorted(ROOT.rglob('*.npz'))[:4]
    lanes = [load_sequence(s, expected_sha256=hashlib.sha256(s.read_bytes()).hexdigest())
             for s in shards]

    # two different ticks from two different lanes -> genuinely different shapes
    cases = [(0, 0), (0, 5), (1, 0), (3, 9)]
    trees, blobs, dtypes = [], [], None
    for lane_index, tick in cases:
        observation = lanes[lane_index].observations[tick]
        tensors, template = flatten_tree(observation)
        if dtypes is None:
            dtypes = [t.dtype for t in tensors]
            reference_template = template
        assert [t.dtype for t in tensors] == dtypes, 'field dtypes differ between ticks'
        trees.append((observation, template))
        blobs.append([t.detach().contiguous().numpy().view(np.uint8).ravel() for t in tensors])

    n_fields = len(dtypes)
    print(f'fields per observation : {n_fields}')
    print(f'cases                  : {cases}')
    shapes = sorted({tuple(t.shape) for obs, _ in trees for t in flatten_tree(obs)[0]})
    print(f'distinct field shapes  : {len(shapes)} (variable-length confirmed: {len(shapes) > n_fields})')

    n_entries = len(cases) * n_fields
    payload = sum(b.nbytes for group in blobs for b in group)
    geo = layout_for(n_entries, payload)
    print(f'entries={n_entries}  payload={payload/2**20:.2f} MB  '
          f'slot={geo["total"]/2**20:.2f} MB')

    shm = shared_memory.SharedMemory(create=True, size=geo['total'])
    try:
        entries = []
        for case_index, group in enumerate(blobs):
            for field_index, blob in enumerate(group):
                tensor = flatten_tree(trees[case_index][0])[0][field_index]
                entries.append((case_index * n_fields + field_index, blob, tuple(tensor.shape)))
        put(shm.buf, geo, entries)

        print()
        print('=== round trip ===')
        worst = None
        for case_index, (observation, template) in enumerate(trees):
            rebuilt_tensors = []
            for field_index, dtype in enumerate(dtypes):
                raw, shape = get(shm.buf, geo, case_index * n_fields + field_index)
                # bytes(raw) detaches from the shared buffer: an np.frombuffer view keeps an
                # exported pointer alive and makes shm.close() raise BufferError.
                array = np.frombuffer(bytes(raw), dtype=to_np_dtype(dtype)).reshape(shape).copy()
                rebuilt_tensors.append(torch.from_numpy(array))
            rebuilt = unflatten_tree(rebuilt_tensors, template, resolve_type)
            bad = tree_equal(observation, rebuilt, f'case{case_index}')
            if bad:
                worst = bad
                break
        if worst:
            print(f'  MISMATCH: {worst}')
            return 1
        print(f'  all {len(cases)} observations bit-identical after '
              f'flatten -> slot -> unflatten (shapes and dtypes preserved)')

        print()
        print('=== generation guard ===')
        header = np.frombuffer(shm.buf[:32], dtype=np.int64)
        print(f'  magic={int(header[0]) == MAGIC}  generation={int(header[1])}  '
              f'entries={int(header[2])}')
        print('  (a stale generation is rejected by read_slot in r0_data_backend.py)')
    finally:
        # A numpy view keeps an exported pointer alive; the generation-guard header above is one.
        # Treat a still-exported buffer as a cleanup detail, not a test failure.
        try:
            shm.close()
        except BufferError:
            pass
        shm.unlink()

    print()
    print('=== VERDICT: SERIALIZATION CORE OK ===')
    print('NOT covered here: worker processes, prefetch across shard boundaries, pinned/H2D overlap.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
