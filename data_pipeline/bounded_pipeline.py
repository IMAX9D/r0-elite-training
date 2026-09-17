"""Bounded two-stage scheduling: no worker port waits for post-processing."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import deque


def ordered_prefetch(items, load, depth=0):
    """Overlap bounded CPU loading with consumption without reordering a lane."""
    if depth < 0:raise ValueError('prefetch depth must be nonnegative')
    if not depth:
        for item in items:yield load(item)
        return
    source=iter(items);pending=deque()
    with ThreadPoolExecutor(max_workers=1) as pool:
        for _ in range(depth):
            try:pending.append(pool.submit(load,next(source)))
            except StopIteration:break
        while pending:
            value=pending.popleft().result()
            try:pending.append(pool.submit(load,next(source)))
            except StopIteration:pass
            yield value


def two_stage(items, capture, post, *, capture_workers, post_workers, backlog,
              proceed=lambda value: True):
    """Yield (item, capture_result, post_result) in completion order.

    At most capture_workers + post_workers + backlog items are in flight.
    Returning a failed capture never sends it to post. Exceptions propagate;
    callers keep their existing per-item quarantine policy. The executor drains
    already submitted work on exit, so no task is abandoned with an owned port.
    """
    if capture_workers < 1 or post_workers < 1 or backlog < 0:
        raise ValueError('invalid pipeline capacity')
    source=iter(items);pending={};exhausted=False
    limit=capture_workers+post_workers+backlog
    with ThreadPoolExecutor(max_workers=capture_workers) as collectors, \
         ThreadPoolExecutor(max_workers=post_workers) as processors:
        while pending or not exhausted:
            while not exhausted and len(pending)<limit:
                try:item=next(source)
                except StopIteration:exhausted=True;break
                pending[collectors.submit(capture,item)]=('capture',item,None)
            if not pending:break
            finished,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:
                stage,item,captured=pending.pop(future);result=future.result()
                if stage=='capture' and proceed(result):
                    pending[processors.submit(post,item,result)]=('post',item,result)
                else:
                    yield item,(result if stage=='capture' else captured),(None if stage=='capture' else result)
