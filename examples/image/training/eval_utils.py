from typing import Callable, Iterable, Iterator, Tuple


def default_batch_size(batch) -> int:
    if isinstance(batch, (tuple, list)):
        return len(batch[0])
    return len(batch)


def iter_batches_until_target(
    data_loader: Iterable,
    target_samples: int,
    batch_size_fn: Callable = default_batch_size,
    test_run: bool = False,
) -> Iterator[Tuple[int, object]]:
    if target_samples <= 0:
        return

    produced = 0
    data_iter_step = 0
    iterator = iter(data_loader)
    while produced < target_samples:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise ValueError("Evaluation data loader is empty.") from error
        yield data_iter_step, batch
        produced += int(batch_size_fn(batch))
        data_iter_step += 1
        if test_run:
            break
