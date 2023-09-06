import torch
import intel_extension_for_pytorch as ipex

_utils = torch.utils.data._utils

def _shutdown_workers(self):
    # Called when shutting down this `_MultiProcessingDataLoaderIter`.
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
    # the logic of this function.
    if _utils is None or _utils.python_exit_status is True or _utils.python_exit_status is None:
        # See (2) of the note. If Python is shutting down, do no-op.
        return
    # Normal exit when last reference is gone / iterator is depleted.
    # See (1) and the second half of the note.
    if hasattr(self, "_shutdown") and not self._shutdown:
        self._shutdown = True
        try:
            # Normal exit when last reference is gone / iterator is depleted.
            # See (1) and the second half of the note.

            # Exit `pin_memory_thread` first because exiting workers may leave
            # corrupted data in `worker_result_queue` which `pin_memory_thread`
            # reads from.
            if hasattr(self, '_pin_memory_thread'):
                # Use hasattr in case error happens before we set the attribute.
                self._pin_memory_thread_done_event.set()
                # Send something to pin_memory_thread in case it is waiting
                # so that it can wake up and check `pin_memory_thread_done_event`
                self._worker_result_queue.put((None, None))
                self._pin_memory_thread.join()
                self._worker_result_queue.cancel_join_thread()
                self._worker_result_queue.close()

            # Exit workers now.
            self._workers_done_event.set()
            for worker_id in range(len(self._workers)):
                # Get number of workers from `len(self._workers)` instead of
                # `self._num_workers` in case we error before starting all
                # workers.
                # If we are using workers_status with persistent_workers
                # we have to shut it down because the worker is paused
                if self._persistent_workers or self._workers_status[worker_id]:
                    self._mark_worker_as_unavailable(worker_id, shutdown=True)
            for w in self._workers:
                # We should be able to join here, but in case anything went
                # wrong, we set a timeout and if the workers fail to join,
                # they are killed in the `finally` block.
                w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
            for q in self._index_queues:
                q.cancel_join_thread()
                q.close()
        finally:
            # Even though all this function does is putting into queues that
            # we have called `cancel_join_thread` on, weird things can
            # happen when a worker is killed by a signal, e.g., hanging in
            # `Event.set()`. So we need to guard this with SIGCHLD handler,
            # and remove pids from the C side data structure only at the
            # end.
            #
            # FIXME: Unfortunately, for Windows, we are missing a worker
            #        error detection mechanism here in this function, as it
            #        doesn't provide a SIGCHLD handler.
            if self._worker_pids_set:
                _utils.signal_handling._remove_worker_pids(id(self))
                self._worker_pids_set = False
            for w in self._workers:
                if w.is_alive():
                    # Existing mechanisms try to make the workers exit
                    # peacefully, but in case that we unfortunately reach
                    # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                    # we kill the worker.
                    w.terminate()

def dataloader_init():
    torch.utils.data.dataloader._MultiProcessingDataLoaderIter._shutdown_workers = _shutdown_workers