import asyncio
import time


class AsyncParallelStreamProcessor:
    """
    Streaming Manager (Producer-Consumer Pattern).
    Decouples ingestion (IO-bound) from processing (CPU-bound)
    allowing their simultaneous execution
    """

    def __init__(self, bloom_filter, batch_size=50_000, max_layers=15, queue_size=10):
        """
        :param bloom_filter: Bloom Filter instance
        :param batch_size: micro-batch dimension
        :param max_layers: Sliding Window limit
        :param queue_size: max queue size
        """
        self.bf = bloom_filter
        self.batch_size = batch_size
        self.max_layers = max_layers
        self.stats = []

        # The Asynchronous Queue is the heart of decoupling
        # maxsize creates "Backpressure": if the calculation is too slow,
        # gently slow down ingestion so as not to run out of RAM.
        self.queue = asyncio.Queue(maxsize=queue_size)

    async def _producer(self, stream_generator):
        """
        INGESTION (Never stops to wait for calculation)
        """
        t_start_io = time.perf_counter()
        buffer = []
        async for item in stream_generator:
            buffer.append(item)
            self.io_time += (time.perf_counter() - t_start_io)
            if len(buffer) >= self.batch_size:
                # putting the batch in the queue. If the queue is full (maxsize),
                # only pauses until the consumer frees up space.
                await self.queue.put(buffer)
                buffer = []  # Reset of local buffer
            t_start_io = time.perf_counter()

        # Management of the last incomplete batch at the end of the stream
        if buffer:
            await self.queue.put(buffer)

        # Termination signal for the Consumer
        await self.queue.put(None)

    def _blocking_batch_injection(self, batch):
        """
        Synchronous wrapper that contains the heavy lifting (CPU-bound).
        It runs in a separate thread to not block the Event Loop.
        """
        if hasattr(self.bf, 'add_batch'):
            self.bf.add_batch(batch)
        else:
            for item in batch:
                self.bf.add(item)

    async def _consumer(self):
        """
        processing (works in background consuming the queue)
        """
        while True:
            # waits for a batch ready in the queue
            batch = await self.queue.get()

            # check of end stream signal
            if batch is None:
                self.queue.task_done()
                break

            start_time = time.perf_counter()

            # Moves blocking work to a system thread.
            # The asyncio Event Loop remains free to continue the _producer.
            t_start_cpu = time.perf_counter()
            await asyncio.to_thread(self._blocking_batch_injection, batch)

            self.cpu_time += (time.perf_counter() - t_start_cpu)
            elapsed = time.perf_counter() - start_time


            # memory management
            self._apply_aging()

            self.stats.append({
                'batch_size': len(batch),
                'time': elapsed,
                'throughput': len(batch) / elapsed
            })

            # Reports to the queue that this batch has completed
            self.queue.task_done()

    async def run_stream(self, stream_generator):
        """
        Main entry point. Starts Producer and Consumer
        at the same time
        """
        self.io_time = 0.0
        self.cpu_time = 0.0
        producer_task = asyncio.create_task(self._producer(stream_generator))
        consumer_task = asyncio.create_task(self._consumer())

        # waits for the conclusion of both the operations
        await asyncio.gather(producer_task, consumer_task)

        print(f"      [ASYNC-PROFILER] Puro I/O Time (Producer): {self.io_time:.2f} s")
        print(f"      [ASYNC-PROFILER] Pure CPU Time (Consumer): {self.cpu_time:.2f} s")

    def _apply_aging(self):
        """
        Aging management
        """
        layer_attr = next((a for a in ['bitmaps', 'shm_blocks', 'filters']
                           if hasattr(self.bf, a)), None)

        if layer_attr:
            layers_list = getattr(self.bf, layer_attr)

            if len(layers_list) > self.max_layers:
                old_layer = layers_list.pop(0)

                try:
                    if hasattr(old_layer, 'close'): old_layer.close()
                    old_layer.unlink()
                except AttributeError:
                    pass

                attrs_to_sync = ['layers_info', 'elements_counts', 'capacities', 'params']
                for attr in attrs_to_sync:
                    metadata_list = getattr(self.bf, attr, None)
                    if isinstance(metadata_list, list) and len(metadata_list) > 0:
                        metadata_list.pop(0)

                # print(f"[AGING] Layer retired. Current layers: {len(layers_list)}")