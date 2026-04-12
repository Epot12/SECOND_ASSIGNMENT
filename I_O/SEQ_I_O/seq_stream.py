import time
import asyncio


class StreamProcessor:
    """
    Universal streaming manager with Micro-batching logic.
    Implementing sequential orchestration: ingestion stops during
    batch processing to ensure data order and consistency.
    """

    def __init__(self, bloom_filter, batch_size=50_000, max_layers=15):
        """
        :param bloom_filter: Instance of any Bloom Filter implementation
        :param batch_size: Micro-batch size before processing
        :param max_layers: Maximum number of layers for Sliding Window (Aging)
        """
        self.bf = bloom_filter
        self.batch_size = batch_size
        self.max_layers = max_layers
        self.buffer = []
        self.stats = []  # measurements for final report


    async def ingest(self, stream_generator):
        """
        Consumes a SYNCHRONOUS data generator (like mmap_url_stream).
        """
        io_time = 0.0
        cpu_time = 0.0

        # measuring I/O
        t_start_io = time.perf_counter()


        for item in stream_generator:
            self.buffer.append(item)

            # accumulating I/O time (reading and decoding mmap)
            io_time += (time.perf_counter() - t_start_io)

            if len(self.buffer) >= self.batch_size:
                # The CPU-bound phase begins (insertion into the Bloom Filter)
                t_start_cpu = time.perf_counter()

                # Even though the loop is synchronous, we process the batch async
                await self.process_current_batch()

                cpu_time += (time.perf_counter() - t_start_cpu)

                # handing over control to the Event Loop for a moment
                # This prevents the mmap loop from blocking other tasks
                await asyncio.sleep(0)

            # Reset the timer for the next reading from the generator
            t_start_io = time.perf_counter()

        # processing last remaining batch (if any)
        if self.buffer:
            t_start_cpu = time.perf_counter()
            await self.process_current_batch()
            cpu_time += (time.perf_counter() - t_start_cpu)

        print(f"      [SEQ-PROFILER] Pure I/O Time: {io_time:.2f} s")
        print(f"      [SEQ-PROFILER] Pure CPU Time: {cpu_time:.2f} s")

    async def process_current_batch(self):
        """
        Injects the batch into the Bloom Filter automatically detecting the
        better strategy (Atomic Batching vs Single Insertion).
        """
        if not self.buffer:
            return

        batch_to_process = self.buffer
        self.buffer = []  # Immediate buffer reset

        start_time = time.perf_counter()

        # Detect whether the filter supports batch (parallel) processing
        # or if it requires sequential element-by-element insertion.
        if hasattr(self.bf, 'add_batch'):
            self.bf.add_batch(batch_to_process)
        else:
            # Fallback for ScalableBloomFilter standard
            for item in batch_to_process:
                self.bf.add(item)

        elapsed = time.perf_counter() - start_time

        # Sliding Window
        self._apply_aging()

        # memorizing measurements
        self.stats.append({
            'batch_size': len(batch_to_process),
            'time': elapsed,
            'throughput': len(batch_to_process) / elapsed
        })

    def _apply_aging(self):

        layer_attr = next((a for a in ['bitmaps', 'shm_blocks', 'filters']
                           if hasattr(self.bf, a)), None)

        if layer_attr:
            layers_list = getattr(self.bf, layer_attr)

            if len(layers_list) > self.max_layers:
                # removing layer
                old_layer = layers_list.pop(0)

                # freeing resources
                try:
                    # first closing local reference, then freeing RAM
                    if hasattr(old_layer, 'close'): old_layer.close()
                    old_layer.unlink()
                except AttributeError:
                    pass

                # removing index 0 from all support lists
                attrs_to_sync = ['layers_info', 'elements_counts', 'capacities', 'params']
                for attr in attrs_to_sync:
                    metadata_list = getattr(self.bf, attr, None)
                    if isinstance(metadata_list, list) and len(metadata_list) > 0:
                        metadata_list.pop(0)

                print(f"[AGING] Layer retired. Current layers: {len(layers_list)}")
