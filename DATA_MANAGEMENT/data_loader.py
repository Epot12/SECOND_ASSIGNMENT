import mmap


def mmap_url_stream(file_path, total_items):
    """
    Simulates a high-performance stream by mapping the file to memory.
    """
    count = 0
    with open(file_path, "rb") as f:  # opening in binary mode for maximum speed
        # mapping the file (access=ACCESS_READ is used to avoid corrupting the data)
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            # using readline() directly on the mmap object
            for line in iter(mm.readline, b""):
                if count >= total_items:
                    break

                # decoding only here
                url = line.decode('utf-8').strip()
                if url:
                    yield url
                    count += 1

