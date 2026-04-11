
import asyncio
import aiofiles


# --- REAL-WORLD DATA GENERATOR (COMMON CRAWL) ---
async def common_crawl_stream(file_path, total_items):
    """
    Asynchronously reads real-world URLs from the Common Crawl dataset.
    This replaces synthetic data with actual Disk I/O operations.
    """
    count = 0
    async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
        async for line in f:
            if count >= total_items:
                break

            url = line.strip()
            if url:
                yield url
                count += 1


                if count % 2000 == 0:
                    await asyncio.sleep(0)