import json


def prepare_full_dataset(input_file, output_file):
    print(f"Start total extraction of URLs from the file {input_file}...")

    extracted_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            try:
                parts = line.split(' ', 2)
                if len(parts) < 3:
                    continue

                metadata = json.loads(parts[2])
                url = metadata.get('url')

                if url:
                    f_out.write(url + '\n')
                    extracted_count += 1

                    if extracted_count % 1_000_000 == 0:
                        print(f"Extracted {extracted_count:,} URL...")

            except (json.JSONDecodeError, IndexError):
                continue

    print(f"Finished! FULL dataset saved in: {output_file}")
    print(f"Total clean URLs ready for benchmarking: {extracted_count:,}")


if __name__ == "__main__":
    # Update file names with correct paths in data/ folder
    prepare_full_dataset("data/cdx-00000.txt", "data/common_crawl_FULL.txt")