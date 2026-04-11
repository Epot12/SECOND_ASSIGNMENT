import json
import os, sys

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

    current_dir = os.path.dirname(os.path.abspath(__file__))


    project_root = os.path.dirname(current_dir)


    input_path = os.path.join(project_root, "DATA", "cdx-00000")
    output_path = os.path.join(project_root, "DATA", "common_crawl_FULL.txt")

    
    prepare_full_dataset(input_path, output_path)