# Dataset Provenance & Reproducibility

To reproduce the benchmarks, the following source dataset is required.

## Source Metadata
- **Provider**: Common Crawl
- **Crawl Archive**: CC-MAIN-2024-10 (February/March 2024)
- **File Subset**: `cdx-00000.gz` (Index File)
- **Format**: CDXJ (JSON-based index)
- **Source URL**: [https://commoncrawl.org/](https://commoncrawl.org/)
- **Direct Download Link**: https://data.commoncrawl.org/cc-index/collections/CC-MAIN-2024-10/indexes/cdx-00000.gz

## File Integrity (Verification)
To ensure the same data is used, verify the file size after extraction:
- **Raw File Name**: `cdx-00000` (or `cdx-00000.txt`)
- **Uncompressed Size**: 5,789.54 MB (approx. 5.65 GB)
- **Total Records**: ~10,000,000+ (depending on specific crawl subset)

## Preparation Steps
1. Download the `.gz` file from the link above.
2. Extract it into the `DATA/` folder.
3. Run `DATA_MANAGEMENT/data_loader.py` to generate `common_crawl_FULL.txt`.

### Verification (SHA-256 Checksum)
 3D9F2F2BAEFF3DB20262B6E5580A8BA34CECBD3742D0C898B484D5ACF5C476B1