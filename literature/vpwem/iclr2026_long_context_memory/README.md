# ICLR 2026 Long Context & Memory collection

This directory is generated from the Connected Researchers category:

- `LLM & Agents / Long Context & Memory`
- 49 papers total
- 20 `KV Cache Management`
- 15 `RAG & Retrieval Augmented`
- 14 `Context Extension & Compression`

## Files

- `manifest.json`: complete metadata, abstracts, Chinese guides, source URLs,
  checksums, and local-file status.
- `manifest.tsv`: compact table for spreadsheet import.
- `source_response.json`: original category response from Connected Researchers.
- `arxiv_resolution.json`: OpenReview-to-arXiv resolution cache.
- `pdfs/`: downloaded original paper PDFs.
- `text/`: full text extracted with `pdftotext -layout`, ready for local search
  and literature analysis.

## Refresh or resume

Run from the repository root:

```bash
python3 tools/download_memory_papers.py --extract-text --workers 4
```

The downloader keeps valid existing files and only retries missing papers.

## Current source limitation

43 of 49 PDFs are available locally. Six accepted ICLR papers currently have no
matching arXiv preprint, and OpenReview returns an anti-bot `403` to this machine:

- Channel-Aware Mixed-Precision Quantization for Efficient Long-Context Inference
- CoMem: Compositional Concept-Graph Memory for Vision–Language Adaptation
- Counterfactual Reasoning for Retrieval-Augmented Generation
- From Collapse to Control: Understanding and Extending Context Length in Emerging Hybrid Models via Universal Position Interpolation
- ProtoKV: Long-context Knowledges Are Already Well-Organized Before Your Query
- RESA: Bringing Back What Sparse Attention Ignores with Residual Estimation

Their metadata, abstracts, method summaries, results, and official OpenReview URLs
remain present in the manifests. If the six PDFs are downloaded manually into
`pdfs/` using their OpenReview IDs as filenames, rerunning the command above will
validate them and extract their text automatically.
