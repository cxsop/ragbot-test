"""
Ingest all markdown chunk files into Neon Postgres rag_chunks table.

Usage:
    python ingest_chunks.py --chunks_dir ./chunks

The chunks directory should contain the extracted markdown files from
taxation_rag_markdown_chunks.zip.
"""

import os
import re
import json
import hashlib
import argparse
import logging
import psycopg2
import psycopg2.extras
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from a markdown file."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
                return meta, body
            except yaml.YAMLError:
                pass
    return {}, text.strip()


def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def ingest_file(cur, filepath: Path, source_name_prefix: str = "taxation") -> bool:
    """Parse one markdown file and upsert into rag_chunks."""
    raw = filepath.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)

    chunk_id = meta.get("chunk_id") or filepath.stem
    title = meta.get("title") or filepath.stem.replace("_", " ").title()
    source_file = meta.get("source_document") or filepath.name
    tags = meta.get("tags") or []
    retrieval_queries = meta.get("retrieval_queries") or []
    source_name = f"{source_name_prefix}_{chunk_id}"

    # Build search_text: title + tags + queries + body
    search_parts = [title] + tags + retrieval_queries + [body]
    search_text = " ".join(str(p) for p in search_parts)

    content_hash = compute_hash(body)

    metadata = {
        "source_section": meta.get("source_section", ""),
        "intended_use": meta.get("intended_use", "RAG chatbot knowledge base"),
        "source_hash_prefix": meta.get("source_hash_prefix", "")
    }

    try:
        cur.execute(
            """
            INSERT INTO rag_chunks
                (source_name, chunk_id, source_file, title, content,
                 content_hash, tags, retrieval_queries, metadata, search_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                source_name        = EXCLUDED.source_name,
                source_file        = EXCLUDED.source_file,
                title              = EXCLUDED.title,
                content            = EXCLUDED.content,
                content_hash       = EXCLUDED.content_hash,
                tags               = EXCLUDED.tags,
                retrieval_queries  = EXCLUDED.retrieval_queries,
                metadata           = EXCLUDED.metadata,
                search_text        = EXCLUDED.search_text,
                updated_at         = now();
            """,
            (
                source_name,
                chunk_id,
                source_file,
                title,
                body,
                content_hash,
                tags,
                retrieval_queries,
                json.dumps(metadata),
                search_text,
            )
        )
        logger.info(f"  ✓ Upserted: {chunk_id} — {title}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed {filepath.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Ingest RAG chunks into Neon Postgres")
    parser.add_argument("--chunks_dir", default="./chunks",
                        help="Directory containing .md chunk files")
    parser.add_argument("--prefix", default="taxation",
                        help="Source name prefix (default: taxation)")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.exists():
        logger.error(f"Directory not found: {chunks_dir}")
        return

    md_files = sorted(chunks_dir.glob("*.md"))
    # Exclude README and combined files
    md_files = [f for f in md_files if f.stem not in ("README", "all_chunks_combined")]

    if not md_files:
        logger.warning("No .md files found.")
        return

    logger.info(f"Found {len(md_files)} chunk files in '{chunks_dir}'")

    if not DATABASE_URL:
        logger.error("DATABASE_URL not set. Check your .env file.")
        return

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    success = 0
    for filepath in md_files:
        if ingest_file(cur, filepath, args.prefix):
            success += 1

    conn.commit()
    cur.close()
    conn.close()

    logger.info(f"\n✅ Ingested {success}/{len(md_files)} chunks successfully.")


if __name__ == "__main__":
    main()
