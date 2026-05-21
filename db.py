import os
import logging
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def get_connection():
    """Create and return a Postgres connection."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set.")
    conn = psycopg2.connect(DATABASE_URL)
    return conn


def search_chunks(query: str, top_k: int = 4) -> list[dict]:
    """
    Search for relevant chunks using Postgres full-text search (tsvector).
    Falls back to ILIKE if full-text returns no results.
    Returns a list of chunk dicts.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Primary: Full-text search using tsvector with ranking
        fts_query = """
            SELECT
                id,
                chunk_id,
                source_name,
                source_file,
                title,
                content,
                tags,
                retrieval_queries,
                metadata,
                ts_rank(search_vector, plainto_tsquery('english', %s)) AS rank
            FROM rag_chunks
            WHERE search_vector @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s;
        """
        cur.execute(fts_query, (query, query, top_k))
        results = cur.fetchall()

        # Fallback: ILIKE search if full-text returns nothing
        if not results:
            logger.info(f"FTS returned no results for '{query}', trying ILIKE fallback.")
            ilike_query = """
                SELECT
                    id,
                    chunk_id,
                    source_name,
                    source_file,
                    title,
                    content,
                    tags,
                    retrieval_queries,
                    metadata,
                    0.0 AS rank
                FROM rag_chunks
                WHERE
                    content ILIKE %s
                    OR title ILIKE %s
                    OR search_text ILIKE %s
                ORDER BY title
                LIMIT %s;
            """
            pattern = f"%{query}%"
            cur.execute(ilike_query, (pattern, pattern, pattern, top_k))
            results = cur.fetchall()

        cur.close()
        return [dict(row) for row in results]

    except Exception as e:
        logger.error(f"Database search error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def get_all_chunk_ids() -> list[str]:
    """Utility: return all chunk IDs in the DB."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT chunk_id FROM rag_chunks ORDER BY chunk_id;")
        rows = cur.fetchall()
        cur.close()
        return [r[0] for r in rows]
    finally:
        if conn:
            conn.close()
