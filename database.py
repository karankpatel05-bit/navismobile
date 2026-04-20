"""
Database layer for Navis LLM training data.
Uses PostgreSQL (Supabase) when DATABASE_URL is set, falls back to JSON file otherwise.
"""
import os
import json

DATABASE_URL = os.getenv('DATABASE_URL', '')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_FILE = os.path.join(BASE_DIR, 'training_data.json')

# ── PostgreSQL helpers ─────────────────────────────────────────
_pool = None

def _get_conn():
    """Get a database connection from the pool."""
    global _pool
    if _pool is None:
        import psycopg2
        from psycopg2 import pool
        _pool = pool.SimpleConnectionPool(1, 5, DATABASE_URL)
    return _pool.getconn()

def _put_conn(conn):
    """Return a connection to the pool."""
    if _pool:
        _pool.putconn(conn)

def _init_db():
    """Create the qa_pairs table if it doesn't exist."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL
                )
            """)
        conn.commit()
    finally:
        _put_conn(conn)

def _seed_from_json():
    """Seed the database from training_data.json if the table is empty."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM qa_pairs")
            count = cur.fetchone()[0]
            if count == 0 and os.path.exists(TRAINING_DATA_FILE):
                with open(TRAINING_DATA_FILE, 'r') as f:
                    data = json.load(f)
                for qa in data.get('qa_pairs', []):
                    cur.execute(
                        "INSERT INTO qa_pairs (question, answer) VALUES (%s, %s)",
                        (qa['question'], qa['answer'])
                    )
                conn.commit()
                print(f"✅ Seeded {len(data.get('qa_pairs', []))} Q&A pairs into database")
    finally:
        _put_conn(conn)

# ── JSON file helpers (fallback) ───────────────────────────────
def _json_load():
    if os.path.exists(TRAINING_DATA_FILE):
        with open(TRAINING_DATA_FILE, 'r') as f:
            return json.load(f)
    return {"qa_pairs": []}

def _json_save(data):
    with open(TRAINING_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# ── Public API (auto-selects DB or JSON) ───────────────────────
def use_database():
    """Check whether we're using PostgreSQL."""
    return bool(DATABASE_URL)

def init_storage():
    """Initialize storage — create tables + seed if using DB, create JSON file if not."""
    if use_database():
        try:
            _init_db()
            _seed_from_json()
            print("✅ Database connected (PostgreSQL)")
        except Exception as e:
            print(f"⚠️  Database init failed: {e}")
            raise
    else:
        if not os.path.exists(TRAINING_DATA_FILE):
            _json_save({"qa_pairs": []})
        print("📁 Using local JSON file for training data")

def load_training_data():
    """Load all Q&A pairs."""
    if use_database():
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, question, answer FROM qa_pairs ORDER BY id")
                rows = cur.fetchall()
            return {"qa_pairs": [{"id": r[0], "question": r[1], "answer": r[2]} for r in rows]}
        finally:
            _put_conn(conn)
    else:
        return _json_load()

def add_qa_pair(question, answer):
    """Add a new Q&A pair. Returns the new ID."""
    if use_database():
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO qa_pairs (question, answer) VALUES (%s, %s) RETURNING id",
                    (question, answer)
                )
                new_id = cur.fetchone()[0]
            conn.commit()
            return new_id
        finally:
            _put_conn(conn)
    else:
        data = _json_load()
        new_id = max((qa.get('id', 0) for qa in data['qa_pairs']), default=0) + 1
        data['qa_pairs'].append({'question': question, 'answer': answer, 'id': new_id})
        _json_save(data)
        return new_id

def delete_qa_pair(qa_id):
    """Delete a Q&A pair by ID."""
    if use_database():
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM qa_pairs WHERE id = %s", (qa_id,))
            conn.commit()
        finally:
            _put_conn(conn)
    else:
        data = _json_load()
        data['qa_pairs'] = [qa for qa in data['qa_pairs'] if qa.get('id') != qa_id]
        _json_save(data)
