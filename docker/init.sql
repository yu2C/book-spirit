CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(32) NOT NULL,
    question TEXT NOT NULL,
    backend VARCHAR(32),
    top_k INT,
    result_ids JSONB,
    latency_ms DOUBLE PRECISION,
    retrieval_mode VARCHAR(32),
    use_rerank BOOLEAN,
    filters JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs (created_at DESC);
