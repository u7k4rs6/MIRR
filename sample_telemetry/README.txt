Optional JSONL for env.telemetry.FileTelemetry — one JSON object per line with a top-level "metrics" key
matching IncidentResponseEnv observation metrics shape (service -> cpu, latency_ms, ...).

Used by docker-compose volume mount at /data for demos; the Gradio app uses the live simulator by default.
