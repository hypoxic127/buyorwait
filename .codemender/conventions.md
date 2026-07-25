# BuyOrWait Project Coding Conventions for CodeMender

## 1. BigQuery & Database Operations
- **Parameterized Queries**: All SQL queries MUST use `QueryJobConfig(query_parameters=...)` for parameterized execution. Never concatenate untrusted strings into SQL.
- **Resource Cleanup**: All temporary BigQuery staging tables MUST be deleted in a `finally` block or explicit cleanup handler, regardless of success or failure.
- **Concurrency & Table Names**: Multi-task Cloud Run Jobs MUST incorporate both `CLOUD_RUN_TASK_INDEX` and a unique execution token (e.g. `RUN_UUID`) into staging table names to eliminate collisions.

## 2. External API Resilience (Steam Web API)
- **HTTP Session Reuse**: External HTTP requests MUST reuse a `requests.Session()` with connection pooling (`HTTPAdapter`).
- **Jittered Backoff for 429**: When rate limited (HTTP 429), use exponential backoff with random jitter (`(2 ** p) + random.uniform(0.5, 1.5)`).
- **Timeouts**: All external network calls MUST explicitly set connection and read timeouts.

## 3. Code Sanitization & Privacy
- **Output Sanitization**: Review text presented to users MUST be sanitized via `clean_text()` to strip Kaomoji, ASCII art, and BBCode formatting tags.
