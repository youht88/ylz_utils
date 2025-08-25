
```sh
langgraph dev --host 0.0.0.0 --no-browser
curl -s --request POST \
    --url "http://localhost:2024/runs/stream" \
    --header 'Content-Type: application/json' \
    --data '{
        "assistant_id": "gemini",
        "input": {
            "messages": [
                {
                    "role": "human",
                    "content": "What is LangGraph?"
                }
            ]
        },
        "stream_mode": "updates"  }'
```

### 查询user为youht2的有关利润的文档(仅使用与rag_key为es)
```sh
uv run main.py start --mode rag --rag_key "es" --glob "*.pdf" --rag_metadata '{"user":"youht2"}'   

uv run main.py start --mode rag --rag_key "es" --message "利润" --rag_filter '{"terms":{"metadata.user.keyword":["youht2"]}}'
```