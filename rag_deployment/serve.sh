#!/usr/bin/env bash
echo "Starting Gunicorn (ignore extra args: $@)"
exec gunicorn --bind 0.0.0.0:8080 app:app --timeout=120

