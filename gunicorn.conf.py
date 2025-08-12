# gunicorn.conf.py
import os
import multiprocessing

# Basic configuration
workers = 1  # Keep low for student/free-tier
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 180
keepalive = 5
max_requests = 1000
max_requests_jitter = 50

# Binding
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'data-analyst-agent'

# Worker settings
worker_tmp_dir = '/dev/shm'
preload_app = True

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
