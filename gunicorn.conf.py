# gunicorn.conf.py
import multiprocessing

workers = 1  # keep low for student / free-tier
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 180
keepalive = 5
loglevel = "info"
accesslog = "-"
errorlog = "-"
