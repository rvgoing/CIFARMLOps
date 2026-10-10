"""Disable third-party telemetry before importing application dependencies."""
import os

os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['DO_NOT_TRACK'] = 'true'
