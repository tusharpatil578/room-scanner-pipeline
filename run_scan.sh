#!/bin/bash
cd /workspaces/room-scanner-pipeline
export PYTHONPATH=/workspaces/room-scanner-pipeline:$PYTHONPATH
python scripts/run_pipeline.py "$@"
