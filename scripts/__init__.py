"""Runnable scripts and production orchestration entrypoints.

Marks ``scripts`` as a package so the runbook orchestrators can share
``scripts._pipeline`` via ``python -m scripts.run_weekly`` etc.
"""
