"""Reproducible next-day Nifty equity research engine."""

from .ingest import IngestionArtifacts, ingest_from_config

__all__ = ["IngestionArtifacts", "ingest_from_config"]
