"""
Enhanced CLI commands for TrackRealties AI Platform.

This module provides CLI commands that use the EnhancedIngestionPipeline
with JSON chunking, embedding generation, and knowledge graph integration.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from ..core.config import get_settings
from ..data.enhanced_ingestion_pipeline import (EnhancedIngestionPipeline,
                                                IngestionResult)

logger = logging.getLogger(__name__)
settings = get_settings()


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--data-type",
    type=click.Choice(["market", "property", "market_data", "property_listing"]),
    required=True,
    help="Type of data to ingest (market data or property listings)",
)
@click.option("--source", default="cli", help="Source of the data")
@click.option("--batch-size", default=100, type=int, help="Number of records to process in a batch")
@click.option("--dry-run", is_flag=True, help="Validate data without saving to database")
@click.option("--skip-embeddings", is_flag=True, help="Skip generating embeddings")
@click.option("--skip-graph", is_flag=True, help="Skip building knowledge graph")
def enhanced_ingest(file_path, data_type, source, batch_size, dry_run, skip_embeddings, skip_graph):
    """
    Ingest data using the enhanced ingestion pipeline with JSON chunking.

    This command processes data through the EnhancedIngestionPipeline, which:
    1. Chunks JSON data semantically based on its structure
    2. Generates vector embeddings for chunks (unless --skip-embeddings is used)
    3. Saves data and chunks to the database (unless --dry-run is used)
    4. Builds a knowledge graph (unless --skip-graph is used)

    Examples:
        trackrealties enhanced-ingest data.json --data-type market --source zillow
        trackrealties enhanced-ingest properties.json --data-type property --batch-size 50
    """

    async def _ingest():
        try:
            # Load data from file
            click.echo(f"Loading data from {file_path}...")
            with open(file_path, "r") as f:
                data = json.load(f)

            # Check if data is a list
            if not isinstance(data, list):
                data = [data]  # Convert single object to list

            click.echo(f"Loaded {len(data)} records from {file_path}")

            # Create and initialize the pipeline
            click.echo("Initializing enhanced ingestion pipeline...")
            pipeline = EnhancedIngestionPipeline(
                batch_size=batch_size, skip_embeddings=skip_embeddings, skip_graph=skip_graph
            )
            await pipeline.initialize(dry_run=dry_run)

            # Process data based on type
            if dry_run:
                click.echo("Dry run mode - validating data without saving to database")

                if data_type == "market_data":
                    click.echo(f"Validating {len(data)} market data records...")
                    result = await pipeline.validate_market_data(source, data)
                else:  # property
                    click.echo(f"Validating {len(data)} property listings...")
                    result = await pipeline.validate_property_listings(source, data)

                # Display validation results
                _display_validation_results(result)
            else:
                # Process data
                if data_type == "market_data":
                    click.echo(f"Processing {len(data)} market data records...")
                    if skip_embeddings:
                        click.echo("Skipping embedding generation")
                    if skip_graph:
                        click.echo("Skipping graph building")

                    result = await pipeline.ingest_market_data(source, data)
                else:  # property
                    click.echo(f"Processing {len(data)} property listings...")
                    if skip_embeddings:
                        click.echo("Skipping embedding generation")
                    if skip_graph:
                        click.echo("Skipping graph building")

                    result = await pipeline.ingest_property_listings(source, data)

            # Display results
            _display_results(result)

        except Exception as e:
            click.echo(f"Error: {e}")
            logger.exception("Error during enhanced ingestion")
            sys.exit(1)

    asyncio.run(_ingest())


def _display_results(result: IngestionResult):
    """Display the results of an ingestion operation."""
    click.echo("\nIngestion Results:")
    click.echo("=" * 40)
    click.echo(f"Total Records: {result.total}")
    click.echo(f"Processed Records: {result.processed}")
    click.echo(f"Failed Records: {result.failed}")

    if result.chunks_created > 0:
        click.echo(f"Chunks Created: {result.chunks_created}")

    if result.embeddings_generated > 0:
        click.echo(f"Embeddings Generated: {result.embeddings_generated}")

    if result.graph_nodes_created > 0:
        click.echo(f"Graph Nodes Created: {result.graph_nodes_created}")

    if result.errors:
        click.echo("\nErrors:")
        for i, error in enumerate(result.errors[:5]):
            click.echo(f"  {i + 1}. {error}")

        if len(result.errors) > 5:
            click.echo(f"  ... and {len(result.errors) - 5} more errors")

    success_rate = result.processed / result.total * 100 if result.total > 0 else 0
    click.echo(f"\nSuccess Rate: {success_rate:.2f}%")
    click.echo("=" * 40)


def _display_validation_results(result: Dict[str, Any]):
    """Display the results of a validation operation."""
    click.echo("\nValidation Results:")
    click.echo("=" * 40)
    click.echo(f"Total Records: {result['total']}")
    click.echo(f"Valid Records: {result['valid']}")
    click.echo(f"Invalid Records: {result['invalid']}")
    click.echo(f"Success Rate: {result['success_rate'] * 100:.2f}%")

    if result["chunking_stats"]["total_chunks"] > 0:
        click.echo(f"Total Chunks: {result['chunking_stats']['total_chunks']}")
        avg_chunks = result["chunking_stats"]["total_chunks"] / result["total"] if result["total"] > 0 else 0
        click.echo(f"Average Chunks per Record: {avg_chunks:.2f}")

    if result["errors"]:
        click.echo("\nErrors:")
        for i, error in enumerate(result["errors"][:5]):  # Show first 5 errors
            click.echo(f"  {i + 1}. {error}")

        if len(result["errors"]) > 5:
            click.echo(f"  ... and {len(result['errors']) - 5} more errors")

    if result["warnings"]:
        click.echo("\nWarnings:")
        for i, warning in enumerate(result["warnings"][:5]):  # Show first 5 warnings
            click.echo(f"  {i + 1}. {warning}")

        if len(result["warnings"]) > 5:
            click.echo(f"  ... and {len(result['warnings']) - 5} more warnings")
    click.echo("=" * 40)


def register_commands(cli_group):
    """Register enhanced commands with the CLI group."""
    cli_group.add_command(enhanced_ingest)
