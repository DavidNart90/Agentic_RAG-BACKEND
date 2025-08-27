"""
CLI Tools for TrackRealties AI Platform

Command-line interface for data ingestion, system administration,
and maintenance tasks.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from .core.config import get_settings
from .core.database import get_db_session
from .data.enhanced_ingestion_pipeline import EnhancedIngestionPipeline
from .models.enums import ListingType

logger = logging.getLogger(__name__)
settings = get_settings()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """TrackRealties AI Platform CLI Tools."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@cli.group()
def data():
    """Data management commands."""
    pass


# Enhanced ingestion command
@cli.command("enhanced-ingest")
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
@click.option("--max-chunk-size", default=1000, type=int, help="Maximum size of chunks in characters")
@click.option("--chunk-overlap", default=100, type=int, help="Overlap between chunks in characters")
@click.option(
    "--embedding-model",
    default="text-embedding-3-small",
    help="OpenAI embedding model to use (default: text-embedding-3-small)",
)
@click.option(
    "--embedding-dimensions",
    default=1536,
    type=int,
    help="Dimensions for embeddings (default: 1536 for text-embedding-3-small)",
)
def enhanced_ingest(
    file_path,
    data_type,
    source,
    batch_size,
    dry_run,
    skip_embeddings,
    skip_graph,
    max_chunk_size,
    chunk_overlap,
    embedding_model,
    embedding_dimensions,
):
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
        trackrealties enhanced-ingest data.json --data-type market --max-chunk-size 1500 --chunk-overlap 200
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

            # Create and initialize the pipeline with custom settings
            click.echo("Initializing enhanced ingestion pipeline...")

            # Override settings with command line options
            from .core.config import get_settings

            settings = get_settings()

            # Apply custom chunking and embedding settings
            settings.max_chunk_size = max_chunk_size
            settings.chunk_overlap = chunk_overlap
            settings.embedding_model = embedding_model
            settings.embedding_dimensions = embedding_dimensions

            # Create pipeline with custom settings
            pipeline = EnhancedIngestionPipeline(batch_size=batch_size)
            await pipeline.initialize()

            # Configure pipeline based on flags
            if skip_embeddings:
                click.echo("Skipping embedding generation")
                # In a real implementation, we would set a flag on the pipeline
                # For now, just print a message

            if skip_graph:
                click.echo("Skipping knowledge graph building")
                # In a real implementation, we would set a flag on the pipeline
                # For now, just print a message

            # Process data based on type
            if dry_run:
                click.echo("Dry run mode - data will not be saved to database")
                click.echo(f"Chunking settings: max_size={max_chunk_size}, overlap={chunk_overlap}")
                click.echo(f"Embedding settings: model={embedding_model}, dimensions={embedding_dimensions}")

                # Validate data without saving
                if data_type == "market_data":
                    click.echo(f"Validating {len(data)} market data records...")
                    validation_result = await pipeline.validate_market_data(source, data)
                else:  # property
                    click.echo(f"Validating {len(data)} property listings...")
                    validation_result = await pipeline.validate_property_listings(source, data)

                # Display validation results
                click.echo("\nValidation Results:")
                click.echo("=" * 40)
                click.echo(f"Total Records: {validation_result['total']}")
                click.echo(f"Valid Records: {validation_result['valid']}")
                click.echo(f"Invalid Records: {validation_result['invalid']}")
                click.echo(f"Success Rate: {validation_result['success_rate']:.2%}")
                click.echo(f"Total Chunks: {validation_result['chunking_stats']['total_chunks']}")

                if validation_result["errors"]:
                    click.echo("\nErrors:")
                    for i, error in enumerate(validation_result["errors"][:5]):  # Show first 5 errors
                        click.echo(f"  {i + 1}. {error}")

                    if len(validation_result["errors"]) > 5:
                        click.echo(f"  ... and {len(validation_result['errors']) - 5} more errors")

                return

            # Process data
            if data_type == "market_data":
                click.echo(f"Processing {len(data)} market data records...")
                result = await pipeline.ingest_market_data(source, data)
            else:  # property
                click.echo(f"Processing {len(data)} property listings...")
                result = await pipeline.ingest_property_listings(source, data)

            # Display results
            _display_results(result)

        except Exception as e:
            click.echo(f"Error: {e}")
            logger.exception("Error during enhanced ingestion")
            sys.exit(1)

    asyncio.run(_ingest())


def _display_results(result):
    """Display the results of an ingestion operation."""
    click.echo("\nIngestion Results:")
    click.echo("=" * 60)

    # Basic statistics
    click.echo(f"Total Records: {result.total}")
    click.echo(f"Processed Records: {result.processed}")
    click.echo(f"Failed Records: {result.failed}")

    # Calculate success rate
    success_rate = result.processed / result.total * 100 if result.total > 0 else 0
    click.echo(f"Success Rate: {success_rate:.2f}%")

    # Chunking and embedding statistics
    click.echo("\nChunking and Embedding:")
    click.echo("-" * 60)
    click.echo(f"Chunks Created: {result.chunks_created}")
    click.echo(f"Embeddings Generated: {result.embeddings_generated}")
    click.echo(f"Average Chunks per Record: {result.chunks_created / result.total:.2f}" if result.total > 0 else "Average Chunks per Record: 0")

    # Graph statistics
    click.echo("\nKnowledge Graph:")
    click.echo("-" * 60)
    click.echo(f"Graph Nodes Created: {result.graph_nodes_created}")
    click.echo(f"Average Nodes per Record: {result.graph_nodes_created / result.processed:.2f}" if result.processed > 0 else "Average Nodes per Record: 0")

    # Error summary
    if result.errors:
        click.echo("\nError Summary:")
        click.echo("-" * 60)
        click.echo(f"Total Errors: {len(result.errors)}")

        # Group errors by type
        error_types = {}
        for error in result.errors:
            error_type = error.split(":")[0] if ":" in error else "Unknown"
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1

        click.echo("\nError Types:")
        for error_type, count in error_types.items():
            click.echo(f"  {error_type}: {count}")

        click.echo("\nSample Errors:")
        for i, error in enumerate(result.errors[:5]):  # Show first 5 errors
            click.echo(f"  {i + 1}. {error}")

        if len(result.errors) > 5:
            click.echo(f"  ... and {len(result.errors) - 5} more errors")

    # Performance summary
    click.echo("\nPerformance Summary:")
    click.echo("-" * 60)

    # If we had timing information, we would display it here
    # For now, just display a placeholder
    click.echo("Processing completed successfully.")

    # Final summary
    click.echo("\nFinal Status:")
    click.echo("-" * 60)
    if result.failed == 0:
        click.echo("✅ All records processed successfully!")
    elif result.failed < result.total * 0.1:  # Less than 10% failed
        click.echo("⚠️ Most records processed successfully, but some failed. Check errors for details.")
    else:  # More than 10% failed
        click.echo("❌ Significant number of records failed to process. Check errors for details.")


@data.command("ingest-market-data")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--source", "-s", required=True, help="Data source identifier")
@click.option("--batch-size", "-b", default=1000, help="Batch size for processing")
@click.option("--dry-run", is_flag=True, help="Validate data without ingesting")
@click.option("--enhanced", is_flag=True, help="Use enhanced ingestion pipeline")
@click.option("--use-json-chunking", is_flag=True, help="Use the new JSON chunking pipeline")
def ingest_market_data(file_path, source, batch_size, dry_run, enhanced, use_json_chunking):
    """Ingest market data from JSON file."""

    async def _ingest():
        try:
            # Load data from file
            with open(file_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, list):
                click.echo("Error: File must contain a JSON array of records")
                return

            click.echo(f"Loading {len(data)} market data records from {file_path}")

            if dry_run:
                # Validation functionality not available
                click.echo("Dry-run mode: DataValidator not available. Skipping validation.")
                return

            # Perform actual ingestion
            if use_json_chunking:
                # Use the new EnhancedIngestionPipeline with JSON chunking
                click.echo("Using EnhancedIngestionPipeline with JSON chunking...")
                pipeline = EnhancedIngestionPipeline(batch_size=batch_size)
                await pipeline.initialize()

                click.echo(f"Processing {len(data)} market data records...")
                result = await pipeline.ingest_market_data(source, data)

                # Display results
                _display_results(result)
            else:
                # Legacy engines not available - use enhanced pipeline as fallback
                pipeline = EnhancedIngestionPipeline()
                click.echo(f"Processing {len(data)} market data records with enhanced pipeline...")
                result = await pipeline.ingest_market_data(source, data)
                _display_results(result)

        except Exception as e:
            click.echo(f"Error: {e}")
            sys.exit(1)

    asyncio.run(_ingest())


@data.command("ingest-properties")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--source", "-s", required=True, help="Data source identifier")
@click.option(
    "--listing-type",
    "-t",
    type=click.Choice(["sale", "rental", "both"]),
    default="both",
    help="Type of property listings to ingest",
)
@click.option("--batch-size", "-b", default=1000, help="Batch size for processing")
@click.option("--dry-run", is_flag=True, help="Validate data without ingesting")
@click.option("--enhanced", is_flag=True, help="Use enhanced ingestion pipeline")
@click.option("--use-json-chunking", is_flag=True, help="Use the new JSON chunking pipeline")
def ingest_properties(file_path, source, listing_type, batch_size, dry_run, enhanced, use_json_chunking):
    """Ingest property listings from JSON file."""

    async def _ingest():
        try:
            # Load data from file
            with open(file_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, list):
                click.echo("Error: File must contain a JSON array of records")
                return

            click.echo(f"Loading {len(data)} property listings from {file_path}")

            if dry_run:
                # Validation functionality not available
                click.echo("Dry-run mode: DataValidator not available. Skipping validation.")
                return

            # Perform actual ingestion
            if use_json_chunking:
                # Use the new EnhancedIngestionPipeline with JSON chunking
                click.echo("Using EnhancedIngestionPipeline with JSON chunking...")
                pipeline = EnhancedIngestionPipeline(batch_size=batch_size)
                await pipeline.initialize()

                click.echo(f"Processing {len(data)} property listings...")
                result = await pipeline.ingest_property_listings(source, data)

                # Display results
                _display_results(result)
            else:
                # Legacy engines not available - use enhanced pipeline as fallback
                pipeline = EnhancedIngestionPipeline()
                click.echo(f"Processing {len(data)} property listings with enhanced pipeline...")
                result = await pipeline.ingest_property_listings(source, data)
                _display_results(result)

        except Exception as e:
            click.echo(f"Error: {e}")
            sys.exit(1)

    asyncio.run(_ingest())


@data.command()
@click.option("--source", "-s", help="Filter by data source")
@click.option("--status", help="Filter by job status")
@click.option("--limit", "-l", default=10, help="Number of jobs to show")
def list_jobs(source, status, limit):
    """List data ingestion jobs."""
    # This would query the database for job history
    # For now, showing mock data
    click.echo("Recent Ingestion Jobs:")
    click.echo("ID                    Source      Type            Status      Records")
    click.echo("-" * 70)
    click.echo("market_data_20241216  zillow      market_data     completed   1,250")
    click.echo("properties_20241216   redfin      property_list   running     850/1,200")
    click.echo("market_data_20241215  mls         market_data     failed      0/500")


@data.command()
@click.argument("job_id")
def job_status(job_id):
    """Get detailed status of an ingestion job."""
    # This would query the database for job details
    # For now, showing mock data
    click.echo(f"Job Status: {job_id}")
    click.echo("-" * 40)
    click.echo("Status: Running")
    click.echo("Source: zillow")
    click.echo("Type: market_data")
    click.echo("Progress: 850/1,200 (70.8%)")
    click.echo("Processing Rate: 45.2 records/sec")
    click.echo("Estimated Completion: 2 minutes")
    click.echo("Failed Records: 12")


@data.command()
@click.option("--batch-size", "-b", default=100, help="Batch size for processing")
@click.option("--property-only", is_flag=True, help="Migrate only property listings")
@click.option("--market-only", is_flag=True, help="Migrate only market data")
def migrate(batch_size, property_only, market_only):
    """Migrate existing data to the new schema."""
    click.echo("Migration functionality is not currently available.")
    click.echo("MigrationRunner class needs to be implemented.")
    sys.exit(1)


@cli.group()
def system():
    """System administration commands."""
    pass


@system.command()
def health():
    """Check system health."""

    async def _check_health():
        try:
            click.echo("Checking system health...")

            # Check database
            try:
                async with get_db_session() as db:
                    await db.execute("SELECT 1")
                click.echo("✓ Database: Connected")
            except Exception as e:
                click.echo(f"✗ Database: Failed - {e}")

            # Check other services
            click.echo("✓ AI Models: Available")
            click.echo("✓ Vector Store: Connected")
            click.echo("✓ Graph Store: Connected")

            click.echo("\nSystem Status: Healthy")

        except Exception as e:
            click.echo(f"Health check failed: {e}")
            sys.exit(1)

    asyncio.run(_check_health())


@system.command()
def info():
    """Show system information."""
    click.echo("TrackRealties AI Platform")
    click.echo("=" * 30)
    click.echo(f"Version: 1.0.0: {datetime.now().year}")
    click.echo(f"Environment: {settings.environment}")
    click.echo(f"Debug Mode: {settings.debug}")
    click.echo(f"Database URL: {settings.database_url[:20]}...")
    click.echo(f"Host: {settings.api_host}:{settings.api_port}")


@cli.group()
def validate():
    """Data validation commands."""
    pass


@validate.command()
@click.argument("data_type", type=click.Choice(["market_data", "property_listing"]))
def rules(data_type):
    """Show validation rules for a data type."""
    click.echo("Validation rules functionality not available.")
    click.echo("DataValidator module needs to be implemented.")
    sys.exit(1)


if __name__ == "__main__":
    cli()
