"""
Enhanced Graph Integration Script for TrackRealties AI Platform.

This script demonstrates how to use the enhanced graph builder and relationship manager
to create a rich knowledge graph from property listings and market data.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

from ...core.config import get_settings
from .enhanced_graph_builder import EnhancedGraphBuilder
from .enhanced_relationship_manager import EnhancedRelationshipManager

logger = logging.getLogger(__name__)
settings = get_settings()


class GraphIntegrationPipeline:
    """
    Integration pipeline that orchestrates the enhanced graph building process.
    """

    def __init__(self):
        self.graph_builder = EnhancedGraphBuilder()
        self.relationship_manager = EnhancedRelationshipManager()
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize all components."""
        await self.graph_builder.initialize()
        await self.relationship_manager.initialize()
        self.logger.info("Graph integration pipeline initialized")

    async def process_property_with_market_context(
        self, property_data: Dict[str, Any], market_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a single property with full market context integration.

        Args:
            property_data: Property listing data
            market_data: List of market data for context

        Returns:
            Processing results
        """

        results = {
            "property_id": property_data.get("id"),
            "success": False,
            "graph_nodes_created": 0,
            "relationships_created": 0,
            "enhancements": {
                "price_segmentation": False,
                "geographic_hierarchy": False,
                "market_context": False,
                "agent_performance": False,
                "property_type": False,
            },
            "errors": [],
        }

        try:
            # 1. Build integrated graph with enhanced features
            self.logger.info(f"Building integrated graph for property {property_data.get('id')}")

            graph_result = await self.graph_builder.build_integrated_graph(property_data, market_data)

            if graph_result.get("success"):
                results["graph_nodes_created"] = (
                    graph_result.get("property_nodes", 0) + graph_result.get("location_nodes", 0) + graph_result.get("price_segments", 0) + graph_result.get("property_types", 0)
                ) 
                results["relationships_created"] += graph_result.get("relationships", 0)

                # Track enhancements
                results["enhancements"]["price_segmentation"] = graph_result.get("price_segments", 0) > 0
                results["enhancements"]["geographic_hierarchy"] = graph_result.get("location_nodes", 0) > 0
                results["enhancements"]["property_type"] = graph_result.get("property_types", 0) > 0

            else:
                results["errors"].extend(graph_result.get("errors", []))

            # 2. Establish market context relationships
            self.logger.debug("Establishing market context relationships")

            market_context_result = await self.relationship_manager.establish_market_context_relationships(
                property_data, market_data
            )

            results["relationships_created"] += market_context_result.get("relationships_created", 0)
            results["enhancements"]["market_context"] = market_context_result.get("relationships_created", 0) > 0

            # 3. Establish agent performance relationships
            self.logger.debug("Establishing agent performance relationships")

            agent_result = await self.relationship_manager.establish_agent_performance_relationships(
                property_data, market_data
            )

            results["relationships_created"] += agent_result.get("relationships_created", 0)
            results["enhancements"]["agent_performance"] = agent_result.get("relationships_created", 0) > 0

            # 4. Find and establish comparable property relationships
            self.logger.debug("Finding comparable properties")

            # For now, we'll use market data to simulate comparable properties
            # In a real implementation, you'd have multiple properties to compare
            comparable_properties = []
            for market in market_data[:3]:  # Use first 3 market entries as mock comparables
                if market.get("median_price"):
                    # Create a mock comparable property based on market data
                    mock_comparable = {
                        "id": f"comp_{market.get('region_id', 'unknown')}",
                        "price": market.get("median_price"),
                        "squareFootage": property_data.get("squareFootage", 1500),  # Default
                        "bedrooms": property_data.get("bedrooms", 3),  # Default
                        "bathrooms": property_data.get("bathrooms", 2),  # Default
                        "propertyType": property_data.get("propertyType", "Single Family"),
                        "city": market.get("city", property_data.get("city")),
                        "county": market.get("county", property_data.get("county")),
                        "state": market.get("state", property_data.get("state")),
                    }
                    comparable_properties.append(mock_comparable)

            if comparable_properties:
                comp_result = await self.relationship_manager.establish_comparable_property_relationships(
                    property_data, comparable_properties
                )
                results["relationships_created"] += comp_result.get("relationships_created", 0)

            # 5. Check if processing was successful
            if results["graph_nodes_created"] > 0 or results["relationships_created"] > 0:
                results["success"] = True
                self.logger.info(
                    f"Successfully processed property {property_data.get('id')}: "
                    f"{results['graph_nodes_created']} nodes, {results['relationships_created']} relationships"
                )
            else:
                results["errors"].append("No nodes or relationships were created")

        except Exception as e:
            error_msg = f"Error processing property {property_data.get('id')}: {str(e)}"
            results["errors"].append(error_msg)
            self.logger.error(error_msg)

        return results

    async def process_batch(
        self, properties: List[Dict[str, Any]], market_data: List[Dict[str, Any]], batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Process a batch of properties with market context.

        Args:
            properties: List of property data
            market_data: List of market data
            batch_size: Number of properties to process in parallel

        Returns:
            Batch processing results
        """

        batch_results = {
            "total_properties": len(properties),
            "processed_successfully": 0,
            "failed": 0,
            "total_nodes_created": 0,
            "total_relationships_created": 0,
            "enhancement_summary": {
                "price_segmentation": 0,
                "geographic_hierarchy": 0,
                "market_context": 0,
                "agent_performance": 0,
                "property_type": 0,
            },
            "errors": [],
            "property_results": [],
        }

        self.logger.info(f"Processing batch of {len(properties)} properties")

        # Process properties in batches
        for i in range(0, len(properties), batch_size):
            batch = properties[i: i + batch_size]

            # Process batch concurrently
            tasks = [self.process_property_with_market_context(prop, market_data) for prop in batch]

            batch_batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            for result in batch_batch_results:
                if isinstance(result, Exception):
                    batch_results["failed"] += 1
                    batch_results["errors"].append(f"Exception: {str(result)}")
                elif isinstance(result, dict):
                    batch_results["property_results"].append(result)

                    if result.get("success"):
                        batch_results["processed_successfully"] += 1
                        batch_results["total_nodes_created"] += result.get("graph_nodes_created", 0)
                        batch_results["total_relationships_created"] += result.get("relationships_created", 0)

                        # Track enhancements
                        for enhancement, enabled in result.get("enhancements", {}).items():
                            if enabled:
                                batch_results["enhancement_summary"][enhancement] += 1
                    else:
                        batch_results["failed"] += 1
                        batch_results["errors"].extend(result.get("errors", []))

        # Calculate success rate
        success_rate = (
            (batch_results["processed_successfully"] / batch_results["total_properties"]) * 100
            if batch_results["total_properties"] > 0
            else 0
        )

        self.logger.info(
            f"Batch processing complete: {batch_results['processed_successfully']}/{batch_results['total_properties']} "
            f"properties processed successfully ({success_rate:.1f}%)"
        )

        return batch_results

    async def analyze_market_coverage(self, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze market data coverage and suggest improvements.

        Args:
            market_data: List of market data

        Returns:
            Coverage analysis results
        """

        coverage_analysis = {
            "total_markets": len(market_data),
            "geographic_coverage": {"states": set(), "counties": set(), "cities": set()},
            "data_quality": {"complete_records": 0, "missing_price": 0, "missing_location": 0, "missing_metrics": 0},
            "market_segments": {"affordable": 0, "moderate": 0, "premium": 0, "luxury": 0},
            "recommendations": [],
        }

        for market in market_data:
            # Geographic coverage
            if market.get("state"):
                coverage_analysis["geographic_coverage"]["states"].add(market["state"])
            if market.get("county"):
                coverage_analysis["geographic_coverage"]["counties"].add(market["county"])
            if market.get("city"):
                coverage_analysis["geographic_coverage"]["cities"].add(market["city"])

            # Data quality
            has_price = market.get("median_price") is not None and market.get("median_price") > 0
            has_location = market.get("location") is not None
            has_key_metrics = all(
                market.get(metric) is not None for metric in ["inventory_count", "days_on_market", "months_supply"]
            )

            if has_price and has_location and has_key_metrics:
                coverage_analysis["data_quality"]["complete_records"] += 1

            if not has_price:
                coverage_analysis["data_quality"]["missing_price"] += 1
            if not has_location:
                coverage_analysis["data_quality"]["missing_location"] += 1
            if not has_key_metrics:
                coverage_analysis["data_quality"]["missing_metrics"] += 1

            # Market segments
            if has_price:
                price = market["median_price"]
                if price < 200000:
                    coverage_analysis["market_segments"]["affordable"] += 1
                elif price < 400000:
                    coverage_analysis["market_segments"]["moderate"] += 1
                elif price < 1000000:
                    coverage_analysis["market_segments"]["premium"] += 1
                else:
                    coverage_analysis["market_segments"]["luxury"] += 1

        # Convert sets to counts
        coverage_analysis["geographic_coverage"] = {
            k: len(v) for k, v in coverage_analysis["geographic_coverage"].items()
        }

        # Generate recommendations
        if coverage_analysis["data_quality"]["missing_price"] > 0:
            coverage_analysis["recommendations"].append(
                f"{coverage_analysis['data_quality']['missing_price']} markets missing price data"
            )

        if coverage_analysis["data_quality"]["missing_location"] > 0:
            coverage_analysis["recommendations"].append(
                f"{coverage_analysis['data_quality']['missing_location']} markets missing location data"
            )

        completeness_rate = (
            (coverage_analysis["data_quality"]["complete_records"] / coverage_analysis["total_markets"]) * 100
            if coverage_analysis["total_markets"] > 0
            else 0
        )

        if completeness_rate < 80:
            coverage_analysis["recommendations"].append(
                f"Data completeness is {completeness_rate:.1f}% - consider improving data quality"
            )

        return coverage_analysis

    async def close(self) -> None:
        """Close all connections."""
        await self.relationship_manager.close()
        self.logger.info("Graph integration pipeline closed")


# Usage example function
async def example_usage():
    """Example of how to use the enhanced graph integration."""

    # Sample property data (from your uploaded sample)
    sample_property = {
        "id": "333-Florida-St,-San-Antonio,-TX-78210",
        "formattedAddress": "333 Florida St, San Antonio, TX 78210",
        "city": "San Antonio",
        "state": "TX",
        "zipCode": "78210",
        "county": "Bexar",
        "latitude": 29.407553,
        "longitude": -98.481689,
        "propertyType": "Multi-Family",
        "bedrooms": 3,
        "bathrooms": 4,
        "squareFootage": 1800,
        "lotSize": 7841,
        "yearBuilt": 1927,
        "status": "Active",
        "price": 750000,
        "listingType": "Standard",
        "daysOnMarket": 120,
        "listingAgent": {"name": "Lisa Blanco", "phone": "2102162696", "email": "lisablancorealtor@gmail.com"},
        "listingOffice": {"name": "Compass RE Texas", "phone": "2103616000"},
    }

    # Sample market data (from your uploaded sample)
    sample_market_data = [
        {
            "location": "San Antonio, TX metro area",
            "date": "2025-06-23 00:00:00",
            "median_price": 425000.0,
            "inventory_count": 26520.0,
            "sales_volume": 1089.0,
            "new_listings": 934.0,
            "days_on_market": 47.0,
            "months_supply": 24.35,
            "price_per_sqft": 197.14,
            "source": "redfin",
            "region_type": "metro",
            "region_id": "19124",
            "city": "San Antonio",
            "state": "TX",
            "county": "Bexar",
        },
        {
            "location": "Bexar County, TX",
            "date": "2025-06-23 00:00:00",
            "median_price": 380000.0,
            "inventory_count": 15200.0,
            "sales_volume": 645.0,
            "new_listings": 520.0,
            "days_on_market": 42.0,
            "months_supply": 23.56,
            "price_per_sqft": 185.50,
            "source": "redfin",
            "region_type": "county",
            "region_id": "2048",
            "city": "San Antonio",
            "state": "TX",
            "county": "Bexar",
        },
    ]

    # Initialize and run the pipeline
    pipeline = GraphIntegrationPipeline()

    try:
        await pipeline.initialize()

        print("=== Enhanced Graph Integration Example ===")

        # 1. Analyze market coverage
        print("\n1. Analyzing market coverage...")
        coverage = await pipeline.analyze_market_coverage(sample_market_data)
        print("Market Coverage Analysis:")
        print(f"  - Total markets: {coverage['total_markets']}")
        print(f"  - States covered: {coverage['geographic_coverage']['states']}")
        print(f"  - Counties covered: {coverage['geographic_coverage']['counties']}")
        print(f"  - Complete records: {coverage['data_quality']['complete_records']}")

        if coverage["recommendations"]:
            print("  Recommendations:")
            for rec in coverage["recommendations"]:
                print(f"    - {rec}")

        # 2. Process single property with market context
        print("\n2. Processing property with market context...")
        result = await pipeline.process_property_with_market_context(sample_property, sample_market_data)

        print("Property Processing Results:")
        print(f"  - Property ID: {result['property_id']}")
        print(f"  - Success: {result['success']}")
        print(f"  - Nodes created: {result['graph_nodes_created']}")
        print(f"  - Relationships created: {result['relationships_created']}")

        print("  Enhancements applied:")
        for enhancement, enabled in result["enhancements"].items():
            status = "✓" if enabled else "✗"
            print(f"    {status} {enhancement.replace('_', ' ').title()}")

        if result["errors"]:
            print(f"  Errors: {result['errors']}")

        # 3. Demonstrate batch processing
        print("\n3. Demonstrating batch processing...")
        properties_batch = [sample_property]  # In real usage, you'd have multiple properties

        batch_result = await pipeline.process_batch(properties_batch, sample_market_data)

        print("Batch Processing Results:")
        print(f"  - Total properties: {batch_result['total_properties']}")
        print(f"  - Processed successfully: {batch_result['processed_successfully']}")
        print(f"  - Failed: {batch_result['failed']}")
        print(f"  - Total nodes created: {batch_result['total_nodes_created']}")
        print(f"  - Total relationships created: {batch_result['total_relationships_created']}")

        print("  Enhancement Summary:")
        for enhancement, count in batch_result["enhancement_summary"].items():
            print(f"    - {enhancement.replace('_', ' ').title()}: {count} properties")

    except Exception as e:
        print(f"Error in example usage: {e}")

    finally:
        await pipeline.close()


# CLI integration function
async def run_enhanced_graph_pipeline(properties_file: str, market_data_file: str, batch_size: int = 10):
    """
    Run the enhanced graph pipeline with data from files.

    Args:
        properties_file: Path to JSON file containing property data
        market_data_file: Path to JSON file containing market data
        batch_size: Number of properties to process in parallel
    """

    pipeline = GraphIntegrationPipeline()

    try:
        # Load data
        with open(properties_file, "r") as f:
            properties = json.load(f)

        with open(market_data_file, "r") as f:
            market_data = json.load(f)

        if not isinstance(properties, list):
            properties = [properties]

        if not isinstance(market_data, list):
            market_data = [market_data]

        print(f"Loaded {len(properties)} properties and {len(market_data)} market records")

        # Initialize pipeline
        await pipeline.initialize()

        # Analyze market coverage first
        print("\nAnalyzing market coverage...")
        coverage = await pipeline.analyze_market_coverage(market_data)

        print("Market Coverage:")
        print(f"  - {coverage['geographic_coverage']['states']} states")
        print(f"  - {coverage['geographic_coverage']['counties']} counties")
        print(f"  - {coverage['geographic_coverage']['cities']} cities")
        print(f"  - {coverage['data_quality']['complete_records']}/{coverage['total_markets']} complete records")

        # Process properties in batches
        print(f"\nProcessing {len(properties)} properties in batches of {batch_size}...")

        results = await pipeline.process_batch(properties, market_data, batch_size)

        # Print summary
        success_rate = (
            (results["processed_successfully"] / results["total_properties"]) * 100
            if results["total_properties"] > 0
            else 0
        )

        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total Properties: {results['total_properties']}")
        print(f"Successfully Processed: {results['processed_successfully']} ({success_rate:.1f}%)")
        print(f"Failed: {results['failed']}")
        print(f"Total Graph Nodes Created: {results['total_nodes_created']}")
        print(f"Total Relationships Created: {results['total_relationships_created']}")

        print("\n=== ENHANCEMENT SUMMARY ===")
        for enhancement, count in results["enhancement_summary"].items():
            percentage = (count / results["total_properties"]) * 100 if results["total_properties"] > 0 else 0
            print(f"{enhancement.replace('_', ' ').title()}: {count} properties ({percentage:.1f}%)")

        if results["errors"]:
            print("\n=== ERRORS ===")
            for i, error in enumerate(results["errors"][:10], 1):  # Show first 10 errors
                print(f"{i}. {error}")

            if len(results["errors"]) > 10:
                print(f"... and {len(results['errors']) - 10} more errors")

        return results

    except Exception as e:
        print(f"Error running enhanced graph pipeline: {e}")
        return None

    finally:
        await pipeline.close()


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())
