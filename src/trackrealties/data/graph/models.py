from typing import Any, Dict, List


class GraphNode:
    def __init__(self, id: str, labels: List[str], properties: Dict[str, Any]):
        self.id = id
        self.labels = labels
        self.properties = properties

    @property
    def primary_label(self) -> str:
        return self.labels[0] if self.labels else ""


class LocationNode(GraphNode):
    def __init__(self, region_id: str, region_name: str, region_type: str, latitude: float, longitude: float):
        super().__init__(
            id=region_id,
            labels=["Location", region_type.capitalize()],
            properties={
                "region_id": region_id,
                "region_name": region_name,
                "region_type": region_type,
                "latitude": latitude,
                "longitude": longitude,
            },
        )


class PropertyNode(GraphNode):
    def __init__(self, property_id: str, property_type: str, price: int, bedrooms: int):
        super().__init__(
            id=property_id,
            labels=["Property", property_type.replace(" ", "")],
            properties={
                "property_id": property_id,
                "property_type": property_type,
                "price": price,
                "bedrooms": bedrooms,
            },
        )
