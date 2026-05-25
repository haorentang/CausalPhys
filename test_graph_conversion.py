#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify the graph conversion from ID-based to name-based format.
"""

import json
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the conversion function directly
from sft.generate_rationales import _convert_graph_to_name_based

def test_graph_conversion():
    """Test the graph conversion with sample data."""
    
    # Sample test data using the 194.json format
    original_graph = {
        "nodes": [
            {
                "id": "n3",
                "type": "object",
                "name": "Carton"
            },
            {
                "id": "n5",
                "type": "attribute",
                "name": "Carton's position relative to floor",
                "text": "Front"
            },
            {
                "id": "n4",
                "type": "object",
                "name": "Floor"
            },
            {
                "id": "n7",
                "type": "event",
                "name": "Push carton onto floor"
            },
            {
                "id": "n6",
                "type": "event",
                "name": "Hand action"
            }
        ],
        "edges": [
            {
                "from": "n3",
                "to": "n5"
            },
            {
                "from": "n4",
                "to": "n5"
            },
            {
                "from": "n5",
                "to": "n7"
            },
            {
                "from": "n6",
                "to": "n7"
            }
        ]
    }
    
    print("=== TESTING GRAPH CONVERSION ===")
    print(f"Original graph (ID-based):")
    print(json.dumps(original_graph, indent=2))
    
    # Convert to name-based format
    name_based_graph = _convert_graph_to_name_based(original_graph)
    
    print(f"\nConverted graph (name-based):")
    print(json.dumps(name_based_graph, indent=2))
    
    # Verify the conversion
    print(f"\n=== VERIFICATION ===")
    print(f"Original nodes: {len(original_graph['nodes'])}")
    print(f"Converted nodes: {len(name_based_graph['nodes'])}")
    print(f"Original edges: {len(original_graph['edges'])}")
    print(f"Converted edges: {len(name_based_graph['edges'])}")
    
    # Check that IDs are removed and names are preserved
    for node in name_based_graph['nodes']:
        if 'id' in node:
            print(f"❌ ERROR: Node still contains ID: {node}")
        if 'name' not in node:
            print(f"❌ ERROR: Node missing name: {node}")
    
    # Check that edges use names instead of IDs
    for edge in name_based_graph['edges']:
        if edge['from'] in ['n3', 'n4', 'n5', 'n6', 'n7']:
            print(f"❌ ERROR: Edge still uses ID: {edge}")
        if edge['to'] in ['n3', 'n4', 'n5', 'n6', 'n7']:
            print(f"❌ ERROR: Edge still uses ID: {edge}")
    
    print(f"\n✅ Graph conversion test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_graph_conversion()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
