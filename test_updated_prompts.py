#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to show the updated prompt format without requiring API calls.
"""

import json
import os

def test_updated_prompts():
    """Test the updated prompts with sample data."""
    
    # Sample test data using the 194.json format
    test_graph = {
        "id": "194",
        "question": "To push the carton onto the floor, in which direction should the hand move? A. Backward B. Forward C. Left D. Right",
        "ground_truth_answer": "A",
        "options": ["A. Backward", "B. Forward", "C. Left", "D. Right"],
        "graph": {
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
    }
    
    # Read the updated prompts
    system_prompt_path = os.path.join("sft", "prompts", "rationale_system.txt")
    user_prompt_path = os.path.join("sft", "prompts", "rationale_user.txt")
    
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        
        with open(user_prompt_path, "r", encoding="utf-8") as f:
            user_prompt = f.read()
        
        print("=== UPDATED SYSTEM PROMPT ===")
        print(system_prompt)
        
        print("\n=== UPDATED USER PROMPT ===")
        print(user_prompt)
        
        # Simulate the graph conversion
        def convert_graph_to_name_based(graph):
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            
            # Create a mapping from node ID to node name
            id_to_name = {}
            for node in nodes:
                node_id = node.get("id", "")
                node_name = node.get("name", "")
                if node_id and node_name:
                    id_to_name[node_id] = node_name
            
            # Convert nodes to entities format (remove ID, keep name and description)
            entities = []
            for node in nodes:
                entity = {}
                if "name" in node:
                    entity["name"] = node["name"]
                if "text" in node:
                    entity["description"] = node["text"]
                if entity:  # Only add if it has content
                    entities.append(entity)
            
            # Convert edges to relations format
            relations = []
            for edge in edges:
                from_id = edge.get("from", "")
                to_id = edge.get("to", "")
                from_name = id_to_name.get(from_id, "")
                to_name = id_to_name.get(to_id, "")
                
                if from_name and to_name:
                    relations.append({
                        "from": from_name,
                        "to": to_name
                    })
            
            return {
                "entities": entities,
                "relations": relations
            }
        
        # Convert the graph
        name_based_graph = convert_graph_to_name_based(test_graph["graph"])
        
        print("\n=== CONVERTED GRAPH (ENTITIES & RELATIONS) ===")
        print(json.dumps(name_based_graph, indent=2))
        
        # Show what the final user prompt would look like
        question = test_graph.get("question", "").strip()
        options = test_graph.get("options", None)
        options_text = (json.dumps(options, ensure_ascii=False, indent=2) if options is not None else "null")
        gt_answer = test_graph.get("ground_truth_answer", None)
        
        user_filled = user_prompt.format(
            question=question,
            options=options_text,
            ground_truth_answer=(str(gt_answer) if gt_answer is not None else "null"),
            graph=json.dumps(name_based_graph, ensure_ascii=False, indent=2),
        )
        
        print("\n=== FINAL USER PROMPT (FILLED) ===")
        print(user_filled)
        
        return True
        
    except Exception as e:
        print(f"Error testing updated prompts: {e}")
        return False

if __name__ == "__main__":
    success = test_updated_prompts()
    if success:
        print("\n✅ Updated prompts test completed successfully!")
    else:
        print("\n❌ Test failed!")
