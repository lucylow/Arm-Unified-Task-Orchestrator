#!/usr/bin/env python3
"""
Tiny Planner for AutoRL - Converts perception outputs to action plans.
This lightweight planner runs on-device and generates JSON action sequences
based on detected UI elements.
"""

import json
import sys


def plan_from_perception(perception):
    """
    Generate an action plan from perception outputs.
    
    Args:
        perception: Dictionary containing:
            - labels: List of detected UI element labels
            - bboxes: List of bounding boxes [x, y, width, height]
            - scores: Optional confidence scores
    
    Returns:
        List of action dictionaries with action type and parameters
    """
    plan = []
    
    labels = perception.get("labels", [])
    bboxes = perception.get("bboxes", [])
    scores = perception.get("scores", [])
    
    print(f"Planning from perception: {len(labels)} labels detected")
    
    # Rule-based planning logic
    if "Login" in labels or "login_button" in labels:
        idx = labels.index("Login") if "Login" in labels else labels.index("login_button")
        if idx < len(bboxes):
            bbox = bboxes[idx]
            plan.append({
                "action": "tap",
                "target": "login_button",
                "coordinates": {
                    "x": bbox[0] + bbox[2] // 2,
                    "y": bbox[1] + bbox[3] // 2
                },
                "confidence": scores[idx] if idx < len(scores) else 1.0
            })
    
    if "Search" in labels or "search_bar" in labels:
        idx = labels.index("Search") if "Search" in labels else labels.index("search_bar")
        plan.append({
            "action": "tap",
            "target": "search_bar",
            "coordinates": {
                "x": bboxes[idx][0] + bboxes[idx][2] // 2,
                "y": bboxes[idx][1] + bboxes[idx][3] // 2
            } if idx < len(bboxes) else {"x": 0, "y": 0}
        })
        plan.append({
            "action": "type",
            "text": "search query",
            "target": "search_bar"
        })
    
    if "Button" in labels:
        button_indices = [i for i, label in enumerate(labels) if label == "Button"]
        for idx in button_indices[:3]:  # Limit to first 3 buttons
            if idx < len(bboxes):
                plan.append({
                    "action": "tap",
                    "target": f"button_{idx}",
                    "coordinates": {
                        "x": bboxes[idx][0] + bboxes[idx][2] // 2,
                        "y": bboxes[idx][1] + bboxes[idx][3] // 2
                    }
                })
    
    # Fallback: wait and observe
    if not plan:
        plan.append({
            "action": "wait",
            "duration": 1,
            "reason": "No actionable elements detected"
        })
    
    # Add final wait for stability
    plan.append({
        "action": "wait",
        "duration": 0.5,
        "reason": "Allow UI to stabilize"
    })
    
    return plan


def main():
    """Demo execution with sample perception data."""
    print("=" * 60)
    print("AutoRL Tiny Planner - Demo")
    print("=" * 60)
    
    # Sample perception outputs
    test_cases = [
        {
            "name": "Login Screen",
            "perception": {
                "labels": ["Login", "Username", "Password"],
                "bboxes": [[100, 200, 150, 50], [100, 100, 200, 40], [100, 150, 200, 40]],
                "scores": [0.95, 0.92, 0.91]
            }
        },
        {
            "name": "Search Interface",
            "perception": {
                "labels": ["Search", "Button", "Menu"],
                "bboxes": [[50, 50, 300, 40], [400, 50, 80, 40], [500, 50, 60, 40]],
                "scores": [0.98, 0.89, 0.87]
            }
        },
        {
            "name": "Empty Screen",
            "perception": {
                "labels": [],
                "bboxes": [],
                "scores": []
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'─' * 60}")
        print(f"Test Case: {test_case['name']}")
        print(f"{'─' * 60}")
        
        perception = test_case['perception']
        print(f"Input: {json.dumps(perception, indent=2)}")
        
        plan = plan_from_perception(perception)
        
        print(f"\nGenerated Plan ({len(plan)} actions):")
        print(json.dumps(plan, indent=2))
    
    print("\n" + "=" * 60)
    print("✅ Planner demo completed")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        # JSON mode: read from stdin
        perception = json.loads(sys.stdin.read())
        plan = plan_from_perception(perception)
        print(json.dumps(plan, indent=2))
    else:
        # Demo mode
        main()
