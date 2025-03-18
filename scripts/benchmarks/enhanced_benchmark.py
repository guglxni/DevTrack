from typing import List, Dict, Any
import json

def load_test_data(file_path: str = DEFAULT_TEST_DATA_PATH) -> List[Dict[str, Any]]:
    """Load test data from a JSON file
    
    Args:
        file_path: Path to the test data file
        
    Returns:
        List of test cases with responses and expected scores
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Ensure the expected format
        if not isinstance(data, list):
            print(f"Error: Test data should be a list, got {type(data)}")
            
            # Check if it's in the old format with test_cases
            if isinstance(data, dict) and "test_cases" in data:
                print("Converting from old format with test_cases...")
                data = data["test_cases"]
                
                # Convert to new format
                new_data = []
                for case in data:
                    new_case = {
                        "response": case.get("caregiver_response", ""),
                        "expected_score": case.get("expected_label", "NOT_RATED"),
                        "domain": case.get("domain", ""),
                        "age_months": int(case.get("age_expected", "0-0").split("-")[0]) if "-" in case.get("age_expected", "0-0") else 0,
                        "milestone": case.get("milestone", "")
                    }
                    new_data.append(new_case)
                return new_data
            else:
                return []
            
        # Print stats about the loaded data
        score_counts = {}
        for case in data:
            expected_score = case.get("expected_score", "UNKNOWN")
            if expected_score not in score_counts:
                score_counts[expected_score] = 0
            score_counts[expected_score] += 1
            
        print(f"Loaded {len(data)} test cases with expected scores: {score_counts}")
        return data
    except Exception as e:
        print(f"Error loading test data: {e}")
        return [] 