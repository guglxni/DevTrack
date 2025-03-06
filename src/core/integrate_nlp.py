#!/usr/bin/env python3

"""
Integration Script for Advanced NLP Module

This script integrates the advanced NLP module with the ASD Assessment API.
It locates the API's assessment engine module and adds the advanced response
analysis capabilities to improve scoring accuracy.
"""

import os
import sys
import shutil
import re
import importlib.util
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("integrate-nlp")

def find_api_directory():
    """Find the API application directory."""
    # Common locations to check
    potential_dirs = [
        ".",  # Current directory
        "./app",
        "./api",
        "./src",
    ]
    
    # Files that indicate we found the API directory
    indicator_files = [
        "app.py",
        "assessment_engine.py",
        "engine.py",
        "api.py"
    ]
    
    for directory in potential_dirs:
        if not os.path.isdir(directory):
            continue
        
        # Check for indicator files
        for file in indicator_files:
            if os.path.isfile(os.path.join(directory, file)):
                logger.info(f"Found API directory: {os.path.abspath(directory)}")
                return os.path.abspath(directory)
    
    # If we can't find it in common locations, search recursively
    logger.info("Searching for API directory recursively...")
    for root, dirs, files in os.walk(".", topdown=True):
        if any(file in files for file in indicator_files):
            logger.info(f"Found API directory: {os.path.abspath(root)}")
            return os.path.abspath(root)
    
    # If still not found, ask the user
    logger.warning("Could not automatically locate API directory.")
    user_dir = input("Please enter the path to the API directory: ").strip()
    if os.path.isdir(user_dir):
        return os.path.abspath(user_dir)
    else:
        logger.error(f"Invalid directory: {user_dir}")
        sys.exit(1)

def find_assessment_engine(api_dir):
    """Find the assessment engine file."""
    # Possible names for the assessment engine file
    engine_files = [
        "assessment_engine.py",
        "engine.py",
        "scoring_engine.py",
        "score.py"
    ]
    
    for file in engine_files:
        path = os.path.join(api_dir, file)
        if os.path.isfile(path):
            logger.info(f"Found assessment engine: {path}")
            return path
    
    # If not in the main directory, check subdirectories
    for root, dirs, files in os.walk(api_dir):
        for file in files:
            if file in engine_files:
                path = os.path.join(root, file)
                logger.info(f"Found assessment engine: {path}")
                return path
    
    logger.error("Could not find assessment engine file.")
    return None

def create_backup(file_path):
    """Create a backup of the original file."""
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    return backup_path

def import_advanced_nlp():
    """Import the advanced NLP module."""
    try:
        # Try to find the advanced_nlp.py file
        if os.path.isfile("advanced_nlp.py"):
            advanced_nlp_path = "advanced_nlp.py"
        else:
            # Search recursively
            for root, dirs, files in os.walk("."):
                if "advanced_nlp.py" in files:
                    advanced_nlp_path = os.path.join(root, "advanced_nlp.py")
                    break
            else:
                logger.error("Could not find advanced_nlp.py")
                return None
        
        # Import the advanced_nlp module
        spec = importlib.util.spec_from_file_location("advanced_nlp", advanced_nlp_path)
        advanced_nlp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(advanced_nlp)
        
        logger.info("Successfully imported advanced_nlp module")
        return advanced_nlp
    except Exception as e:
        logger.error(f"Error importing advanced_nlp: {str(e)}")
        return None

def integrate_nlp_with_engine(engine_path, advanced_nlp):
    """Integrate the advanced NLP module with the assessment engine."""
    try:
        # Read the engine file
        with open(engine_path, 'r') as f:
            engine_code = f.read()
        
        # Check if already integrated
        if "advanced_nlp" in engine_code:
            logger.info("Advanced NLP is already integrated with the engine.")
            return True
        
        # Find the scoring function - common patterns
        scoring_patterns = [
            r'def\s+score_response\s*\(',
            r'def\s+analyze_response\s*\(',
            r'def\s+calculate_score\s*\(',
            r'def\s+process_response\s*\('
        ]
        
        scoring_function = None
        for pattern in scoring_patterns:
            match = re.search(pattern, engine_code)
            if match:
                # Find the function body
                start_pos = match.start()
                # Find the end of the function by matching indentation
                lines = engine_code[start_pos:].split('\n')
                function_lines = [lines[0]]
                indent = len(lines[1]) - len(lines[1].lstrip())
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() and len(line) - len(line.lstrip()) <= indent:
                        if not line.strip().startswith(('elif', 'else', 'except', 'finally')):
                            break
                    function_lines.append(line)
                
                scoring_function = '\n'.join(function_lines)
                break
        
        if not scoring_function:
            logger.error("Could not find the scoring function in the engine.")
            return False
        
        # Create enhanced engine code
        enhanced_engine = engine_code.replace(
            scoring_function,
            get_enhanced_scoring_function(scoring_function)
        )
        
        # Add imports at the top
        import_lines = '\nimport importlib.util\nimport os\n'
        if not 'import importlib.util' in enhanced_engine:
            # Find the last import line
            import_section = re.search(r'^(import .*?\n|from .*? import .*?\n)+', enhanced_engine, re.MULTILINE)
            if import_section:
                enhanced_engine = enhanced_engine[:import_section.end()] + import_lines + enhanced_engine[import_section.end():]
            else:
                # Add after shebang and docstring if present
                match = re.search(r'^(#!.*\n)?(\"{3}|\'{3}).*?(\"{3}|\'{3})', enhanced_engine, re.DOTALL)
                if match:
                    enhanced_engine = enhanced_engine[:match.end()] + '\n' + import_lines + enhanced_engine[match.end():]
                else:
                    enhanced_engine = import_lines + enhanced_engine
        
        # Add the advanced NLP initialization code
        nlp_init_code = '''
# Advanced NLP integration
def _load_advanced_nlp():
    """Load the advanced NLP module if available."""
    try:
        # Try to find the advanced_nlp.py file
        nlp_paths = [
            os.path.join(os.path.dirname(__file__), "advanced_nlp.py"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "advanced_nlp.py"),
            "advanced_nlp.py"
        ]
        
        for path in nlp_paths:
            if os.path.isfile(path):
                spec = importlib.util.spec_from_file_location("advanced_nlp", path)
                advanced_nlp = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(advanced_nlp)
                # Set up optimizations for Apple Silicon
                advanced_nlp.setup_optimizations()
                return advanced_nlp
        return None
    except Exception as e:
        print(f"Warning: Could not load advanced NLP module: {str(e)}")
        return None

# Try to load advanced NLP module
_advanced_nlp = _load_advanced_nlp()
'''
        
        # Find the best place to insert the NLP init code
        class_match = re.search(r'class\s+\w+.*?:', enhanced_engine)
        if class_match:
            insert_pos = class_match.start()
        else:
            # If no class found, insert before the first function
            func_match = re.search(r'def\s+\w+\s*\(', enhanced_engine)
            insert_pos = func_match.start() if func_match else 0
        
        enhanced_engine = enhanced_engine[:insert_pos] + nlp_init_code + enhanced_engine[insert_pos:]
        
        # Write the enhanced engine
        with open(engine_path, 'w') as f:
            f.write(enhanced_engine)
        
        logger.info("Successfully integrated advanced NLP with the assessment engine.")
        return True
    
    except Exception as e:
        logger.error(f"Error integrating NLP with engine: {str(e)}")
        return False

def get_enhanced_scoring_function(original_function):
    """Create an enhanced version of the scoring function that uses advanced NLP."""
    
    # Extract function name and parameters
    function_pattern = r'def\s+(\w+)\s*\((.*?)\):'
    match = re.search(function_pattern, original_function)
    
    if not match:
        logger.error("Could not parse function signature.")
        return original_function
        
    function_name = match.group(1)
    function_params = match.group(2)
    
    # Find indentation
    indent_match = re.search(r'\n(\s+)', original_function)
    indent = indent_match.group(1) if indent_match else '    '
    
    # Create enhanced function
    enhanced_function = f"""def {function_name}({function_params}):
    \"\"\"
    Enhanced scoring function for developmental milestones using advanced NLP.
    This is an enhanced version that uses more sophisticated language processing.
    \"\"\"
    # Import necessary modules
    import os
    import re
    from enum import Enum
    import importlib.util
    
    # Define Score enum if it doesn't exist in the current scope
    if 'Score' not in locals() and 'Score' not in globals():
        try:
            # Try to import Score from a module
            for module_name in ['assessment_engine', 'engine', 'models']:
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, 'Score'):
                        Score = module.Score
                        break
                except ImportError:
                    continue
            else:
                # Define our own Score enum
                class Score(Enum):
                    CANNOT_DO = 0
                    LOST_SKILL = 1
                    EMERGING = 2
                    WITH_SUPPORT = 3
                    INDEPENDENT = 4
                    NOT_RATED = -1
        except Exception as e:
            print(f"Error setting up Score enum: {{e}}")
            # Fallback to simple integers
            class Score:
                CANNOT_DO = 0
                LOST_SKILL = 1
                EMERGING = 2
                WITH_SUPPORT = 3
                INDEPENDENT = 4
                NOT_RATED = -1
    
    # Try to use the advanced NLP module
    try:
        # Check if we already imported the analyzer
        if 'nlp_analyzer' not in globals():
            # Try importing the AdvancedResponseAnalyzer
            advanced_nlp_path = os.path.join(os.path.dirname(__file__), 'advanced_nlp.py')
            if os.path.exists(advanced_nlp_path):
                spec = importlib.util.spec_from_file_location("advanced_nlp", advanced_nlp_path)
                advanced_nlp = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(advanced_nlp)
                
                # Create analyzer instance
                global nlp_analyzer
                nlp_analyzer = advanced_nlp.AdvancedResponseAnalyzer()
            else:
                # Look in other common locations
                module_paths = [
                    'advanced_nlp.py',
                    './advanced_nlp.py',
                    '../advanced_nlp.py',
                    './utils/advanced_nlp.py',
                    './nlp/advanced_nlp.py'
                ]
                
                for path in module_paths:
                    if os.path.exists(path):
                        spec = importlib.util.spec_from_file_location("advanced_nlp", path)
                        advanced_nlp = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(advanced_nlp)
                        
                        # Create analyzer instance
                        global nlp_analyzer
                        nlp_analyzer = advanced_nlp.AdvancedResponseAnalyzer()
                        break
        
        # Special handling for complex phrases that might be misinterpreted
        response_text_lower = response_text.lower() if 'response_text' in locals() else milestone_response.lower() if 'milestone_response' in locals() else ""
        
        # 1. Handle phrases like "no, not yet" - should be CANNOT_DO (0), not LOST_SKILL (1)
        if re.search(r'\\b(no|not),?\\s+(yet|yet started|started yet)', response_text_lower):
            print(f"Enhanced NLP detected 'not yet' pattern - scoring as CANNOT_DO")
            return Score.CANNOT_DO
        
        # 2. Handle phrases with complex negation like "not at all, he has never walked independently"
        if re.search(r'not at all', response_text_lower) and len(response_text_lower.split('not at all')) > 1:
            explanation = response_text_lower.split('not at all')[1]
            if any(term in explanation for term in ['never', "hasn't", "doesn't", 'cannot', "can't"]):
                print(f"Enhanced NLP detected complex negation with explanation - scoring as CANNOT_DO")
                return Score.CANNOT_DO

        # 3. Clear negation indicators
        if (re.search(r'\\bnever\\b', response_text_lower) or
            re.search(r'\\bno\\b', response_text_lower) or 
            re.search(r'\\bnot\\b', response_text_lower)) and not re.search(r'not only|not just', response_text_lower):
            print(f"Enhanced NLP detected clear negation - scoring as CANNOT_DO")
            return Score.CANNOT_DO
        
        # Use the advanced analyzer if available
        if 'nlp_analyzer' in globals():
            result = nlp_analyzer.analyze_response(
                milestone if 'milestone' in locals() else milestone_behavior if 'milestone_behavior' in locals() else "unknown",
                response_text if 'response_text' in locals() else milestone_response if 'milestone_response' in locals() else ""
            )
            
            score_value = result.get('score', None)
            if score_value is not None:
                print(f"Advanced NLP analysis: {{result.get('score_label')}}")
                # Convert score value to Score enum
                for score_type in Score:
                    if hasattr(score_type, 'value') and score_type.value == score_value:
                        return score_type
                    elif not hasattr(score_type, 'value') and getattr(Score, score_type) == score_value:
                        return getattr(Score, score_type)
                
                # Direct integer fallback
                if score_value == 0:
                    return Score.CANNOT_DO
                elif score_value == 1:
                    return Score.LOST_SKILL
                elif score_value == 2:
                    return Score.EMERGING
                elif score_value == 3:
                    return Score.WITH_SUPPORT
                elif score_value == 4:
                    return Score.INDEPENDENT
                else:
                    return Score.NOT_RATED
    
    except Exception as e:
        print(f"Error using advanced NLP: {{e}}") 
    
    # Fallback to original function logic
    {indent}# Original function implementation
    {original_function.split(':', 1)[1].lstrip()}
    """
    
    return enhanced_function

def test_integration():
    """Test the integration by running a simple scoring example."""
    try:
        # Import the assessment engine
        sys.path.append(".")
        sys.path.append("./app")
        
        # Try to find the assessment engine module
        engine_module = None
        engine_module_names = ["enhanced_assessment_engine", "assessment_engine", "engine", "scoring_engine", "score"]
        
        for module_name in engine_module_names:
            try:
                engine_module = importlib.import_module(module_name)
                logger.info(f"Successfully imported module: {module_name}")
                break
            except ImportError:
                continue
        
        if not engine_module:
            logger.error("Could not import any assessment engine module for testing.")
            return False
        
        # Find the AssessmentEngine class
        engine_class = None
        for attr_name in dir(engine_module):
            if attr_name.endswith('Engine') and hasattr(getattr(engine_module, attr_name), 'score_response'):
                engine_class = getattr(engine_module, attr_name)
                logger.info(f"Found engine class: {attr_name}")
                break
        
        if not engine_class:
            logger.error("Could not find engine class with score_response method.")
            return False
            
        # Create an instance of the engine
        engine = engine_class()
        logger.info(f"Successfully created instance of {engine_class.__name__}")
        
        # Define comprehensive test cases covering all response types
        test_cases = [
            # CANNOT_DO (0) test cases
            {
                'milestone': 'walks independently',
                'response': 'no, not yet',
                'expected_score': 0,
                'expected_label': 'CANNOT_DO'
            },
            {
                'milestone': 'walks independently',
                'response': 'not at all',
                'expected_score': 0,
                'expected_label': 'CANNOT_DO'
            },
            {
                'milestone': 'walks independently',
                'response': 'not at all, he has never walked independently',
                'expected_score': 0,
                'expected_label': 'CANNOT_DO'
            },
            {
                'milestone': 'walks independently',
                'response': 'no',
                'expected_score': 0,
                'expected_label': 'CANNOT_DO'
            },
            
            # LOST_SKILL (1) test cases
            {
                'milestone': 'walks independently',
                'response': 'he used to walk but has regressed',
                'expected_score': 1,
                'expected_label': 'LOST_SKILL'
            },
            {
                'milestone': 'walks independently',
                'response': 'previously could walk but not anymore',
                'expected_score': 1,
                'expected_label': 'LOST_SKILL'
            },
            
            # EMERGING (2) test cases
            {
                'milestone': 'walks independently',
                'response': 'sometimes, but not consistently',
                'expected_score': 2,
                'expected_label': 'EMERGING'
            },
            {
                'milestone': 'walks independently',
                'response': 'occasionally, it depends on the day',
                'expected_score': 2,
                'expected_label': 'EMERGING'
            },
            {
                'milestone': 'walks independently',
                'response': 'trying to, but still developing',
                'expected_score': 2,
                'expected_label': 'EMERGING'
            },
            
            # WITH_SUPPORT (3) test cases
            {
                'milestone': 'walks independently',
                'response': 'he can walk with help',
                'expected_score': 3,
                'expected_label': 'WITH_SUPPORT'
            },
            {
                'milestone': 'walks independently',
                'response': 'only when I hold his hand',
                'expected_score': 3,
                'expected_label': 'WITH_SUPPORT'
            },
            {
                'milestone': 'walks independently',
                'response': 'needs support to walk',
                'expected_score': 3,
                'expected_label': 'WITH_SUPPORT'
            },
            
            # INDEPENDENT (4) test cases
            {
                'milestone': 'walks independently',
                'response': 'yes, he walks independently',
                'expected_score': 4,
                'expected_label': 'INDEPENDENT'
            },
            {
                'milestone': 'walks independently',
                'response': 'absolutely, he does this all the time',
                'expected_score': 4,
                'expected_label': 'INDEPENDENT'
            },
            
            # Problematic cases that previously failed
            {
                'milestone': 'selects and brings familiar objects from another room when asked',
                'response': 'no, he does not perform the action',
                'expected_score': 0,
                'expected_label': 'CANNOT_DO'
            },
            {
                'milestone': 'selects and brings familiar objects from another room when asked',
                'response': 'not sure',
                'expected_score': 0,  # This should be CANNOT_DO, not LOST_SKILL
                'expected_label': 'CANNOT_DO'
            }
        ]
        
        # Run the tests
        logger.info(f"Running {len(test_cases)} test cases...")
        
        # Define a mapping from numeric scores to labels
        score_labels = {
            0: "CANNOT_DO",
            1: "LOST_SKILL",
            2: "EMERGING",
            3: "WITH_SUPPORT",
            4: "INDEPENDENT",
            -1: "NOT_RATED"
        }
        
        # Track test results
        passed = 0
        failed = 0
        
        for i, test_case in enumerate(test_cases):
            milestone = test_case['milestone']
            response = test_case['response']
            expected_score = test_case['expected_score']
            expected_label = test_case['expected_label']
            
            try:
                # Score the response
                result = engine.score_response(milestone, response)
                
                # Convert result to numeric score if it's an enum
                if hasattr(result, 'value'):
                    score = result.value
                else:
                    score = result
                
                # Get the label for the score
                score_label = score_labels.get(score, "UNKNOWN")
                
                # Check if the result matches the expected score
                if score == expected_score:
                    logger.info(f"✅ Test {i+1} PASSED: '{response}' -> {score_label} ({score})")
                    passed += 1
                else:
                    logger.error(f"❌ Test {i+1} FAILED: '{response}' -> Got {score_label} ({score}), Expected {expected_label} ({expected_score})")
                    failed += 1
                    
            except Exception as e:
                logger.error(f"❌ Test {i+1} ERROR: '{response}' -> {str(e)}")
                failed += 1
        
        # Log summary
        logger.info(f"Test Summary: {passed} passed, {failed} failed out of {len(test_cases)} tests")
        
        # Return True if all tests passed
        return failed == 0
        
    except Exception as e:
        logger.error(f"Error running integration test: {str(e)}")
        return False

def main():
    """Main entry point for the script."""
    logger.info("Starting integration of advanced NLP with the assessment engine")
    
    # Create backup of existing enhanced_assessment_engine.py
    create_backup("enhanced_assessment_engine.py") 
    
    # Find the API directory
    api_dir = find_api_directory()
    if not api_dir:
        logger.error("Failed to find API directory.")
        return 1
    
    # Find the assessment engine
    engine_path = find_assessment_engine(api_dir)
    if not engine_path:
        logger.error("Failed to find assessment engine.")
        return 1
    
    # Import the advanced NLP module
    advanced_nlp = import_advanced_nlp()
    if not advanced_nlp:
        logger.error("Failed to import advanced NLP module.")
        return 1
    
    # Integrate with the engine
    success = integrate_nlp_with_engine(engine_path, advanced_nlp)
    if not success:
        logger.error("Failed to integrate NLP with the engine.")
        return 1
    
    # Test the integration
    success = test_integration()
    if success:
        logger.info("Integration successful!")
        
        # Provide instructions for using the enhanced system
        print("\n" + "="*80)
        print("INTEGRATION SUCCESSFUL")
        print("="*80)
        print("\nThe advanced NLP module has been integrated with the assessment engine.")
        print("\nTo test the enhanced scoring with problematic responses:")
        print("  1. Make sure the API server is running:")
        print("     $ ./start_api.sh")
        print()
        print("  2. Test responses for 'walks independently' milestone:")
        print("     $ ./test_response.sh test \"walks independently\" \"no, not yet\"")
        print("     $ ./test_response.sh test \"walks independently\" \"not at all, he has never walked independently\"")
        print()
        print("These responses should now be correctly scored as CANNOT_DO (0).")
        print("\nIf the integration doesn't work correctly, you can modify the handle_complex_response")
        print("function in advanced_nlp.py to handle additional patterns.")
        print("="*80)
        
        return 0
    else:
        logger.error("Integration test failed.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Integration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Integration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 