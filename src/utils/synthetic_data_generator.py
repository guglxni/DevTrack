"""
Synthetic Data Generator for ASD Assessment Tool

This script generates realistic caregiver responses for testing the ASD assessment system.
It can create responses across different developmental levels to test the scoring engine.
"""

import os
import json
import random
import asyncio
import argparse
from typing import List, Dict, Any
import pandas as pd
from enhanced_assessment_engine import EnhancedAssessmentEngine, Score, DevelopmentalMilestone

# Set to True to use OpenAI's API if available
USE_OPENAI = False

try:
    import openai
    if os.environ.get("OPENAI_API_KEY"):
        USE_OPENAI = True
        openai.api_key = os.environ.get("OPENAI_API_KEY")
except ImportError:
    print("OpenAI package not found. Using template-based generation instead.")
    USE_OPENAI = False

# Templates for different scoring levels
RESPONSE_TEMPLATES = {
    Score.CANNOT_DO: [
        "No, {child_pronoun} cannot {behavior_lower} at all. {child_pronoun_cap} has never shown any ability to {criteria_lower}.",
        "{child_pronoun_cap} doesn't {behavior_lower} yet. When I try to encourage {child_pronoun} to {criteria_lower}, {child_pronoun} doesn't respond.",
        "I've never seen {child_pronoun} {behavior_lower}. {child_pronoun_cap} seems to struggle with {criteria_lower}.",
        "{child_pronoun_cap} hasn't demonstrated this skill at all. {child_pronoun_cap} doesn't seem interested in trying to {criteria_lower}.",
        "This is definitely not something {child_pronoun} can do. {child_pronoun_cap} shows no signs of being able to {criteria_lower}."
    ],
    Score.LOST_SKILL: [
        "{child_pronoun_cap} used to be able to {behavior_lower} a few months ago, but has stopped. {child_pronoun_cap} no longer tries to {criteria_lower}.",
        "This is concerning because {child_pronoun} could do this before. Around {regression_age} months, {child_pronoun} would {criteria_lower}, but now {child_pronoun} doesn't anymore.",
        "{child_pronoun_cap} has regressed in this area. {child_pronoun_cap} previously could {criteria_lower}, but has lost this ability.",
        "We noticed {child_pronoun} stopped doing this around {regression_age} months. {child_pronoun_cap} used to {criteria_lower} regularly, but now shows no interest.",
        "This was a skill {child_pronoun} had mastered, but has since lost. {child_pronoun_cap} no longer attempts to {criteria_lower} like {child_pronoun} used to."
    ],
    Score.EMERGING: [
        "{child_pronoun_cap} is just beginning to {behavior_lower}. Sometimes {child_pronoun} will attempt to {criteria_lower}, but it's inconsistent.",
        "I've seen {child_pronoun} try to {criteria_lower} occasionally. It's an emerging skill that {child_pronoun}'s working on.",
        "{child_pronoun_cap} shows some interest in this. About 25% of the time, {child_pronoun} will try to {criteria_lower}.",
        "This is starting to develop. {child_pronoun_cap} occasionally {criteria_lower}, but not consistently yet.",
        "{child_pronoun_cap} is making progress with this skill. {child_pronoun_cap} sometimes {criteria_lower}, especially when encouraged."
    ],
    Score.WITH_SUPPORT: [
        "{child_pronoun_cap} can {behavior_lower} with my help. When supported, {child_pronoun} is able to {criteria_lower} successfully.",
        "With assistance, {child_pronoun} manages to {criteria_lower}. {child_pronoun_cap} needs guidance but can do it.",
        "{child_pronoun_cap} performs this skill when I help {child_pronoun}. {child_pronoun_cap} can {criteria_lower} with support.",
        "This is something {child_pronoun} needs assistance with. {child_pronoun_cap} can {criteria_lower} when I'm there to guide {child_pronoun}.",
        "{child_pronoun_cap} relies on my help to {behavior_lower}. With support, {child_pronoun} successfully {criteria_lower}."
    ],
    Score.INDEPENDENT: [
        "{child_pronoun_cap} does this independently all the time. {child_pronoun_cap} can easily {criteria_lower} without any help.",
        "This is a strong skill for {child_pronoun}. {child_pronoun_cap} consistently {criteria_lower} on {child_pronoun}'s own.",
        "{child_pronoun_cap} has mastered this completely. {child_pronoun_cap} {criteria_lower} independently in various situations.",
        "No concerns with this skill. {child_pronoun_cap} {criteria_lower} perfectly without assistance.",
        "{child_pronoun_cap} is very good at this. {child_pronoun_cap} independently {criteria_lower} whenever the opportunity arises."
    ]
}

def generate_response_from_template(milestone: DevelopmentalMilestone, score: Score, child_gender: str = "they") -> str:
    """Generate a synthetic response using templates"""
    # Set pronouns based on gender
    if child_gender.lower() in ["m", "male", "boy", "he"]:
        pronouns = {"child_pronoun": "he", "child_pronoun_cap": "He", "child_possessive": "his"}
    elif child_gender.lower() in ["f", "female", "girl", "she"]:
        pronouns = {"child_pronoun": "she", "child_pronoun_cap": "She", "child_possessive": "her"}
    else:
        pronouns = {"child_pronoun": "they", "child_pronoun_cap": "They", "child_possessive": "their"}
    
    # Extract age range
    start_age = int(milestone.age_range.split('-')[0].strip())
    end_age = int(milestone.age_range.split('-')[1].split()[0].strip())
    
    # For regression, pick an age when the child might have had the skill
    regression_age = max(start_age - random.randint(3, 6), 1)
    
    # Format the template with milestone specifics
    templates = RESPONSE_TEMPLATES[score]
    template = random.choice(templates)
    
    return template.format(
        behavior_lower=milestone.behavior.lower(),
        criteria_lower=milestone.criteria.lower(),
        regression_age=regression_age,
        **pronouns
    )

async def generate_response_with_openai(milestone: DevelopmentalMilestone, score: Score, child_gender: str = "they") -> str:
    """Generate a more realistic response using OpenAI API"""
    if not USE_OPENAI:
        return generate_response_from_template(milestone, score, child_gender)
    
    # Set pronouns based on gender
    if child_gender.lower() in ["m", "male", "boy", "he"]:
        gender_text = "boy (he/him)"
    elif child_gender.lower() in ["f", "female", "girl", "she"]:
        gender_text = "girl (she/her)"
    else:
        gender_text = "child (they/them)"
    
    # Map score to description
    score_descriptions = {
        Score.CANNOT_DO: "CANNOT DO - The child shows no ability to perform this skill",
        Score.LOST_SKILL: "LOST SKILL - The child could do this before but has regressed and lost this ability",
        Score.EMERGING: "EMERGING - The child is beginning to show this skill but is inconsistent",
        Score.WITH_SUPPORT: "WITH SUPPORT - The child can do this with assistance or guidance",
        Score.INDEPENDENT: "INDEPENDENT - The child does this completely independently"
    }
    
    prompt = f"""
Generate a realistic response from a parent or caregiver describing their child's development for an ASD assessment.

The milestone being assessed is: {milestone.behavior}
The specific criteria is: {milestone.criteria}
The child's age range for this skill is: {milestone.age_range}
The child is a {gender_text}
The developmental level is: {score_descriptions[score]}

Write a detailed 2-3 sentence response as if you are the parent/caregiver describing the child's ability with this skill. Be specific about behaviors you observe, include realistic details, and match the developmental level indicated above.
"""

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a parent or caregiver of a child being assessed for developmental milestones."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return generate_response_from_template(milestone, score, child_gender)

async def generate_synthetic_dataset(
    age: int, 
    gender: str = "they",
    score_distribution: Dict[Score, float] = None,
    use_openai: bool = USE_OPENAI,
    output_file: str = "synthetic_responses.json"
) -> List[Dict[str, Any]]:
    """
    Generate a full synthetic dataset for a child of specified age
    
    Args:
        age: Child's age in months
        gender: Child's gender (male/female/they)
        score_distribution: Optional distribution of scores to generate
                           (e.g., {Score.INDEPENDENT: 0.5, Score.WITH_SUPPORT: 0.3, Score.EMERGING: 0.2})
        use_openai: Whether to use OpenAI API for generation
        output_file: File to save the synthetic data
                          
    Returns:
        List of dictionaries with milestone and response data
    """
    # Initialize the engine
    engine = EnhancedAssessmentEngine(use_embeddings=False)  # Embeddings not needed for generation
    engine.set_child_age(age)
    
    # Default score distribution if none provided
    if not score_distribution:
        score_distribution = {
            Score.INDEPENDENT: 0.5,    # 50% independent skills
            Score.WITH_SUPPORT: 0.2,   # 20% skills with support
            Score.EMERGING: 0.1,       # 10% emerging skills
            Score.LOST_SKILL: 0.1,     # 10% lost skills
            Score.CANNOT_DO: 0.1       # 10% cannot do
        }
    
    # Check that score distribution sums to 1
    total = sum(score_distribution.values())
    if abs(total - 1.0) > 0.01:
        print(f"Warning: Score distribution sums to {total}, normalizing to 1.0")
        score_distribution = {k: v/total for k, v in score_distribution.items()}
    
    # Generate responses for each milestone
    synthetic_data = []
    milestone = engine.get_next_milestone()
    
    while milestone:
        # Assign a score based on the distribution
        score = random.choices(
            list(score_distribution.keys()),
            weights=list(score_distribution.values()),
            k=1
        )[0]
        
        # Generate response
        if use_openai:
            response_text = await generate_response_with_openai(milestone, score, gender)
        else:
            response_text = generate_response_from_template(milestone, score, gender)
        
        # Add to dataset
        synthetic_data.append({
            "milestone_behavior": milestone.behavior,
            "milestone_criteria": milestone.criteria,
            "milestone_domain": milestone.domain,
            "milestone_age_range": milestone.age_range,
            "actual_score": score.name,
            "actual_score_value": score.value,
            "response": response_text
        })
        
        # Move to next milestone
        milestone = engine.get_next_milestone()
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(synthetic_data, f, indent=2)
    
    print(f"Generated {len(synthetic_data)} synthetic responses and saved to {output_file}")
    return synthetic_data

async def evaluate_scoring_accuracy(
    synthetic_data: List[Dict[str, Any]], 
    use_embeddings: bool = True
) -> pd.DataFrame:
    """
    Evaluate how accurately the scoring engine scores the synthetic data
    
    Args:
        synthetic_data: List of dictionaries with milestone and response data
        use_embeddings: Whether to use embeddings for scoring
        
    Returns:
        DataFrame with evaluation results
    """
    # Initialize the engine with the selected scoring method
    engine = EnhancedAssessmentEngine(use_embeddings=use_embeddings)
    
    # Extract the oldest age from the dataset
    max_age = 36  # Default
    for item in synthetic_data:
        age_range = item["milestone_age_range"]
        end_age = int(age_range.split('-')[1].split()[0].strip())
        max_age = max(max_age, end_age)
    
    # Set the age to ensure all milestones are available
    engine.set_child_age(max_age)
    
    results = []
    
    # Process each synthetic response
    for item in synthetic_data:
        # Find the corresponding milestone
        milestone = None
        for m in engine.milestones:
            if (m.behavior == item["milestone_behavior"] and 
                m.domain == item["milestone_domain"]):
                milestone = m
                break
        
        if not milestone:
            print(f"Warning: Milestone {item['milestone_behavior']} not found")
            continue
        
        # Score the response
        if use_embeddings:
            predicted_score = await engine.analyze_response(item["response"], milestone)
        else:
            predicted_score = engine.analyze_response_keywords(item["response"], milestone)
        
        # Convert actual score string to enum
        actual_score = Score[item["actual_score"]]
        
        # Store result
        results.append({
            "milestone": item["milestone_behavior"],
            "domain": item["milestone_domain"],
            "age_range": item["milestone_age_range"],
            "actual_score": actual_score.name,
            "actual_score_value": actual_score.value,
            "predicted_score": predicted_score.name,
            "predicted_score_value": predicted_score.value,
            "correct": actual_score == predicted_score,
            "response": item["response"]
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate accuracy metrics
    accuracy = df["correct"].mean() * 100
    print(f"Overall accuracy: {accuracy:.2f}%")
    
    by_domain = df.groupby("domain")["correct"].mean() * 100
    print("\nAccuracy by domain:")
    for domain, acc in by_domain.items():
        print(f"{domain}: {acc:.2f}%")
    
    by_score = df.groupby("actual_score")["correct"].mean() * 100
    print("\nAccuracy by score level:")
    for score, acc in by_score.items():
        print(f"{score}: {acc:.2f}%")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    confusion = pd.crosstab(df["actual_score"], df["predicted_score"], normalize="index") * 100
    print(confusion)
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for ASD assessment")
    parser.add_argument("--age", type=int, default=24, help="Child's age in months")
    parser.add_argument("--gender", type=str, default="they", help="Child's gender (male/female/they)")
    parser.add_argument("--output", type=str, default="synthetic_responses.json", help="Output file")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI for generation")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate scoring accuracy")
    parser.add_argument("--embeddings", action="store_true", help="Use embeddings for evaluation")
    
    args = parser.parse_args()
    
    async def main():
        # Generate synthetic data
        synthetic_data = await generate_synthetic_dataset(
            age=args.age,
            gender=args.gender,
            use_openai=args.openai and USE_OPENAI,
            output_file=args.output
        )
        
        # Evaluate if requested
        if args.evaluate:
            print("\nEvaluating scoring accuracy...")
            results_df = await evaluate_scoring_accuracy(
                synthetic_data=synthetic_data,
                use_embeddings=args.embeddings
            )
            
            # Save evaluation results
            results_file = args.output.replace(".json", "_evaluation.csv")
            results_df.to_csv(results_file, index=False)
            print(f"Evaluation results saved to {results_file}")
    
    asyncio.run(main()) 