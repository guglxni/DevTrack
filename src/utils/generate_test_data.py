#!/usr/bin/env python3
"""
Test Data Generator for ASD Developmental Milestone Assessment API

This script generates test data for the ASD Developmental Milestone Assessment API 
based on different developmental profiles defined in test_configs.py.
"""

import os
import json
import random
import argparse
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Import local modules
from enhanced_assessment_engine import EnhancedAssessmentEngine, Score, DevelopmentalMilestone
import test_configs

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate test data for ASD Assessment API")
parser.add_argument("--age", type=int, default=24, help="Child's age in months")
parser.add_argument("--profile", type=str, default="neurotypical", 
                    choices=["neurotypical", "delay", "asd", "uneven_motor", "uneven_cognitive", "random"],
                    help="Developmental profile to use")
parser.add_argument("--output", type=str, default="test_data.json", 
                    help="Output file for generated data")
parser.add_argument("--count", type=int, default=10, 
                    help="Number of test cases to generate")
parser.add_argument("--domains", type=str, default="all",
                    help="Comma-separated list of domains to include (e.g., 'GM,FM,COG')")
parser.add_argument("--response_length", type=str, default="medium",
                    choices=["short", "medium", "long"],
                    help="Length of generated responses")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducible results")
parser.add_argument("--use_openai", action="store_true",
                    help="Use OpenAI API for generating more realistic responses")

# Response templates for different score levels
TEMPLATES = {
    Score.INDEPENDENT: [
        "My child can {milestone_text} without any help.",
        "Yes, {child_pronoun} can {milestone_text} independently.",
        "{child_name} has mastered this skill and does it on {child_posessive} own regularly.",
        "This is something {child_name} does well without assistance.",
        "Definitely yes, {child_pronoun} does this with no help needed."
    ],
    Score.WITH_SUPPORT: [
        "My child can {milestone_text} with some help or guidance.",
        "{child_name} is able to do this when I assist {child_objective}.",
        "{child_pronoun} can do this but needs some support from me.",
        "With a little assistance, {child_name} can accomplish this.",
        "{child_pronoun} does this with support but not completely independently."
    ],
    Score.EMERGING: [
        "My child is just starting to {milestone_text}.",
        "{child_name} shows some signs of this skill but it's not consistent.",
        "{child_pronoun} occasionally attempts to do this but it's still developing.",
        "This is an emerging skill - {child_pronoun} tries but isn't quite there yet.",
        "Sometimes {child_name} shows this ability, but it's not reliable."
    ],
    Score.CANNOT_DO: [
        "My child cannot {milestone_text} yet.",
        "No, {child_name} isn't able to do this at this time.",
        "{child_pronoun} hasn't developed this skill yet.",
        "This is something {child_name} hasn't shown any ability to do.",
        "{child_pronoun} doesn't do this even with help."
    ],
    Score.LOST_SKILL: [
        "My child used to be able to {milestone_text}, but no longer does.",
        "{child_name} could do this before but has lost this ability.",
        "{child_pronoun} previously demonstrated this skill but has regressed.",
        "This is something {child_name} used to do, but stopped around {regression_age}.",
        "{child_pronoun} had this skill but lost it."
    ]
}

# Extended response templates for more detailed responses
EXTENDED_TEMPLATES = {
    Score.INDEPENDENT: [
        "My child can {milestone_text} without any help. {child_pronoun} mastered this skill about {mastery_time} and does it consistently in different settings like {setting1} and {setting2}.",
        "Yes, {child_name} demonstrates this ability independently. I've noticed {child_pronoun} does this regularly, especially when {context}. {child_pronoun} seems to enjoy it and will often {related_behavior}.",
        "{child_name} has completely mastered this skill. {child_pronoun} does it on {child_posessive} own not only at home but also when we're {alternate_setting}. It's been consistent for {consistency_time}.",
        "This is something {child_name} does very well without any assistance. {child_pronoun} first showed this ability when {first_instance} and has been doing it confidently ever since.",
        "Definitely yes. {child_name} is very independent with this skill and even helps {sibling_or_peer} with it. {child_pronoun} demonstrates this ability {frequency} and in various contexts."
    ],
    Score.WITH_SUPPORT: [
        "My child can {milestone_text} with some help. Usually I need to {support_action} and then {child_pronoun} can complete the rest. {child_pronoun} seems to be getting more confident with practice.",
        "{child_name} does this when I provide some assistance. Specifically, {child_pronoun} needs help with {specific_challenge} but can manage {specific_strength} on {child_posessive} own.",
        "{child_pronoun} is getting better at this but still needs my support. I've noticed improvement over the past {improvement_time} - before that {child_pronoun} needed much more help.",
        "With a little guidance, {child_name} can do this. I usually {prompt_method} to get {child_objective} started, and then {child_pronoun} continues with occasional reminders.",
        "{child_pronoun} does this with support from {helper}. The specific help needed is {specific_support}, but {child_pronoun} can handle {partial_skill} independently."
    ],
    Score.EMERGING: [
        "My child is just beginning to {milestone_text}. I first noticed attempts about {emergence_time} ago. It's inconsistent - sometimes {child_pronoun} will try when {trigger_condition} but other times {child_pronoun} shows no interest.",
        "{child_name} shows early signs of this skill. {child_pronoun} has tried {attempt_count} times in the past week, succeeding about {success_rate}. It seems to depend on {variable_factor}.",
        "{child_pronoun} occasionally attempts this, especially when {motivation_factor}. The skill isn't reliable yet, but I can see {child_pronoun} is making progress compared to {comparison_time} ago.",
        "This is definitely emerging. {child_name} will {partial_achievement} but hasn't mastered {remaining_challenge} yet. We're practicing by {practice_method}.",
        "Sometimes {child_name} shows this ability, particularly when {child_pronoun} is {emotional_state}. It's not consistent though - I would say {child_pronoun} succeeds about {percentage} of the time."
    ],
    Score.CANNOT_DO: [
        "My child cannot {milestone_text} yet. We've tried encouraging this through {attempt_method} but {child_pronoun} doesn't seem ready. {child_pronoun} tends to {alternative_behavior} instead.",
        "No, {child_name} isn't able to do this at this time. When I try to have {child_objective} {milestone_text}, {child_pronoun} becomes {emotional_response} and {avoidance_behavior}.",
        "{child_pronoun} hasn't developed this skill yet. I've observed other children {child_posessive} age doing this, but {child_name} shows {obstacle_description} when attempting it.",
        "This is something {child_name} hasn't shown any ability to do. Instead of {milestone_text}, {child_pronoun} will {substitute_behavior}. We've been working on prerequisite skills like {prerequisite}.",
        "{child_pronoun} doesn't do this even with significant help and encouragement. When we try, {child_pronoun} typically {resistance_behavior}. We've consulted with {professional} about this."
    ],
    Score.LOST_SKILL: [
        "My child used to be able to {milestone_text} around {skill_age}, but stopped doing it about {regression_time} ago. This coincided with {coinciding_event}, and we haven't seen the skill since despite {restoration_attempt}.",
        "{child_name} could do this consistently for about {duration_before_loss}, but has lost this ability. The regression seemed {regression_speed} and happened when {child_pronoun} was around {regression_age}.",
        "{child_pronoun} previously demonstrated this skill regularly. I have videos from when {child_pronoun} was {documented_age} showing {child_objective} doing this. Now {child_pronoun} {current_behavior} instead.",
        "This is something {child_name} used to do from age {onset_age} until about {regression_age}. The change was {nature_of_change} and occurred around the time that {life_event}.",
        "{child_pronoun} had this skill but lost it. Before the regression, {child_pronoun} would {past_behavior} regularly. Now even with {support_type}, {child_pronoun} doesn't show any sign of this ability."
    ]
}

# Variables to fill in templates
CHILD_NAMES = ["Alex", "Jamie", "Jordan", "Taylor", "Casey", "Riley", "Morgan", "Avery", "Charlie", "Jessie"]
CHILD_PRONOUNS = {"he": {"pronoun": "he", "posessive": "his", "objective": "him"},
                 "she": {"pronoun": "she", "posessive": "her", "objective": "her"},
                 "they": {"pronoun": "they", "posessive": "their", "objective": "them"}}

TIME_PERIODS = ["2 weeks", "a month", "3 months", "several weeks", "a couple of months"]
SETTINGS = ["at home", "at daycare", "at the park", "during playdates", "at grandma's house", 
            "during meals", "at bedtime", "in the car", "at the store", "during bath time"]
FREQUENCIES = ["daily", "several times a day", "almost every time", "regularly", "consistently",
               "often", "most of the time", "whenever the opportunity arises"]
HELPERS = ["me", "dad", "mom", "grandparents", "daycare teacher", "older sibling", "babysitter"]
EMOTIONS = ["happy", "excited", "calm", "focused", "tired", "hungry", "overwhelmed", "interested"]
PERCENTAGES = ["20%", "about half the time", "60-70%", "one in three attempts", "rarely"]

class TestDataGenerator:
    """Generates test data for the ASD Assessment API based on developmental profiles"""
    
    def __init__(self, age: int = 24, profile_type: str = "neurotypical", 
                 response_length: str = "medium", use_openai: bool = False,
                 domains: str = "all"):
        """
        Initialize the test data generator
        
        Args:
            age: Child's age in months
            profile_type: Type of developmental profile to use
            response_length: Length of responses to generate (short, medium, long)
            use_openai: Whether to use OpenAI API for more realistic responses
            domains: Comma-separated list of domains to include, or "all"
        """
        self.age = age
        self.profile_type = profile_type
        self.response_length = response_length
        self.use_openai = use_openai
        
        # Initialize assessment engine
        self.engine = EnhancedAssessmentEngine()
        self.engine.set_child_age(age)
        
        # Set domains to include
        self.domains = domains.split(',') if domains != "all" else []
        
        # Create child info
        self.child = {
            "name": random.choice(CHILD_NAMES),
            "age": age,
            "gender": random.choice(list(CHILD_PRONOUNS.keys()))
        }
        
        # Get pronouns based on gender
        self.pronouns = CHILD_PRONOUNS[self.child["gender"]]
        
        # Get all milestones for the child's age
        self.all_milestones = self.engine.milestones
        
        # Filter by age
        self.all_milestones = [m for m in self.all_milestones if self._is_age_appropriate(m)]
        
        # Filter by domains if specified
        if self.domains:
            self.all_milestones = [m for m in self.all_milestones 
                                  if any(domain.lower() in m.domain.lower() for domain in self.domains)]
        
        print(f"Found {len(self.all_milestones)} milestones for age {age}")
        
        # Get age-appropriate profile
        if profile_type == "random":
            # Randomly select a profile type for each test case
            self.profile_types = ["neurotypical", "delay", "asd", 
                                 "uneven_motor", "uneven_cognitive"]
        else:
            self.profile_types = [profile_type]
    
    def _is_age_appropriate(self, milestone: DevelopmentalMilestone) -> bool:
        """Check if milestone is appropriate for the child's age"""
        age_range = milestone.age_range
        
        # Handle different age range formats
        if "-" in age_range:
            parts = age_range.replace("months", "").replace("m", "").strip().split("-")
            try:
                min_age = int(parts[0])
                max_age = int(parts[1])
                return min_age <= self.age <= max_age
            except (ValueError, IndexError):
                return False
        elif "+" in age_range:
            try:
                min_age = int(age_range.replace("+", "").replace("months", "").replace("m", "").strip())
                return self.age >= min_age
            except ValueError:
                return False
        else:
            try:
                exact_age = int(age_range.replace("months", "").replace("m", "").strip())
                return self.age == exact_age
            except ValueError:
                return False

    def _get_variables_for_template(self, milestone: DevelopmentalMilestone) -> Dict[str, str]:
        """Generate variables to fill in response templates"""
        # Basic variables
        variables = {
            "child_name": self.child["name"],
            "child_pronoun": self.pronouns["pronoun"],
            "child_posessive": self.pronouns["posessive"],
            "child_objective": self.pronouns["objective"],
            "milestone_text": milestone.behavior.lower(),
            # For regression scenarios
            "regression_age": f"{max(1, self.child['age'] - random.randint(3, 6))} months",
        }
        
        # Add first setting before creating extended variables
        variables["setting1"] = random.choice(SETTINGS)
        
        # Extended variables for longer responses
        extended_vars = {
            # Independent variables
            "mastery_time": random.choice(TIME_PERIODS),
            "setting2": random.choice([s for s in SETTINGS if s != variables["setting1"]]),
            "context": f"we are {random.choice(SETTINGS)}",
            "related_behavior": f"try to {random.choice(['do it again', 'show others', 'practice more'])}",
            "alternate_setting": random.choice(SETTINGS),
            "consistency_time": random.choice(TIME_PERIODS),
            "first_instance": f"we were {random.choice(SETTINGS)}",
            "frequency": random.choice(FREQUENCIES),
            "sibling_or_peer": random.choice(["siblings", "friends", "cousins", "peers"]),
            
            # With support variables
            "support_action": f"help with {random.choice(['starting', 'positioning', 'guiding', 'demonstrating'])}",
            "specific_challenge": f"the {random.choice(['beginning', 'difficult parts', 'transitions', 'finishing'])}",
            "specific_strength": f"the {random.choice(['simpler aspects', 'familiar parts', 'repetitive elements'])}",
            "improvement_time": random.choice(TIME_PERIODS),
            "prompt_method": f"{random.choice(['verbally prompt', 'show', 'guide', 'remind'])}",
            "helper": random.choice(HELPERS),
            "specific_support": f"{random.choice(['verbal cues', 'physical guidance', 'demonstrations', 'reminders'])}",
            "partial_skill": f"the {random.choice(['basic parts', 'preparation', 'simpler elements'])}",
            
            # Emerging variables
            "emergence_time": random.choice(TIME_PERIODS),
            "trigger_condition": f"{random.choice(['motivated', 'in a good mood', 'well-rested', 'interested'])}",
            "attempt_count": str(random.randint(2, 10)),
            "success_rate": f"{random.randint(1, 4)} out of 10 times",
            "variable_factor": f"{random.choice(['mood', 'time of day', 'who is present', 'level of interest'])}",
            "motivation_factor": f"{random.choice(['interested in the outcome', 'sees others doing it', 'feeling confident'])}",
            "comparison_time": random.choice(TIME_PERIODS),
            "partial_achievement": f"{random.choice(['attempt part of the task', 'start the process', 'show understanding'])}",
            "remaining_challenge": f"the {random.choice(['whole process', 'difficult aspects', 'consistency'])}",
            "practice_method": f"{random.choice(['daily practice', 'games', 'modeling', 'breaking it down'])}",
            "emotional_state": random.choice(EMOTIONS),
            "percentage": random.choice(PERCENTAGES),
            
            # Cannot do variables
            "attempt_method": f"{random.choice(['demonstrations', 'games', 'encouragement', 'practice'])}",
            "alternative_behavior": f"{random.choice(['avoid it', 'get frustrated', 'lose interest', 'try something else'])}",
            "emotional_response": random.choice(EMOTIONS),
            "avoidance_behavior": f"{random.choice(['tries to change activities', 'cries', 'walks away', 'gets distracted'])}",
            "obstacle_description": f"{random.choice(['difficulty', 'lack of interest', 'frustration', 'confusion'])}",
            "substitute_behavior": f"{random.choice(['do something simpler', 'ask for help', 'get frustrated', 'give up'])}",
            "prerequisite": f"{random.choice(['simpler skills', 'building blocks', 'foundation abilities'])}",
            "resistance_behavior": f"{random.choice(['refuses', 'gets upset', 'disengages', 'changes subject'])}",
            "professional": f"{random.choice(['our pediatrician', 'a therapist', 'their teacher', 'a specialist'])}",
            
            # Lost skill variables
            "skill_age": f"{max(1, self.child['age'] - random.randint(6, 10))} months",
            "regression_time": random.choice(TIME_PERIODS),
            "coinciding_event": f"{random.choice(['a move', 'starting daycare', 'a new sibling', 'an illness'])}",
            "restoration_attempt": f"{random.choice(['encouragement', 'practice', 'professional help', 'consistency'])}",
            "duration_before_loss": f"{random.choice(['a few weeks', 'several months', 'about a month'])}",
            "regression_speed": f"{random.choice(['sudden', 'gradual', 'somewhat abrupt', 'over time'])}",
            "documented_age": f"{max(1, self.child['age'] - random.randint(4, 8))} months",
            "current_behavior": f"{random.choice(['refuses', 'shows no interest', 'seems confused', 'gets frustrated'])}",
            "onset_age": f"{max(1, self.child['age'] - random.randint(8, 12))} months",
            "nature_of_change": f"{random.choice(['sudden', 'gradual', 'noticeable', 'concerning'])}",
            "life_event": f"{random.choice(['we moved', 'started daycare', 'had a medical event', 'family change'])}",
            "past_behavior": f"{random.choice(['do it eagerly', 'demonstrate the skill', 'practice it', 'show mastery'])}",
            "support_type": f"{random.choice(['encouragement', 'assistance', 'demonstration', 'practice'])}",
        }
        
        # Combine basic and extended variables
        variables.update(extended_vars)
        return variables

    def generate_random_response(self, milestone: DevelopmentalMilestone, score: Score) -> str:
        """Generate a random response for a milestone based on the score"""
        # Choose template based on response length
        if self.response_length == "short":
            templates = TEMPLATES[score]
        else:  # medium or long
            templates = EXTENDED_TEMPLATES[score]
        
        # Select a random template
        template = random.choice(templates)
        
        # Fill in template variables
        variables = self._get_variables_for_template(milestone)
        
        # Generate response from template
        response = template.format(**variables)
        
        # For long responses, add additional contextual information
        if self.response_length == "long":
            additional_context = [
                f" I've noticed this {random.choice(['consistently', 'frequently', 'occasionally', 'rarely'])} over the past {random.choice(TIME_PERIODS)}.",
                f" This is {random.choice(['typical', 'unusual', 'common', 'rare'])} compared to {self.child['name']}'s other skills.",
                f" {self.pronouns['pronoun'].capitalize()} seems to be {random.choice(['progressing well', 'developing typically', 'having some challenges', 'showing mixed progress'])} in this area.",
                f" We've been {random.choice(['working on this', 'practicing regularly', 'getting help with this', 'focusing on related skills'])}.",
                f" Other family members have {random.choice(['noticed the same thing', 'pointed this out too', 'been helping with this', 'seen similar patterns'])}."
            ]
            response += random.choice(additional_context)
        
        return response

    def get_score_for_milestone(self, milestone: DevelopmentalMilestone, profile_type: str) -> Score:
        """Determine the appropriate score for a milestone based on the profile"""
        # Get score distribution for the profile
        if profile_type in ["uneven_motor", "uneven_cognitive"] and milestone.domain:
            profile = test_configs.get_profile(self.age, profile_type, milestone.domain)
        else:
            profile = test_configs.get_profile(self.age, profile_type)
        
        # If no profile found or empty profile, use neurotypical
        if not profile:
            profile = test_configs.get_profile(self.age, "neurotypical")
        
        # Convert profile to list of scores with weights
        weighted_scores = []
        for score, weight in profile.items():
            weighted_scores.extend([score] * int(weight * 100))
        
        # Randomly select a score based on weights
        return random.choice(weighted_scores)

    def generate_test_case(self, milestone: DevelopmentalMilestone, profile_type: str) -> Dict:
        """Generate a single test case for a milestone"""
        # Determine the appropriate score for this milestone based on profile
        score = self.get_score_for_milestone(milestone, profile_type)
        
        # Generate a response based on the score
        response = self.generate_random_response(milestone, score)
        
        # Create test case object
        test_case = {
            "milestone": milestone.behavior,
            "domain": milestone.domain,
            "age_expected": milestone.age_range,
            "caregiver_response": response,
            "expected_score": score.value,
            "expected_label": score.name,
            "profile_type": profile_type
        }
        
        return test_case

    async def generate_test_dataset(self, count: int = 10) -> List[Dict]:
        """Generate a full test dataset with the specified number of test cases"""
        test_dataset = []
        
        # Ensure we have enough milestones
        sample_count = min(count, len(self.all_milestones))
        
        if sample_count == 0:
            print("No appropriate milestones found for the specified age and domains!")
            return []
            
        milestones = random.sample(self.all_milestones, sample_count)
        
        # Generate test cases
        for i in range(count):
            # Select milestone (with wrapping if count > number of milestones)
            milestone = milestones[i % sample_count]
            
            # Select profile type (randomly if "random" was chosen)
            profile = random.choice(self.profile_types)
            
            # Generate test case
            test_case = self.generate_test_case(milestone, profile)
            test_dataset.append(test_case)
        
        return test_dataset

    def save_dataset(self, dataset: List[Dict], filename: str) -> None:
        """Save the generated dataset to a JSON file"""
        # Create output dictionary with metadata
        output = {
            "metadata": {
                "age": self.age,
                "profile_type": self.profile_type,
                "response_length": self.response_length,
                "generated_at": datetime.now().isoformat(),
                "child": self.child
            },
            "test_cases": dataset
        }
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        print(f"Test dataset saved to {filename}")

async def main():
    """Main function to run the test data generator"""
    # Parse command line arguments
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Create test data generator
    generator = TestDataGenerator(
        age=args.age,
        profile_type=args.profile,
        response_length=args.response_length,
        use_openai=args.use_openai,
        domains=args.domains
    )
    
    # Generate test dataset
    print(f"Generating {args.count} test cases...")
    dataset = await generator.generate_test_dataset(args.count)
    
    # Save dataset to file
    generator.save_dataset(dataset, args.output)

if __name__ == "__main__":
    asyncio.run(main()) 