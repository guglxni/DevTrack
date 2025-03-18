import json
import os

# Load the review queue
with open('data/continuous_learning/review_queue.json', 'r') as f:
    review_queue = json.load(f)

# Check for sample reviews
sample_reviews = [item for item in review_queue if 'id' in item and isinstance(item['id'], str) and item['id'].startswith('sample_')]
print(f'Found {len(sample_reviews)} sample reviews')

# Check if they have the status field
status_count = sum(1 for item in sample_reviews if 'status' in item)
print(f'{status_count} sample reviews have the status field')

# Check if they have the milestone_context field
milestone_context_count = sum(1 for item in sample_reviews if 'milestone_context' in item)
print(f'{milestone_context_count} sample reviews have the milestone_context field')

# Check if they have the milestone field
milestone_count = sum(1 for item in sample_reviews if 'milestone' in item)
print(f'{milestone_count} sample reviews have the milestone field')

# Print the first sample review
if sample_reviews:
    print('\nFirst sample review:')
    for key, value in sample_reviews[0].items():
        if isinstance(value, dict):
            print(f'{key}: {{...}}')
        else:
            print(f'{key}: {value}') 