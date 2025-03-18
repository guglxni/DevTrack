import json
import os

# Load the review queue
with open('data/continuous_learning/review_queue.json', 'r') as f:
    review_queue = json.load(f)

# Find a sample review
sample_review = None
for item in review_queue:
    if 'id' in item and isinstance(item['id'], str) and item['id'].startswith('sample_'):
        sample_review = item
        break

if sample_review:
    print('Sample review found:')
    print(f'ID: {sample_review.get("id")}')
    print(f'Status: {sample_review.get("status")}')
    print(f'Has milestone_context: {"milestone_context" in sample_review}')
    print(f'Has milestone: {"milestone" in sample_review}')
    
    # Check if the sample review is in the first 100 items
    position = review_queue.index(sample_review)
    print(f'Position in review queue: {position} (out of {len(review_queue)} items)')
else:
    print('No sample review found') 