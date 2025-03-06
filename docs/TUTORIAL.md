# ASD Developmental Assessment Tool - Tutorial

This tutorial will guide you through the setup and use of the ASD Developmental Assessment Tool.

## Setup

### 1. Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start the Backend

Start the FastAPI backend server:

```bash
python app.py
```

This will launch the backend server on http://localhost:8000.

### 3. Start the Frontend

In a separate terminal, start the Streamlit frontend:

```bash
streamlit run streamlit_app.py
```

This will open the Streamlit interface in your browser (typically at http://localhost:8501).

## Using the Assessment Tool

### 1. Starting an Assessment

1. When you first open the Streamlit app, you'll see a welcome screen with information about the developmental domains.
2. Set the child's age using the slider in the sidebar.
3. Click "Start Assessment" to begin.

### 2. Completing the Assessment

For each milestone:

1. Read the milestone behavior and criteria carefully.
2. Enter the caregiver's description of the child's behavior in the text area.
   - Be specific about how the child performs the skill.
   - Include details about the frequency, independence level, and quality of performance.
3. Alternatively, use the quick response buttons for common scenarios.
4. Submit your response to proceed to the next milestone.

### 3. Viewing Results

After completing all milestones:

1. The system will automatically generate a report.
2. You'll see a radar chart showing the child's development profile across domains.
3. The "Domain Scores" tab provides detailed scores for each area.
4. The "Response History" tab allows you to review all responses and scores.

## Advanced Features

### Generating Synthetic Data

For testing or demonstration purposes, you can generate synthetic data:

```bash
python synthetic_data_generator.py --age 24 --gender male --output data/example_responses.json
```

Options:
- `--age`: Child's age in months (default: 24)
- `--gender`: Child's gender (male/female/they, default: they)
- `--output`: Output file location
- `--openai`: Use OpenAI API for more natural responses (requires API key)
- `--evaluate`: Evaluate the scoring accuracy on generated data
- `--embeddings`: Use embeddings for evaluation (slower but more accurate)

### API Integration

You can integrate with the assessment API programmatically:

```python
import requests

API_BASE = "http://localhost:8000"

# Set child age
response = requests.post(f"{API_BASE}/set-child-age", json={"age": 24})

# Get next milestone
milestone = requests.get(f"{API_BASE}/next-milestone").json()

# Submit a response
result = requests.post(f"{API_BASE}/score-response", json={
    "response": "Child does this independently without help",
    "milestone_behavior": milestone["behavior"]
}).json()

# Generate report
report = requests.get(f"{API_BASE}/generate-report").json()
```

## Customization

### Adding New Milestones

To add new developmental milestones:

1. Edit the `_initialize_milestones` method in `enhanced_assessment_engine.py`
2. Add your milestone to the appropriate domain and age range
3. Restart the application

### Embedding Models

By default, the system uses the `all-MiniLM-L6-v2` model for embeddings. To use a different model:

```python
engine = EnhancedAssessmentEngine(use_embeddings=True, model_name="your-preferred-model")
```

Popular alternatives include:
- `paraphrase-MiniLM-L6-v2` (faster)
- `all-mpnet-base-v2` (more accurate but slower)
- `all-distilroberta-v1` (balanced)

## Troubleshooting

### Common Issues

1. **Model download fails**: If the embedding model fails to download, the system will automatically fall back to keyword-based scoring.

2. **Memory issues**: If you encounter memory errors with the embedding model, try:
   - Using a smaller model
   - Setting `use_embeddings=False` temporarily
   - Reducing batch size for parallel processing

3. **API connection errors**: Ensure the FastAPI backend is running and accessible from the Streamlit app.

### Getting Help

For additional assistance:
- Review the API documentation at http://localhost:8000/docs
- Check the GitHub repository for updates and issues
- Contact the developers at [your-email@example.com] 