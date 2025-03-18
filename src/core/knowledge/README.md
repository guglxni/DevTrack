# Knowledge Engineering Module

This module implements domain-specific knowledge and enhanced prompt engineering for developmental milestone assessments. It provides a structured approach to incorporating expert knowledge into the scoring system, particularly for the LLM-based scorer.

## Architecture

The Knowledge Engineering module consists of two main components:

1. **Developmental Domains**: Structured representation of domain-specific knowledge
2. **Prompt Templates**: Domain-specific prompt templates for LLM-based scoring

```
Knowledge Engineering Module
├── Developmental Domains
│   ├── Motor
│   ├── Communication
│   ├── Social
│   └── Cognitive
└── Prompt Templates
    ├── Base Templates
    ├── Domain-Specific Templates
    └── Template Management
```

## Developmental Domains

Each developmental domain (Motor, Communication, Social, Cognitive) is represented as a `DevelopmentalDomain` object with the following attributes:

- **Code**: Short code (e.g., "MOTOR")
- **Name**: Full name (e.g., "Motor Development")
- **Description**: Detailed description of the domain
- **Milestone Types**: Types of milestones in this domain
- **Assessment Considerations**: Special considerations for assessment
- **Category Rubrics**: Detailed rubrics for each scoring category
- **Domain-Specific Prompts**: References to domain-specific prompt templates

Each category rubric contains:
- **Description**: General description of the category
- **Criteria**: Specific criteria for this category
- **Examples**: Example responses that fit this category
- **Keywords**: Keywords that suggest this category
- **Transitions**: Boundary indicators with neighboring categories

## Prompt Templates

The prompt templates system provides domain-specific prompts for the LLM-based scorer. These templates are designed to enhance the accuracy of the scoring by providing domain-specific context and guidance to the LLM.

### Template Structure

Each template is a JSON file with the following structure:

```json
{
  "template_id": "domain_template",
  "domain_code": "DOMAIN",
  "domain_name": "Domain Development",
  "base_template": "...",
  "domain_guidance": "...",
  "category_descriptions": {
    "cannot_do_desc": "...",
    "with_support_desc": "...",
    "emerging_desc": "...",
    "independent_desc": "...",
    "lost_skill_desc": "..."
  },
  "version": "1.0.0"
}
```

### Domain-Specific Guidance

Each domain template includes specialized guidance for evaluating responses in that domain:

- **Motor**: Focus on quality and consistency of movement, physical assistance, environmental adaptations
- **Communication**: Focus on frequency and consistency of communication, initiation vs. response, range of functions
- **Social**: Focus on interest in others, initiation of interactions, quality of social exchanges, joint attention
- **Cognitive**: Focus on problem-solving approaches, curiosity, memory, flexibility of thinking, attention span

### Category Descriptions

Each template includes domain-specific descriptions for each scoring category:

- **CANNOT_DO**: Child shows no evidence of the skill
- **WITH_SUPPORT**: Child performs the skill with assistance or in limited contexts
- **EMERGING**: Child is beginning to show the skill but is inconsistent
- **INDEPENDENT**: Child performs the skill consistently without help
- **LOST_SKILL**: Child previously demonstrated the skill but has regressed

## Usage

### Accessing Domain Knowledge

```python
from src.core.knowledge import get_domain_by_name, get_all_domains

# Get a specific domain
motor_domain = get_domain_by_name("motor")

# Get all domains
all_domains = get_all_domains()

# Access domain attributes
print(motor_domain.name)
print(motor_domain.description)

# Get category rubric
emerging_rubric = motor_domain.get_category_rubric("EMERGING")
print(emerging_rubric.description)
print(emerging_rubric.criteria)
```

### Using Domain-Specific Prompts

```python
from src.core.knowledge import load_prompt, format_prompt_with_context

# Load a domain-specific prompt template
template = load_prompt("motor")

# Format the prompt with context
formatted_prompt = format_prompt_with_context(
    template,
    response="My child is starting to crawl but needs help sometimes.",
    milestone_context={
        "behavior": "Crawling",
        "criteria": "Child moves on hands and knees across the floor",
        "age_range": "6-10 months",
        "domain": "motor"
    }
)

# Use the formatted prompt with an LLM
# ...
```

### Creating Custom Templates

```python
from src.core.knowledge import create_domain_specific_prompt, validate_prompt, save_prompt

# Create a new domain-specific prompt template
template = create_domain_specific_prompt("motor")

# Customize the template
template["domain_guidance"] = "Custom guidance for motor assessment..."

# Validate the template
is_valid, message = validate_prompt(template)
if is_valid:
    # Save the template
    save_prompt(template)
```

## Integration with LLM Scorer

The Knowledge Engineering module is integrated with the LLM-based scorer to provide domain-specific prompts for more accurate scoring. The LLM scorer will automatically use the appropriate domain-specific prompt template based on the domain specified in the milestone context.

```python
from src.core.scoring.llm_scorer import LLMBasedScorer

# Initialize the LLM scorer with domain-specific prompts enabled
scorer = LLMBasedScorer({
    "use_domain_specific_prompts": True,
    "custom_templates_dir": "config/prompt_templates"
})

# Score a response with domain context
result = scorer.score(
    response="My child is starting to crawl but needs help sometimes.",
    milestone_context={
        "behavior": "Crawling",
        "criteria": "Child moves on hands and knees across the floor",
        "age_range": "6-10 months",
        "domain": "motor"
    }
)
```

## Performance Impact

Based on benchmarking, domain-specific prompts have shown the following impacts:

- **Accuracy**: Improved accuracy for cognitive and motor domains
- **Confidence**: Consistent high confidence across domains
- **Processing Time**: Slight increase in processing time (approximately 2 seconds per sample)

## Future Improvements

1. **Fine-tuning**: Further refinement of domain-specific prompts based on expert feedback
2. **Milestone-Specific Prompts**: More granular prompts for specific milestone types
3. **Age-Specific Guidance**: Incorporate age-specific considerations into prompts
4. **Cross-Domain Integration**: Better handling of milestones that span multiple domains
5. **Multilingual Support**: Templates for multiple languages 