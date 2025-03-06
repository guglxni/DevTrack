#!/usr/bin/env python3
"""
ASD Developmental Milestone Assessment API - Test Configurations

This script provides predefined test scenarios for different ages and developmental profiles.
It can be used to generate realistic test data for specific developmental patterns.
"""

from enhanced_assessment_engine import Score

# Standard developmental profiles for different ages
# These dictate what percentage of milestones a child would score at each level
# Format: {Score_Enum: percentage}

# Neurotypical profiles by age
NEUROTYPICAL_PROFILES = {
    # Early ages (0-12 months)
    6: {
        Score.INDEPENDENT: 0.6,    # 60% mastered skills
        Score.WITH_SUPPORT: 0.2,   # 20% skills with help  
        Score.EMERGING: 0.15,      # 15% emerging skills
        Score.CANNOT_DO: 0.05,     # 5% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression
    },
    12: {
        Score.INDEPENDENT: 0.65,   # 65% mastered skills
        Score.WITH_SUPPORT: 0.2,   # 20% skills with help
        Score.EMERGING: 0.1,       # 10% emerging skills
        Score.CANNOT_DO: 0.05,     # 5% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression
    },
    
    # Middle ages (18-24 months)
    18: {
        Score.INDEPENDENT: 0.7,    # 70% mastered skills
        Score.WITH_SUPPORT: 0.15,  # 15% skills with help
        Score.EMERGING: 0.1,       # 10% emerging skills
        Score.CANNOT_DO: 0.05,     # 5% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression
    },
    24: {
        Score.INDEPENDENT: 0.75,   # 75% mastered skills
        Score.WITH_SUPPORT: 0.15,  # 15% skills with help
        Score.EMERGING: 0.08,      # 8% emerging skills
        Score.CANNOT_DO: 0.02,     # 2% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression
    },
    
    # Later ages (30-36 months)
    30: {
        Score.INDEPENDENT: 0.8,    # 80% mastered skills
        Score.WITH_SUPPORT: 0.12,  # 12% skills with help
        Score.EMERGING: 0.06,      # 6% emerging skills
        Score.CANNOT_DO: 0.02,     # 2% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression
    },
    36: {
        Score.INDEPENDENT: 0.85,   # 85% mastered skills
        Score.WITH_SUPPORT: 0.1,   # 10% skills with help
        Score.EMERGING: 0.04,      # 4% emerging skills
        Score.CANNOT_DO: 0.01,     # 1% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression
    }
}

# Developmental delay profiles (general delays across domains)
DELAY_PROFILES = {
    # Early ages (0-12 months)
    6: {
        Score.INDEPENDENT: 0.3,    # 30% mastered skills
        Score.WITH_SUPPORT: 0.3,   # 30% skills with help
        Score.EMERGING: 0.2,       # 20% emerging skills
        Score.CANNOT_DO: 0.2,      # 20% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression
    },
    12: {
        Score.INDEPENDENT: 0.35,   # 35% mastered skills
        Score.WITH_SUPPORT: 0.3,   # 30% skills with help
        Score.EMERGING: 0.2,       # 20% emerging skills
        Score.CANNOT_DO: 0.15,     # 15% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression
    },
    
    # Middle ages (18-24 months)
    18: {
        Score.INDEPENDENT: 0.4,    # 40% mastered skills
        Score.WITH_SUPPORT: 0.3,   # 30% skills with help
        Score.EMERGING: 0.2,       # 20% emerging skills
        Score.CANNOT_DO: 0.1,      # 10% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression
    },
    24: {
        Score.INDEPENDENT: 0.45,   # 45% mastered skills
        Score.WITH_SUPPORT: 0.25,  # 25% skills with help
        Score.EMERGING: 0.2,       # 20% emerging skills
        Score.CANNOT_DO: 0.1,      # 10% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression
    },
    
    # Later ages (30-36 months)
    30: {
        Score.INDEPENDENT: 0.5,    # 50% mastered skills
        Score.WITH_SUPPORT: 0.25,  # 25% skills with help
        Score.EMERGING: 0.15,      # 15% emerging skills
        Score.CANNOT_DO: 0.1,      # 10% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression
    },
    36: {
        Score.INDEPENDENT: 0.55,   # 55% mastered skills
        Score.WITH_SUPPORT: 0.25,  # 25% skills with help
        Score.EMERGING: 0.12,      # 12% emerging skills
        Score.CANNOT_DO: 0.08,     # 8% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression
    }
}

# ASD profile with regression patterns
ASD_PROFILES = {
    # Early ages (0-12 months)
    6: {
        Score.INDEPENDENT: 0.4,    # 40% mastered skills
        Score.WITH_SUPPORT: 0.25,  # 25% skills with help
        Score.EMERGING: 0.2,       # 20% emerging skills
        Score.CANNOT_DO: 0.15,     # 15% cannot do yet
        Score.LOST_SKILL: 0.0      # 0% regression (typically not seen yet)
    },
    12: {
        Score.INDEPENDENT: 0.3,    # 30% mastered skills
        Score.WITH_SUPPORT: 0.2,   # 20% skills with help
        Score.EMERGING: 0.2,       # 20% emerging skills
        Score.CANNOT_DO: 0.2,      # 20% cannot do yet
        Score.LOST_SKILL: 0.1      # 10% regression (beginning to show)
    },
    
    # Middle ages (18-24 months) - regression often becomes apparent
    18: {
        Score.INDEPENDENT: 0.25,   # 25% mastered skills
        Score.WITH_SUPPORT: 0.2,   # 20% skills with help
        Score.EMERGING: 0.15,      # 15% emerging skills
        Score.CANNOT_DO: 0.25,     # 25% cannot do yet
        Score.LOST_SKILL: 0.15     # 15% regression (more obvious)
    },
    24: {
        Score.INDEPENDENT: 0.2,    # 20% mastered skills
        Score.WITH_SUPPORT: 0.15,  # 15% skills with help
        Score.EMERGING: 0.15,      # 15% emerging skills
        Score.CANNOT_DO: 0.3,      # 30% cannot do yet
        Score.LOST_SKILL: 0.2      # 20% regression (significant)
    },
    
    # Later ages (30-36 months) - pattern established
    30: {
        Score.INDEPENDENT: 0.15,   # 15% mastered skills
        Score.WITH_SUPPORT: 0.15,  # 15% skills with help
        Score.EMERGING: 0.15,      # 15% emerging skills
        Score.CANNOT_DO: 0.35,     # 35% cannot do yet
        Score.LOST_SKILL: 0.2      # 20% regression (maintained)
    },
    36: {
        Score.INDEPENDENT: 0.1,    # 10% mastered skills
        Score.WITH_SUPPORT: 0.15,  # 15% skills with help
        Score.EMERGING: 0.15,      # 15% emerging skills
        Score.CANNOT_DO: 0.4,      # 40% cannot do yet
        Score.LOST_SKILL: 0.2      # 20% regression (maintained)
    }
}

# Specific domain profiles - showing strength in some areas, weakness in others
# This represents uneven development patterns often seen in neurodevelopmental conditions

# Profile with strong motor skills, weak language and social skills
UNEVEN_MOTOR_STRONG_PROFILES = {
    24: {  # Example for 24 months
        # Motor domains
        "GM": {  # Gross Motor
            Score.INDEPENDENT: 0.7,
            Score.WITH_SUPPORT: 0.2,
            Score.EMERGING: 0.05,
            Score.CANNOT_DO: 0.05,
            Score.LOST_SKILL: 0.0
        },
        "FM": {  # Fine Motor
            Score.INDEPENDENT: 0.65,
            Score.WITH_SUPPORT: 0.25,
            Score.EMERGING: 0.05,
            Score.CANNOT_DO: 0.05,
            Score.LOST_SKILL: 0.0
        },
        
        # Language domains
        "RL": {  # Receptive Language
            Score.INDEPENDENT: 0.2,
            Score.WITH_SUPPORT: 0.2,
            Score.EMERGING: 0.25,
            Score.CANNOT_DO: 0.25,
            Score.LOST_SKILL: 0.1
        },
        "EL": {  # Expressive Language
            Score.INDEPENDENT: 0.15,
            Score.WITH_SUPPORT: 0.2,
            Score.EMERGING: 0.3,
            Score.CANNOT_DO: 0.3,
            Score.LOST_SKILL: 0.05
        },
        
        # Social-emotional domains
        "SOC": {  # Social
            Score.INDEPENDENT: 0.2,
            Score.WITH_SUPPORT: 0.2,
            Score.EMERGING: 0.3,
            Score.CANNOT_DO: 0.2,
            Score.LOST_SKILL: 0.1
        },
        "EMO": {  # Emotional
            Score.INDEPENDENT: 0.25,
            Score.WITH_SUPPORT: 0.25,
            Score.EMERGING: 0.2,
            Score.CANNOT_DO: 0.2,
            Score.LOST_SKILL: 0.1
        },
        
        # Other domains
        "ADL": {  # Activities of Daily Living
            Score.INDEPENDENT: 0.3,
            Score.WITH_SUPPORT: 0.3,
            Score.EMERGING: 0.2,
            Score.CANNOT_DO: 0.15,
            Score.LOST_SKILL: 0.05
        },
        "COG": {  # Cognitive
            Score.INDEPENDENT: 0.35,
            Score.WITH_SUPPORT: 0.25,
            Score.EMERGING: 0.2,
            Score.CANNOT_DO: 0.15,
            Score.LOST_SKILL: 0.05
        }
    }
}

# Profile with strong cognitive and language skills, weak motor and social skills
UNEVEN_COGNITIVE_STRONG_PROFILES = {
    24: {  # Example for 24 months
        # Motor domains
        "GM": {  # Gross Motor
            Score.INDEPENDENT: 0.3,
            Score.WITH_SUPPORT: 0.25,
            Score.EMERGING: 0.2,
            Score.CANNOT_DO: 0.2,
            Score.LOST_SKILL: 0.05
        },
        "FM": {  # Fine Motor
            Score.INDEPENDENT: 0.35,
            Score.WITH_SUPPORT: 0.25,
            Score.EMERGING: 0.2,
            Score.CANNOT_DO: 0.15,
            Score.LOST_SKILL: 0.05
        },
        
        # Language domains
        "RL": {  # Receptive Language
            Score.INDEPENDENT: 0.65,
            Score.WITH_SUPPORT: 0.2,
            Score.EMERGING: 0.1,
            Score.CANNOT_DO: 0.05,
            Score.LOST_SKILL: 0.0
        },
        "EL": {  # Expressive Language
            Score.INDEPENDENT: 0.6,
            Score.WITH_SUPPORT: 0.25,
            Score.EMERGING: 0.1,
            Score.CANNOT_DO: 0.05,
            Score.LOST_SKILL: 0.0
        },
        
        # Social-emotional domains
        "SOC": {  # Social
            Score.INDEPENDENT: 0.2,
            Score.WITH_SUPPORT: 0.2,
            Score.EMERGING: 0.3,
            Score.CANNOT_DO: 0.2,
            Score.LOST_SKILL: 0.1
        },
        "EMO": {  # Emotional
            Score.INDEPENDENT: 0.25,
            Score.WITH_SUPPORT: 0.25,
            Score.EMERGING: 0.2,
            Score.CANNOT_DO: 0.2,
            Score.LOST_SKILL: 0.1
        },
        
        # Other domains
        "ADL": {  # Activities of Daily Living
            Score.INDEPENDENT: 0.4,
            Score.WITH_SUPPORT: 0.3,
            Score.EMERGING: 0.15,
            Score.CANNOT_DO: 0.15,
            Score.LOST_SKILL: 0.0
        },
        "COG": {  # Cognitive
            Score.INDEPENDENT: 0.7,
            Score.WITH_SUPPORT: 0.2,
            Score.EMERGING: 0.05,
            Score.CANNOT_DO: 0.05,
            Score.LOST_SKILL: 0.0
        }
    }
}

# Function to get a profile based on age and type
def get_profile(age, profile_type="neurotypical", domain=None):
    """
    Get a developmental profile distribution based on age and type
    
    Args:
        age: Child's age in months
        profile_type: Type of profile (neurotypical, delay, asd, uneven_motor, uneven_cognitive)
        domain: Specific domain to get profile for (used with uneven profiles)
        
    Returns:
        Dictionary with score distributions
    """
    # Find the nearest age profile
    available_ages = [6, 12, 18, 24, 30, 36]
    nearest_age = min(available_ages, key=lambda x: abs(x - age))
    
    # Select profile type
    if profile_type == "neurotypical":
        profile = NEUROTYPICAL_PROFILES.get(nearest_age, {})
    elif profile_type == "delay":
        profile = DELAY_PROFILES.get(nearest_age, {})
    elif profile_type == "asd":
        profile = ASD_PROFILES.get(nearest_age, {})
    elif profile_type == "uneven_motor":
        age_profile = UNEVEN_MOTOR_STRONG_PROFILES.get(nearest_age, {})
        profile = age_profile.get(domain) if domain else age_profile
    elif profile_type == "uneven_cognitive":
        age_profile = UNEVEN_COGNITIVE_STRONG_PROFILES.get(nearest_age, {})
        profile = age_profile.get(domain) if domain else age_profile
    else:
        # Default to neurotypical
        profile = NEUROTYPICAL_PROFILES.get(nearest_age, {})
    
    return profile

# Example usage
if __name__ == "__main__":
    import json
    
    # Print example profiles
    print("Example profiles:")
    print("\nNeurotypical 24-month-old:")
    print(json.dumps({k.name: v for k, v in get_profile(24, "neurotypical").items()}, indent=2))
    
    print("\nDevelopmental delay 24-month-old:")
    print(json.dumps({k.name: v for k, v in get_profile(24, "delay").items()}, indent=2))
    
    print("\nASD profile 24-month-old:")
    print(json.dumps({k.name: v for k, v in get_profile(24, "asd").items()}, indent=2))
    
    print("\nUneven development (motor strong) 24-month-old - GM domain:")
    print(json.dumps({k.name: v for k, v in get_profile(24, "uneven_motor", "GM").items()}, indent=2)) 