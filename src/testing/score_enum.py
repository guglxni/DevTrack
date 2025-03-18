"""
Simplified Score enum for testing purposes.
This avoids dependencies on the core scoring module.
"""

from enum import Enum

class Score(Enum):
    """Score categories for developmental milestones"""
    CANNOT_DO = 0
    LOST_SKILL = 1
    EMERGING = 2
    WITH_SUPPORT = 3
    INDEPENDENT = 4
    NOT_RATED = -1 