"""
Object classification modules for volleyball game analysis.

This package provides specialized classification systems for volleyball tracking:
    - PositionLockedRefereeClassifier: Identifies referees using spatial anchors
    - TeamAssigner: Assigns players to teams based on court geometry

The referee classifier uses position-based scoring with camera motion compensation
to maintain stable referee identification. The team assigner uses cross-product
geometry to determine which side of the net each player is on, with a voting
mechanism for temporal stability.
"""

from .referee_classifier import PositionLockedRefereeClassifier
from .team_assigner import TeamAssigner

__all__ = ['PositionLockedRefereeClassifier', 'TeamAssigner']
