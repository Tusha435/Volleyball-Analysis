"""
Team assignment system for volleyball players based on court positioning.

This module provides functionality to assign volleyball players to teams by
analyzing their position relative to the net line. Team classification uses
geometric principles to determine which side of the court each player occupies.
"""
import numpy as np
from collections import Counter


class TeamAssigner:
    """
    Assigns players to teams based on their position relative to the volleyball net.

    This classifier uses a geometric approach to determine team membership by
    calculating which side of the net line a player is positioned on. The system
    maintains a voting mechanism to ensure stable team assignments even when
    players move near the net or when detection noise occurs.

    The net line is defined by two points on the court, and team assignment is
    determined by calculating the cross product between the net vector and the
    player position vector. This approach is robust to camera angle and net
    orientation.

    Attributes:
        net_p1: First endpoint of the net line as a numpy array (x, y coordinates).
        net_p2: Second endpoint of the net line as a numpy array (x, y coordinates).
    """

    def __init__(self):
        """
        Initializes the team assigner with no net position defined.

        The net position must be set using set_net() or update_net_position()
        before team assignments can be made.
        """
        self.net_p1 = None
        self.net_p2 = None

    def set_net(self, p1, p2):
        """
        Establishes the initial net line position on the court.

        Args:
            p1: Tuple or array-like containing (x, y) coordinates for the first
                endpoint of the net line.
            p2: Tuple or array-like containing (x, y) coordinates for the second
                endpoint of the net line.
        """
        self.net_p1 = np.array(p1)
        self.net_p2 = np.array(p2)

    def update_net_position(self, p1, p2):
        """
        Updates the net line position to account for camera movement or corrections.

        This method allows dynamic adjustment of the net line during video processing,
        enabling the system to handle camera motion or refinements to the net position
        as more information becomes available.

        Args:
            p1: Tuple or array-like containing (x, y) coordinates for the first
                endpoint of the net line.
            p2: Tuple or array-like containing (x, y) coordinates for the second
                endpoint of the net line.
        """
        self.net_p1 = np.array(p1)
        self.net_p2 = np.array(p2)

    def assign(self, track):
        """
        Assigns a team designation to a player track based on court position.

        This method calculates which side of the net a player is on using the
        cross product of the net vector and the player position vector. The
        assignment is added to the track's voting history, and a final team
        designation is made once sufficient votes have been collected.

        The voting mechanism prevents flickering team assignments when players
        move near the net line or when temporary detection errors occur.

        Args:
            track: Track object representing a player. The track must have a
                centroid attribute containing (x, y) coordinates and team_votes
                and team attributes for storing assignment data.
        """
        if self.net_p1 is None or self.net_p2 is None:
            return

        cx, cy = track.centroid
        net_vec = self.net_p2 - self.net_p1
        player_vec = np.array([cx, cy]) - self.net_p1

        cross = net_vec[0] * player_vec[1] - net_vec[1] * player_vec[0]
        new_team = 'A' if cross > 0 else 'B'

        track.team_votes.append(new_team)

        if len(track.team_votes) >= 5:
            most_common = Counter(track.team_votes).most_common(1)[0][0]
            track.team = most_common
