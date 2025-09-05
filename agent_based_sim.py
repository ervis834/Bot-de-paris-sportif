"""Agent-based simulation for football matches with 22 individual players."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logger = logging.getLogger(__name__)


class PlayerPosition(Enum):
    """Player positions on the field."""
    GOALKEEPER = "GK"
    DEFENDER = "DEF" 
    MIDFIELDER = "MID"
    FORWARD = "FWD"


class ActionType(Enum):
    """Types of actions players can perform."""
    PASS = "pass"
    SHOOT = "shoot"
    DRIBBLE = "dribble"
    TACKLE = "tackle"
    INTERCEPT = "intercept"
    MOVE = "move"
    SAVE = "save"


@dataclass
class PlayerState:
    """Current state of a player."""
    player_id: int
    team: int  # 0 or 1
    position: PlayerPosition
    x: float  # Field position (0-100)
    y: float  # Field position (0-100)
    has_ball: bool = False
    stamina: float = 100.0
    skill: float = 70.0  # 0-100 scale
    speed: float = 70.0
    aggression: float = 50.0
    last_action: Optional[ActionType] = None


@dataclass
class MatchState:
    """Current state of the match."""
    minute: int = 0
    score_home: int = 0
    score_away: int = 0
    ball_x: float = 50.0
    ball_y: float = 50.0
    ball_carrier: Optional[int] = None
    possession_team: int = 0
    phase: str = "play"  # play, corner, free_kick, goal_kick, etc.


class FootballAgent:
    """Individual football player agent."""
    
    def __init__(self, player_id: int, team: int, position: PlayerPosition, 
                 x: float, y: float, attributes: Dict[str, float]):
        self.state = PlayerState(
            player_id=player_id,
            team=team,
            position=position,
            x=x,
            y=y,
            skill=attributes.get('skill', 70.0),
            speed=attributes.get('speed', 70.0),
            stamina=attributes.get('stamina', 100.0),
            aggression=attributes.get('aggression', 50.0)
        )
        
        # Tactical instructions
        self.formation_x = x
        self.formation_y = y
        self.role = self._determine_role()
        
    def _determine_role(self) -> str:
        """Determine player's tactical role."""
        if self.state.position == PlayerPosition.GOALKEEPER:
            return "goalkeeper"
        elif self.state.position == PlayerPosition.DEFENDER:
            return "defender"
        elif self.state.position == PlayerPosition.MIDFIELDER:
            return "midfielder"  
        else:
            return "forward"
    
    def decide_action(self, match_state: MatchState, nearby_players: List['FootballAgent']) -> Tuple[ActionType, Dict]:
        """Decide what action to take based on current situation."""
        if self.state.has_ball:
            return self._decide_with_ball(match_state, nearby_players)
        else:
            return self._decide_without_ball(match_state, nearby_players)
    
    def _decide_with_ball(self, match_state: MatchState, nearby_players: List['FootballAgent']) -> Tuple[ActionType, Dict]:
        """Decide action when player has the ball."""
        # Calculate distances and options
        goal_distance = self._distance_to_goal(match_state)
        teammates = [p for p in nearby_players if p.state.team == self.state.team and p.state.player_id != self.state.player_id]
        opponents = [p for p in nearby_players if p.state.team != self.state.team]
        
        # Shooting decision
        if goal_distance < 25 and self.role in ["forward", "midfielder"] and random.random() < 0.3:
            shooting_angle = self._calculate_shooting_angle()
            return ActionType.SHOOT, {
                'target_x': 100 if self.state.team == 0 else 0,
                'target_y': 50 + random.uniform(-10, 10),
                'power': min(100, self.state.skill + random.uniform(-20, 20)),
                'angle': shooting_angle
            }
        
        # Passing decision
        best_pass_target = self._find_best_pass_target(teammates, opponents)
        if best_pass_target and random.random() < 0.7:
            return ActionType.PASS, {
                'target_player': best_pass_target.state.player_id,
                'target_x': best_pass_target.state.x,
                'target_y': best_pass_target.state.y,
                'accuracy': self._calculate_pass_accuracy(best_pass_target)
            }
        
        # Dribbling decision
        if len(opponents) == 1 and random.random() < 0.4:
            return ActionType.DRIBBLE, {
                'direction': self._calculate_dribble_direction(opponents[0]),
                'success_prob': self._calculate_dribble_success(opponents[0])
            }
        
        # Default: safe pass or move
        if teammates:
            safe_target = min(teammates, key=lambda p: self._distance_to_player(p))
            return ActionType.PASS, {
                'target_player': safe_target.state.player_id,
                'target_x': safe_target.state.x,
                'target_y': safe_target.state.y,
                'accuracy': self._calculate_pass_accuracy(safe_target)
            }
        
        return ActionType.MOVE, {'direction': 'forward'}
    
    def _decide_without_ball(self, match_state: MatchState, nearby_players: List['FootballAgent']) -> Tuple[ActionType, Dict]:
        """Decide action when player doesn't have the ball."""
        ball_distance = np.sqrt((self.state.x - match_state.ball_x)**2 + (self.state.y - match_state.ball_y)**2)
        
        # If close to ball, try to get it
        if ball_distance < 5:
            ball_carrier = next((p for p in nearby_players if p.state.has_ball), None)
            
            if ball_carrier and ball_carrier.state.team != self.state.team:
                # Try to tackle
                return ActionType.TACKLE, {
                    'target_player': ball_carrier.state.player_id,
                    'success_prob': self._calculate_tackle_success(ball_carrier)
                }
            elif not ball_carrier:
                # Ball is free, move to it
                return ActionType.MOVE, {
                    'target_x': match_state.ball_x,
                    'target_y': match_state.ball_y
                }
        
        # Tactical movement
        if match_state.possession_team == self.state.team:
            # Support attack
            target_x, target_y = self._calculate_support_position(match_state)
        else:
            # Defend
            target_x, target_y = self._calculate_defensive_position(match_state)
        
        return ActionType.MOVE, {
            'target_x': target_x,
            'target_y': target_y
        }
    
    def _distance_to_goal(self, match_state: MatchState) -> float:
        """Calculate distance to opponent's goal."""
        goal_x = 100 if self.state.team == 0 else 0
        return np.sqrt((self.state.x - goal_x)**2 + (self.state.y - 50)**2)
    
    def _distance_to_player(self, other_player: 'FootballAgent') -> float:
        """Calculate distance to another player."""
        return np.sqrt((self.state.x - other_player.state.x)**2 + (self.state.y - other_player.state.y)**2)
    
    def _find_best_pass_target(self, teammates: List['FootballAgent'], opponents: List['FootballAgent']) -> Optional['FootballAgent']:
        """Find the best teammate to pass to."""
        if not teammates:
            return None
        
        best_target = None
        best_score = -1
        
        for teammate in teammates:
            # Calculate pass quality score
            distance = self._distance_to_player(teammate)
            if distance > 30:  # Too far
                continue
            
            # Check if pass lane is clear
            pass_blocked = any(
                self._is_player_blocking_pass(opponent, teammate) 
                for opponent in opponents
            )
            
            if pass_blocked:
                continue
            
            # Score based on position and distance
            forward_progress = teammate.state.x - self.state.x if self.state.team == 0 else self.state.x - teammate.state.x
            score = forward_progress * 0.5 + (30 - distance) * 0.3 + teammate.state.skill * 0.2
            
            if score > best_score:
                best_score = score
                best_target = teammate
        
        return best_target
    
    def _is_player_blocking_pass(self, opponent: 'FootballAgent', target: 'FootballAgent') -> bool:
        """Check if an opponent is blocking a potential pass."""
        # Simplified line intersection check
        pass_distance = self._distance_to_player(target)
        opponent_distance_to_line = self._point_to_line_distance(
            opponent.state.x, opponent.state.y,
            self.state.x, self.state.y,
            target.state.x, target.state.y
        )
        
        return opponent_distance_to_line < 3 and opponent_distance_to_line < pass_distance
    
    def _point_to_line_distance(self, px: float, py: float, 
                               x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate distance from point to line segment."""
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return np.sqrt(A * A + B * B)
        
        param = dot / len_sq
        
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
        
        dx = px - xx
        dy = py - yy
        return np.sqrt(dx * dx + dy * dy)
    
    def _calculate_pass_accuracy(self, target: 'FootballAgent') -> float:
        """Calculate probability of successful pass."""
        distance = self._distance_to_player(target)
        base_accuracy = self.state.skill / 100
        distance_penalty = min(0.3, distance / 100)
        stamina_factor = self.state.stamina / 100
        
        return max(0.1, base_accuracy - distance_penalty * stamina_factor)
    
    def _calculate_dribble_direction(self, opponent: 'FootballAgent') -> float:
        """Calculate best direction for dribbling past opponent."""
        # Simple: go around the opponent
        angle_to_opponent = np.arctan2(
            opponent.state.y - self.state.y,
            opponent.state.x - self.state.x
        )
        
        # Go perpendicular to avoid opponent
        return angle_to_opponent + np.pi / 2
    
    def _calculate_dribble_success(self, opponent: 'FootballAgent') -> float:
        """Calculate probability of successful dribble."""
        skill_diff = self.state.skill - opponent.state.skill
        speed_diff = self.state.speed - opponent.state.speed
        
        base_prob = 0.5
        skill_factor = skill_diff / 200  # Normalize
        speed_factor = speed_diff / 200
        
        return max(0.1, min(0.9, base_prob + skill_factor + speed_factor))
    
    def _calculate_tackle_success(self, ball_carrier: 'FootballAgent') -> float:
        """Calculate probability of successful tackle."""
        skill_diff = self.state.skill - ball_carrier.state.skill
        aggression_factor = self.state.aggression / 100
        
        base_prob = 0.4
        skill_factor = skill_diff / 200
        
        return max(0.1, min(0.8, base_prob + skill_factor * aggression_factor))
    
    def _calculate_shooting_angle(self) -> float:
        """Calculate shooting angle towards goal."""
        goal_x = 100 if self.state.team == 0 else 0
        goal_y = 50
        
        return np.arctan2(goal_y - self.state.y, goal_x - self.state.x)
    
    def _calculate_support_position(self, match_state: MatchState) -> Tuple[float, float]:
        """Calculate position to support attack."""
        # Move towards opponent's goal while maintaining formation
        if self.state.team == 0:
            target_x = min(90, self.formation_x + 10)
        else:
            target_x = max(10, self.formation_x - 10)
        
        target_y = self.formation_y + random.uniform(-5, 5)
        return target_x, max(5, min(95, target_y))
    
    def _calculate_defensive_position(self, match_state: MatchState) -> Tuple[float, float]:
        """Calculate position for defending."""
        # Move towards own goal while tracking ball
        ball_y_influence = (match_state.ball_y - 50) * 0.3
        
        if self.state.team == 0:
            target_x = max(10, self.formation_x - 5)
        else:
            target_x = min(90, self.formation_x + 5)
        
        target_y = self.formation_y + ball_y_influence
        return target_x, max(5, min(95, target_y))
    
    def execute_action(self, action: ActionType, params: Dict, match_state: MatchState) -> Dict:
        """Execute the decided action and return result."""
        result = {'success': False, 'description': ''}
        
        if action == ActionType.PASS:
            success_prob = params['accuracy']
            success = random.random() < success_prob
            
            if success:
                # Pass successful - ball moves to target
                result['success'] = True
                result['description'] = f"Player {self.state.player_id} passes to player {params['target_player']}"
                result['new_ball_x'] = params['target_x']
                result['new_ball_y'] = params['target_y']
                result['new_ball_carrier'] = params['target_player']
            else:
                # Pass intercepted or misplaced
                result['description'] = f"Player {self.state.player_id} pass failed"
                result['turnover'] = True
        
        elif action == ActionType.SHOOT:
            # Calculate shot success based on distance, angle, and skill
            goal_distance = self._distance_to_goal(match_state)
            shot_power = params['power']
            
            # Base probability decreases with distance
            base_prob = max(0.05, 0.4 - goal_distance / 100)
            skill_factor = self.state.skill / 100
            power_factor = min(1.0, shot_power / 80)
            
            shot_prob = base_prob * skill_factor * power_factor
            
            if random.random() < shot_prob:
                result['success'] = True
                result['description'] = f"GOAL! Player {self.state.player_id} scores!"
                result['goal'] = True
                result['scoring_team'] = self.state.team
            else:
                result['description'] = f"Player {self.state.player_id} shot missed"
                result['shot_missed'] = True
        
        elif action == ActionType.TACKLE:
            success_prob = params['success_prob']
            success = random.random() < success_prob
            
            if success:
                result['success'] = True
                result['description'] = f"Player {self.state.player_id} wins the ball"
                result['ball_won'] = True
                result['new_ball_carrier'] = self.state.player_id
            else:
                result['description'] = f"Player {self.state.player_id} tackle failed"
        
        elif action == ActionType.DRIBBLE:
            success_prob = params['success_prob']
            success = random.random() < success_prob
            
            if success:
                result['success'] = True
                result['description'] = f"Player {self.state.player_id} successful dribble"
                # Move player forward
                direction = params['direction']
                distance = 5
                new_x = self.state.x + distance * np.cos(direction)
                new_y = self.state.y + distance * np.sin(direction)
                result['new_player_x'] = max(0, min(100, new_x))
                result['new_player_y'] = max(0, min(100, new_y))
            else:
                result['description'] = f"Player {self.state.player_id} dribble failed"
                result['turnover'] = True
        
        elif action == ActionType.MOVE:
            # Always successful
            result['success'] = True
            target_x = params.get('target_x', self.state.x)
            target_y = params.get('target_y', self.state.y)
            
            # Move towards target (with speed limitations)
            max_movement = self.state.speed / 10  # Speed affects movement range
            
            dx = target_x - self.state.x
            dy = target_y - self.state.y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > max_movement:
                # Can't reach target in one step
                dx = (dx / distance) * max_movement
                dy = (dy / distance) * max_movement
            
            result['new_player_x'] = max(0, min(100, self.state.x + dx))
            result['new_player_y'] = max(0, min(100, self.state.y + dy))
            result['description'] = f"Player {self.state.player_id} moves"
        
        # Update stamina
        self.state.stamina = max(0, self.state.stamina - 0.5)
        
        return result


class AgentBasedFootballSimulation:
    """Main agent-based football simulation."""
    
    def __init__(self, home_team_attributes: Dict, away_team_attributes: Dict):
        self.match_state = MatchState()
        self.home_agents = []
        self.away_agents = []
        self.match_events = []
        
        # Create agents
        self._initialize_teams(home_team_attributes, away_team_attributes)
        
        # Simulation parameters
        self.max_minutes = 90
        self.steps_per_minute = 10
        self.max_steps = self.max_minutes * self.steps_per_minute
        
    def _initialize_teams(self, home_attrs: Dict, away_attrs: Dict):
        """Initialize both teams with their formations."""
        # Standard 4-4-2 formation positions
        home_positions = [
            (10, 50, PlayerPosition.GOALKEEPER),   # GK
            (20, 20, PlayerPosition.DEFENDER),     # RB
            (20, 35, PlayerPosition.DEFENDER),     # CB
            (20, 65, PlayerPosition.DEFENDER),     # CB
            (20, 80, PlayerPosition.DEFENDER),     # LB
            (40, 25, PlayerPosition.MIDFIELDER),   # RM
            (40, 40, PlayerPosition.MIDFIELDER),   # CM
            (40, 60, PlayerPosition.MIDFIELDER),   # CM
            (40, 75, PlayerPosition.MIDFIELDER),   # LM
            (70, 40, PlayerPosition.FORWARD),      # ST
            (70, 60, PlayerPosition.FORWARD),      # ST
        ]
        
        away_positions = [
            (90, 50, PlayerPosition.GOALKEEPER),   # GK
            (80, 80, PlayerPosition.DEFENDER),     # RB (mirrored)
            (80, 65, PlayerPosition.DEFENDER),     # CB
            (80, 35, PlayerPosition.DEFENDER),     # CB
            (80, 20, PlayerPosition.DEFENDER),     # LB
            (60, 75, PlayerPosition.MIDFIELDER),   # RM (mirrored)
            (60, 60, PlayerPosition.MIDFIELDER),   # CM
            (60, 40, PlayerPosition.MIDFIELDER),   # CM
            (60, 25, PlayerPosition.MIDFIELDER),   # LM
            (30, 60, PlayerPosition.FORWARD),      # ST (mirrored)
            (30, 40, PlayerPosition.FORWARD),      # ST
        ]
        
        # Create home team agents
        for i, (x, y, position) in enumerate(home_positions):
            agent = FootballAgent(
                player_id=i,
                team=0,
                position=position,
                x=x, y=y,
                attributes=home_attrs
            )
            self.home_agents.append(agent)
        
        # Create away team agents
        for i, (x, y, position) in enumerate(away_positions):
            agent = FootballAgent(
                player_id=i + 11,  # Different IDs
                team=1,
                position=position,
                x=x, y=y,
                attributes=away_attrs
            )
            self.away_agents.append(agent)
        
        # Home team starts with ball
        self.home_agents[9].state.has_ball = True  # Forward has ball
        self.match_state.ball_x = self.home_agents[9].state.x
        self.match_state.ball_y = self.home_agents[9].state.y
        self.match_state.ball_carrier = self.home_agents[9].state.player_id
    
    def simulate_match(self, max_minutes: int = 90) -> Dict:
        """Simulate a complete match."""
        self.max_minutes = max_minutes
        self.max_steps = max_minutes * self.steps_per_minute
        
        logger.info(f"Starting agent-based simulation: {max_minutes} minutes")
        
        for step in range(self.max_steps):
            self.match_state.minute = step // self.steps_per_minute
            
            # Simulate one time step
            self._simulate_step()
            
            # Check if match should end early (rare)
            if self._should_end_match():
                break
        
        # Calculate final statistics
        result = self._calculate_match_result()
        
        logger.info(f"Match finished: {result['score_home']}-{result['score_away']}")
        
        return result
    
    def _simulate_step(self):
        """Simulate one time step of the match."""
        all_agents = self.home_agents + self.away_agents
        
        # Find ball carrier
        ball_carrier = next((agent for agent in all_agents if agent.state.has_ball), None)
        
        if not ball_carrier:
            # Ball is loose - find closest player
            closest_agent = min(all_agents, key=lambda a: np.sqrt(
                (a.state.x - self.match_state.ball_x)**2 + 
                (a.state.y - self.match_state.ball_y)**2
            ))
            if np.sqrt((closest_agent.state.x - self.match_state.ball_x)**2 + 
                      (closest_agent.state.y - self.match_state.ball_y)**2) < 2:
                closest_agent.state.has_ball = True
                self.match_state.ball_carrier = closest_agent.state.player_id
                self.match_state.possession_team = closest_agent.state.team
                ball_carrier = closest_agent
        
        # Each agent decides and executes action
        for agent in all_agents:
            # Find nearby players (within 20 units)
            nearby_players = [
                other for other in all_agents 
                if other.state.player_id != agent.state.player_id and
                np.sqrt((agent.state.x - other.state.x)**2 + (agent.state.y - other.state.y)**2) < 20
            ]
            
            # Agent decides action
            action, params = agent.decide_action(self.match_state, nearby_players)
            
            # Execute action
            result = agent.execute_action(action, params, self.match_state)
            
            # Process result
            self._process_action_result(agent, result)
    
    def _process_action_result(self, agent: FootballAgent, result: Dict):
        """Process the result of an agent's action."""
        if not result['success']:
            return
        
        # Update positions
        if 'new_player_x' in result:
            agent.state.x = result['new_player_x']
            agent.state.y = result['new_player_y']
            
            # Update ball position if player has ball
            if agent.state.has_ball:
                self.match_state.ball_x = agent.state.x
                self.match_state.ball_y = agent.state.y
        
        # Handle ball movement
        if 'new_ball_x' in result:
            self.match_state.ball_x = result['new_ball_x']
            self.match_state.ball_y = result['new_ball_y']
        
        # Handle possession changes
        if 'new_ball_carrier' in result:
            # Remove ball from all players
            for a in self.home_agents + self.away_agents:
                a.state.has_ball = False
            
            # Give ball to new carrier
            new_carrier_id = result['new_ball_carrier']
            all_agents = self.home_agents + self.away_agents
            new_carrier = next((a for a in all_agents if a.state.player_id == new_carrier_id), None)
            
            if new_carrier:
                new_carrier.state.has_ball = True
                self.match_state.ball_carrier = new_carrier_id
                self.match_state.possession_team = new_carrier.state.team
        
        # Handle turnovers
        if result.get('turnover'):
            # Give ball to random opponent
            current_team = self.match_state.possession_team
            opposing_team = 1 - current_team
            
            if opposing_team == 0:
                candidates = self.home_agents
            else:
                candidates = self.away_agents
            
            # Find closest opponent to ball
            closest_opponent = min(candidates, key=lambda a: np.sqrt(
                (a.state.x - self.match_state.ball_x)**2 + 
                (a.state.y - self.match_state.ball_y)**2
            ))
            
            # Remove ball from all players
            for a in self.home_agents + self.away_agents:
                a.state.has_ball = False
            
            closest_opponent.state.has_ball = True
            self.match_state.ball_carrier = closest_opponent.state.player_id
            self.match_state.possession_team = opposing_team
        
        # Handle goals
        if result.get('goal'):
            scoring_team = result['scoring_team']
            if scoring_team == 0:
                self.match_state.score_home += 1
            else:
                self.match_state.score_away += 1
            
            # Reset for kickoff
            self._reset_for_kickoff(1 - scoring_team)  # Non-scoring team kicks off
            
            # Record event
            self.match_events.append({
                'minute': self.match_state.minute,
                'event': 'goal',
                'team': scoring_team,
                'player': agent.state.player_id,
                'description': result['description']
            })
        
        # Record significant events
        if result.get('goal') or result.get('ball_won'):
            self.match_events.append({
                'minute': self.match_state.minute,
                'event': agent.state.last_action.value if agent.state.last_action else 'unknown',
                'team': agent.state.team,
                'player': agent.state.player_id,
                'description': result['description']
            })
    
    def _reset_for_kickoff(self, kicking_team: int):
        """Reset positions for kickoff."""
        # Reset all player positions to formation
        for agent in self.home_agents:
            agent.state.x = agent.formation_x
            agent.state.y = agent.formation_y
            agent.state.has_ball = False
        
        for agent in self.away_agents:
            agent.state.x = agent.formation_x
            agent.state.y = agent.formation_y
            agent.state.has_ball = False
        
        # Center the ball and give to kicking team
        self.match_state.ball_x = 50
        self.match_state.ball_y = 50
        
        if kicking_team == 0:
            kicker = self.home_agents[9]  # Forward
        else:
            kicker = self.away_agents[9]   # Forward
        
        kicker.state.has_ball = True
        self.match_state.ball_carrier = kicker.state.player_id
        self.match_state.possession_team = kicking_team
    
    def _should_end_match(self) -> bool:
        """Check if match should end early (rare circumstances)."""
        # Could implement early endings for extreme scores, red cards, etc.
        return False
    
    def _calculate_match_result(self) -> Dict:
        """Calculate final match statistics."""
        # Basic possession calculation
        possession_events = [event for event in self.match_events if 'ball_won' in event.get('description', '')]
        
        home_possession_events = len([e for e in possession_events if e['team'] == 0])
        away_possession_events = len([e for e in possession_events if e['team'] == 1])
        total_possession_events = home_possession_events + away_possession_events
        
        if total_possession_events > 0:
            home_possession = home_possession_events / total_possession_events
            away_possession = away_possession_events / total_possession_events
        else:
            home_possession = 0.5
            away_possession = 0.5
        
        # Count shots
        shot_events = [event for event in self.match_events if 'shot' in event.get('description', '').lower()]
        home_shots = len([e for e in shot_events if e['team'] == 0])
        away_shots = len([e for e in shot_events if e['team'] == 1])
        
        # Count goals
        goal_events = [event for event in self.match_events if event.get('event') == 'goal']
        
        return {
            'score_home': self.match_state.score_home,
            'score_away': self.match_state.score_away,
            'possession_home': home_possession,
            'possession_away': away_possession,
            'shots_home': home_shots,
            'shots_away': away_shots,
            'goals_home': len([e for e in goal_events if e['team'] == 0]),
            'goals_away': len([e for e in goal_events if e['team'] == 1]),
            'total_events': len(self.match_events),
            'match_events': self.match_events,
            'final_minute': self.match_state.minute
        }
    
    def visualize_match_state(self, save_path: Optional[str] = None):
        """Create a visualization of the current match state."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Draw field
        field = patches.Rectangle((0, 0), 100, 100, linewidth=2, edgecolor='white', facecolor='green', alpha=0.3)
        ax.add_patch(field)
        
        # Draw center circle
        center_circle = patches.Circle((50, 50), 10, linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(center_circle)
        
        # Draw penalty areas
        home_penalty = patches.Rectangle((0, 25), 18, 50, linewidth=2, edgecolor='white', facecolor='none')
        away_penalty = patches.Rectangle((82, 25), 18, 50, linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(home_penalty)
        ax.add_patch(away_penalty)
        
        # Draw goals
        home_goal = patches.Rectangle((0, 40), 5, 20, linewidth=3, edgecolor='white', facecolor='none')
        away_goal = patches.Rectangle((95, 40), 5, 20, linewidth=3, edgecolor='white', facecolor='none')
        ax.add_patch(home_goal)
        ax.add_patch(away_goal)
        
        # Plot players
        home_x = [agent.state.x for agent in self.home_agents]
        home_y = [agent.state.y for agent in self.home_agents]
        away_x = [agent.state.x for agent in self.away_agents]
        away_y = [agent.state.y for agent in self.away_agents]
        
        ax.scatter(home_x, home_y, c='blue', s=100, alpha=0.8, label='Home Team')
        ax.scatter(away_x, away_y, c='red', s=100, alpha=0.8, label='Away Team')
        
        # Plot ball
        ball_marker = 'o' if self.match_state.ball_carrier else 'X'
        ax.scatter([self.match_state.ball_x], [self.match_state.ball_y], 
                  c='yellow', s=150, marker=ball_marker, edgecolor='black', linewidth=2, label='Ball')
        
        # Highlight ball carrier
        if self.match_state.ball_carrier:
            all_agents = self.home_agents + self.away_agents
            ball_carrier = next((a for a in all_agents if a.state.player_id == self.match_state.ball_carrier), None)
            if ball_carrier:
                circle = patches.Circle((ball_carrier.state.x, ball_carrier.state.y), 3, 
                                      linewidth=3, edgecolor='yellow', facecolor='none')
                ax.add_patch(circle)
        
        # Set labels and title
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_xlabel('Field Length')
        ax.set_ylabel('Field Width')
        ax.set_title(f'Match State - Minute {self.match_state.minute} | Score: {self.match_state.score_home}-{self.match_state.score_away}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


# Example usage and integration functions
def simulate_match_with_predictions(home_team_attrs: Dict, away_team_attrs: Dict, 
                                  num_simulations: int = 100) -> Dict:
    """Run multiple simulations and return aggregated results."""
    results = []
    
    for _ in range(num_simulations):
        simulation = AgentBasedFootballSimulation(home_team_attrs, away_team_attrs)
        result = simulation.simulate_match()
        results.append(result)
    
    # Aggregate results
    home_wins = sum(1 for r in results if r['score_home'] > r['score_away'])
    draws = sum(1 for r in results if r['score_home'] == r['score_away'])
    away_wins = sum(1 for r in results if r['score_home'] < r['score_away'])
    
    avg_home_goals = np.mean([r['score_home'] for r in results])
    avg_away_goals = np.mean([r['score_away'] for r in results])
    avg_home_possession = np.mean([r['possession_home'] for r in results])
    
    return {
        'simulations': num_simulations,
        'home_win_prob': home_wins / num_simulations,
        'draw_prob': draws / num_simulations,
        'away_win_prob': away_wins / num_simulations,
        'avg_home_goals': avg_home_goals,
        'avg_away_goals': avg_away_goals,
        'avg_home_possession': avg_home_possession,
        'avg_away_possession': 1 - avg_home_possession,
        'detailed_results': results
    }