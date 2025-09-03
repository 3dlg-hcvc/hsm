import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum
import time
import math
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from rtree import index

# Project imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from hsm_core.vlm.utils import round_nested_values

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

@dataclass
class MotifPlacement:
    """Represents a placed motif with all necessary information."""
    center_x: float
    center_y: float
    rotation: float
    bbox: List[Tuple[float, float]]
    score: float = 0.0


@dataclass
class DFSSolverConfig:
    """Configuration for the DFS solver."""
    grid_size: float = 0.1
    max_duration: float = 10.0
    max_candidates_per_motif: int = 10
    # wall_proximity_threshold: float = 0.1 # now use grid size
    alignment_threshold: float = -0.7
    epsilon: float = 1e-6
    
    # soft constraints
    initial_placement_range: float = 5.0 # the distance (in meters) over which initial placement score decays
    initial_placement_weight: float = 5.0 # weight for initial placement score
    wall_alignment_weight: float = 2.5  # weight for wall alignment score
    wall_alignment_range: float = 0.5   # the distance (in meters) over which wall score decays


class ConstraintType(Enum):
    """Types of constraints that can be applied to motifs."""
    EDGE_ALIGNMENT = "edge"
    IGNORE_COLLISION = "ignore_collision"
    ROTATION = "rotation"


class DFSSolver:
    """
    DFS-based support region optimization solver.
    """
    
    def __init__(self, config: Optional[DFSSolverConfig] = None):
        self.config = config or DFSSolverConfig()
        self.reset_state()
    
    def reset_state(self) -> None:
        """Reset solver state for a new problem."""
        self.surface_poly: Optional[Polygon] = None
        self.motifs: List[Tuple[str, Tuple[float, float, float], List[Dict]]] = []
        self.constraints: Dict[str, List[Dict]] = {}
        self.initial_placements: Dict[str, MotifPlacement] = {}
        self.original_names: Dict[str, str] = {}
        
        # Spatial indexing
        self.spatial_index: Optional[index.Index] = None
        self._idx_to_poly: Dict[int, Tuple[str, Polygon]] = {}
        self._motif_name_to_idx: Dict[str, int] = {}
        self._motif_idx_counter: int = 0
        
        # Search state
        self.solutions: List[Dict[str, MotifPlacement]] = []
        self.start_time: float = 0.0
        self._grid_cache: Dict[str, List[Tuple[float, float]]] = {}
    
    def solve(
        self, 
        surface_poly: Polygon, 
        motifs: List[Tuple[str, Tuple[float, float, float], List[Dict]]], 
        initial_placed: Optional[Dict[str, List]] = None, 
        verbose: bool = False
    ) -> List[Dict[str, MotifPlacement]]:
        """
        Solve the support region optimization problem.
        
        Args:
            surface_poly: The surface polygon
            motifs: List of (name, dimensions, constraints) tuples
            initial_placed: Initial placements as preferences
            verbose: Enable verbose logging
            
        Returns:
            List of solution dictionaries mapping motif names to placements
        """
        self.reset_state()
        self.surface_poly = surface_poly
        self.motifs = motifs
        self.constraints = {motif[0]: motif[2] for motif in motifs if motif[2]}
        
        print(f"\n=== Starting DFS Solver ===")
        print(f"Support Region: {surface_poly} with area: {surface_poly.area:.2f}")
        print(f"motifs to place: {[o[0] for o in motifs]}")
        print(motifs)
        print(f"Grid size: {self.config.grid_size}m")
        
        # Process initial placements
        self._process_initial_placements(initial_placed or {})
        
        # Initialize spatial index
        self.spatial_index = index.Index()
        
        # Separate fixed obstacles from movable motifs
        fixed_motifs, movable_motifs = self._separate_motifs()
        
        # Add obstacles to spatial index
        self._add_obstacles_to_index(fixed_motifs)
        
        # Sort motifs by area (largest first) for better search efficiency
        sorted_motifs = sorted(
            movable_motifs, 
            key=lambda x: x[1][0] * x[1][2], 
            reverse=True
        )
        
        # Start DFS search
        self.start_time = time.time()
        self._dfs_search(sorted_motifs, {}, 0, verbose)
        
        elapsed = time.time() - self.start_time
        print(f"\n=== Solver Finished ===")
        print(f"Solutions found: {len(self.solutions)}")
        print(f"Time elapsed: {elapsed:.2f}s")
        
        return self.solutions
    
    def _process_initial_placements(self, initial_placed: Dict[str, List]) -> None:
        """Process initial placements into motifPlacement motifs."""
        for motif_name, placement_data in initial_placed.items():
            if len(placement_data) >= 4:
                center = placement_data[0]
                rotation = placement_data[1]
                bbox = placement_data[2]
                score = placement_data[3] if len(placement_data) > 3 else 1.0
                
                self.initial_placements[motif_name] = MotifPlacement(
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    rotation=float(rotation),
                    bbox=bbox,
                    score=float(score)
                )
    
    def _separate_motifs(self) -> Tuple[List, List]:
        """Separate fixed obstacles from movable motifs."""
        fixed_motifs: List = []
        movable_motifs: List = []
        
        for motif in self.motifs:
            # Check if motif is explicitly marked as fixed
            # motifs with initial placements but not marked as fixed are still movable
            # (initial placement is just a preference/starting point)
            is_fixed = False
            for constraint in motif[2]:  # motif[2] contains constraints
                if constraint.get("constraint") == "is_fixed" or constraint.get("is_fixed"):
                    is_fixed = True
                    break
            
            if is_fixed:
                fixed_motifs.append(motif)
            else:
                movable_motifs.append(motif)
        
        return fixed_motifs, movable_motifs
    
    def _add_obstacles_to_index(self, obstacles: List) -> None:
        """Add fixed obstacles to the spatial index."""
        for obstacle_data in obstacles:
            motif_name = obstacle_data[0]
            if motif_name in self.initial_placements:
                placement = self.initial_placements[motif_name]
                poly = Polygon(placement.bbox)
                idx = self._motif_idx_counter
                
                if self.spatial_index is not None:
                    self.spatial_index.insert(idx, poly.bounds)
                
                self._idx_to_poly[idx] = (motif_name, poly)
                self._motif_name_to_idx[motif_name] = idx
                self._motif_idx_counter += 1
    
    def _dfs_search(
        self, 
        remaining_motifs: List, 
        current_placement: Dict[str, MotifPlacement], 
        depth: int, 
        verbose: bool
    ) -> None:
        """
        DFS search with backtracking.
        
        Args:
            remaining_motifs: motifs still to be placed
            current_placement: Currently placed motifs
            depth: Search tree depth (number of motifs placed so far)
            verbose: Enable verbose logging
        """
        # Check timeout
        if time.time() - self.start_time > self.config.max_duration:
            if verbose:
                print(f"Timeout at depth {depth}")
            # Save partial solution if it's the best we have
            if depth > 0 and not self.solutions:
                self.solutions.append(dict(current_placement))
            return
        
        # Base case: all motifs placed
        if not remaining_motifs:
            if verbose:
                print(f"Found complete solution at depth {depth}")
            self.solutions.append(dict(current_placement))
            return
        
        # Get next motif to place
        current_motif = remaining_motifs[0]
        motif_name, motif_dims, motif_constraints = current_motif
        
        if verbose:
            print(f"Search depth {depth}: Placing {motif_name}, motif_dims: {motif_dims}, motif_constraints: {motif_constraints}")
        
        # Generate and evaluate candidates using this motif's specific dimensions
        candidates = self._generate_candidates(motif_name, motif_dims, current_placement)
        valid_candidates = self._filter_and_score_candidates(
            candidates, motif_name, motif_dims, motif_constraints, current_placement, verbose
        )
        
        if not valid_candidates:
            if verbose:
                print(f"No valid positions for {motif_name}, trying next motif.")
            # skip current motif and recurse on the rest.
            self._dfs_search(remaining_motifs[1:], current_placement, depth, verbose)
            return
        
        # Track how many candidates we actually try
        candidates_tried = 0
        max_candidates = min(len(valid_candidates), self.config.max_candidates_per_motif)
        
        # Try top candidates
        for i, (placement, score) in enumerate(valid_candidates[:self.config.max_candidates_per_motif]):
            candidates_tried += 1
            if verbose:
                print(f"  Trying candidate {i+1}/{max_candidates} (score: {score:.3f})")
            
            # Place motif
            current_placement[motif_name] = placement
            self._add_to_spatial_index(motif_name, placement)
            
            # Recursion
            self._dfs_search(remaining_motifs[1:], current_placement, depth + 1, verbose)
            
            # Backtrack
            self._remove_from_spatial_index(motif_name)
            del current_placement[motif_name]
            
            # Early termination if we found a solution
            if self.solutions:
                if verbose:
                    print(f"  Solution found after trying {candidates_tried}/{max_candidates} candidates for {motif_name}")
                return
    
    def _generate_candidates(
        self, 
        motif_name: str, 
        motif_dims: Tuple[float, float, float], 
        current_placement: Dict[str, MotifPlacement]
    ) -> List[MotifPlacement]:
        """Generate candidate placements for an motif using its specific dimensions."""
        candidates = []
        w, h, d = motif_dims
        
        # Check for initial placement preference
        if motif_name in self.initial_placements:
            initial = self.initial_placements[motif_name]
            if self._is_placement_valid(initial, current_placement, ignore_collision=False):
                candidates.append(initial)
        
        # Generate grid-based candidates
        grid_points = self._get_grid_points()
        rotations = self._get_rotation_candidates(motif_name)
        
        for x, y in grid_points:
            for rotation in rotations:
                # Create bbox using this motif's specific width (w) and depth (d)
                bbox = self._create_rotated_bbox(x, y, w, d, rotation)
                placement = MotifPlacement(
                    center_x=x,
                    center_y=y,
                    rotation=rotation,
                    bbox=bbox
                )
                candidates.append(placement)
        
        return candidates
    
    def _get_grid_points(self) -> List[Tuple[float, float]]:
        """Get grid points within the surface polygon."""
        if not self.surface_poly:
            return []
        
        # Use cached grid if available
        cache_key = f"{self.surface_poly.bounds}_{self.config.grid_size}"
        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]
        
        min_x, min_y, max_x, max_y = self.surface_poly.bounds
        
        # Handle degenerate cases
        width = max_x - min_x
        height = max_y - min_y
        
        if width < self.config.epsilon or height < self.config.epsilon:
            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            if self.surface_poly.contains(Point(center_x, center_y)):
                grid_points = [(center_x, center_y)]
            else:
                grid_points = []
        else:
            # Generate efficient grid
            x_coords = np.arange(min_x, max_x, self.config.grid_size)
            y_coords = np.arange(min_y, max_y, self.config.grid_size)
            
            # Vectorized containment check
            grid_points = []
            for x in x_coords:
                for y in y_coords:
                    if self.surface_poly.contains(Point(x, y)):
                        grid_points.append((x, y))
        
        # Cache the result
        self._grid_cache[cache_key] = grid_points
        return grid_points
    
    def _get_rotation_candidates(self, motif_name: str) -> List[float]:
        """Get rotation candidates for an motif based on constraints."""
        constraints = self.constraints.get(motif_name, [])
        
        # Check for specific rotation constraint
        for constraint in constraints:
            if constraint.get("constraint") == ConstraintType.ROTATION.value:
                angle = constraint.get("angle")
                if angle is not None:
                    return [float(angle)]
        
        # Default rotations
        return [0.0, 90.0, 180.0, 270.0]
    
    def _filter_and_score_candidates(
        self,
        candidates: List[MotifPlacement],
        motif_name: str,
        motif_dims: Tuple[float, float, float],
        constraints: List[Dict],
        current_placement: Dict[str, MotifPlacement],
        verbose: bool
    ) -> List[Tuple[MotifPlacement, float]]:
        """Filter candidates by hard constraints and score by soft constraints."""
        valid_candidates = []
        
        # Check ignore collision constraint
        ignore_collision = any(
            c.get("constraint") == ConstraintType.IGNORE_COLLISION.value
            for c in constraints
        )
        
        for candidate in candidates:
            # Hard constraints
            if not self._check_hard_constraints(candidate, motif_dims, constraints, current_placement, ignore_collision):
                continue
            
            # Soft constraints (scoring)
            score = self._calculate_soft_score(candidate, motif_name, constraints, current_placement)
            candidate.score = score
            valid_candidates.append((candidate, score))
        
        # Sort by score (descending)
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if verbose and valid_candidates:
            print(f"    Top candidate scores: {[f'{s:.3f}' for _, s in valid_candidates[:3]]}")
        
        return valid_candidates
    
    def _check_hard_constraints(
        self,
        placement: MotifPlacement,
        motif_dims: Tuple[float, float, float],
        constraints: List[Dict],
        current_placement: Dict[str, MotifPlacement],
        ignore_collision: bool
    ) -> bool:
        """Check if placement satisfies all hard constraints."""
        # Surface containment
        if not self._check_surface_containment(placement):
            return False
        
        # Collision check (unless ignored)
        if not ignore_collision and self._check_collision(placement, current_placement):
            return False
        
        # Edge alignment constraint
        for constraint in constraints:
            if constraint.get("constraint") == ConstraintType.EDGE_ALIGNMENT.value:
                if not self._check_edge_alignment(placement, motif_dims, constraint):
                    return False
        
        return True
    
    def _check_surface_containment(self, placement: MotifPlacement) -> bool:
        """Check if motif is contained within the surface."""
        if not self.surface_poly:
            return False
        
        try:
            motif_poly = Polygon(placement.bbox)
            
            # Clean geometries to avoid topology issues
            surface_clean = self._clean_polygon(self.surface_poly)
            motif_clean = self._clean_polygon(motif_poly)
            
            # Strict containment check first
            if surface_clean.contains(motif_clean):
                return True
            
            # Tolerance check for numerical precision
            if not surface_clean.intersects(motif_clean):
                return False
            
            difference = motif_clean.difference(surface_clean)
            return difference.area < self.config.epsilon
            
        except Exception:
            # Fallback to bounds check
            surface_bounds = self.surface_poly.bounds
            motif_bounds = Polygon(placement.bbox).bounds
            return (
                motif_bounds[0] >= surface_bounds[0] - self.config.epsilon and
                motif_bounds[1] >= surface_bounds[1] - self.config.epsilon and
                motif_bounds[2] <= surface_bounds[2] + self.config.epsilon and
                motif_bounds[3] <= surface_bounds[3] + self.config.epsilon
            )
    
    def _clean_polygon(self, polygon: Polygon) -> Polygon:
        """Clean polygon to fix topology issues."""
        if polygon.is_valid and not polygon.is_empty:
            return polygon
        
        try:
            cleaned = polygon.buffer(0)
            if cleaned.is_valid and not cleaned.is_empty:
                return cleaned
            else:
                cleaned = polygon.buffer(self.config.epsilon)
                return cleaned if cleaned.is_valid else polygon
        except Exception:
            return polygon
    
    def _check_collision(
        self, 
        placement: MotifPlacement, 
        current_placement: Dict[str, MotifPlacement]
    ) -> bool:
        """Check if placement collides with existing motifs."""
        motif_poly = Polygon(placement.bbox)
        
        # Check against spatial index (fixed obstacles)
        if self.spatial_index:
            for idx in self.spatial_index.intersection(motif_poly.bounds):
                _, other_poly = self._idx_to_poly[idx]
                if motif_poly.intersects(other_poly):
                    return True
        
        # Check against currently placed motifs
        for other_placement in current_placement.values():
            other_poly = Polygon(other_placement.bbox)
            if motif_poly.intersects(other_poly):
                return True
        
        return False
    
    def _check_edge_alignment(self, placement: MotifPlacement, motif_dims: Tuple[float, float, float], constraint: Dict) -> bool:
        """Check if motif is properly aligned to a wall edge."""
        if not self.surface_poly:
            return False
        
        # get the walls to check
        walls_to_check = self._get_target_walls(constraint)
        if not walls_to_check:
            return False
        
        # Check alignment with any of the target walls
        for wall in walls_to_check:
            if self._check_wall_alignment(placement, motif_dims, wall):
                return True
        
        return False
    
    def _check_wall_alignment(self, placement: MotifPlacement, motif_dims: Tuple[float, float, float], wall: LineString) -> bool:
        """Check if placement is properly aligned to a specific wall."""
        # Check orientation (front faces into surface)
        if not self._check_front_orientation(placement, wall):
            return False
        
        # Check back edge proximity
        return self._check_back_edge_proximity(placement, motif_dims, wall)
    
    def _check_front_orientation(self, placement: MotifPlacement, wall: LineString) -> bool:
        """Check if motif's front faces into the surface relative to the wall."""
        wall_coords = list(wall.coords)
        wall_vec = np.array(wall_coords[1]) - np.array(wall_coords[0])
        
        # Calculate outward normal (pointing into surface)
        outward_normal = np.array([wall_vec[1], -wall_vec[0]])
        norm = np.linalg.norm(outward_normal)
        if norm == 0:
            return False
        outward_normal /= norm
        
        # Calculate motif's front direction
        front_angle_rad = math.radians(placement.rotation + 270.0)
        front_vector = np.array([math.cos(front_angle_rad), math.sin(front_angle_rad)])
        
        # Check if front faces into surface
        dot_product = np.dot(front_vector, outward_normal)
        return dot_product < self.config.alignment_threshold
    
    def _check_back_edge_proximity(self, placement: MotifPlacement, motif_dims: Tuple[float, float, float], wall: LineString) -> bool:
        """Check if motif's back edge is close enough to the wall."""
        if not motif_dims:
            return False
        
        original_width, _, original_depth = motif_dims

        # Calculate the back and edge vectors
        back_angle_rad = math.radians(placement.rotation + 90.0)
        back_vector = np.array([math.cos(back_angle_rad), math.sin(back_angle_rad)])
        along_edge_angle_rad = math.radians(placement.rotation)
        edge_vector = np.array([math.cos(along_edge_angle_rad), math.sin(along_edge_angle_rad)])

        # Find the corners of the back edge
        center_pos = np.array([placement.center_x, placement.center_y])
        back_edge_center = center_pos + back_vector * (original_depth / 2.0)
        
        # Calculate the distance to the wall
        half_width_vec = edge_vector * (original_width / 2.0)
        corner1 = back_edge_center + half_width_vec
        corner2 = back_edge_center - half_width_vec
        dist1 = Point(corner1).distance(wall)
        dist2 = Point(corner2).distance(wall)

        # Valid placement if the closest corner is within half grid size
        return bool(min(dist1, dist2) <= self.config.grid_size)
    
    def _calculate_soft_score(
        self,
        placement: MotifPlacement,
        motif_name: str,
        constraints: List[Dict],
        current_placement: Dict[str, MotifPlacement]
    ) -> float:
        """Calculate soft constraint score for placement."""
        score: float = 0.0
        
        # Initial position preference (high weight)
        if motif_name in self.initial_placements:
            initial = self.initial_placements[motif_name]
            distance = math.hypot(
                placement.center_x - initial.center_x,
                placement.center_y - initial.center_y
            )
            # Normalize and invert distance (closer = higher score)
            init_score = max(0.0, 1.0 - distance / self.config.initial_placement_range)
            score += self.config.initial_placement_weight * init_score
        
        # Wall alignment preference
        edge_constraint = next((c for c in constraints if c.get("constraint") == ConstraintType.EDGE_ALIGNMENT.value), None)

        if edge_constraint and self.surface_poly:
            # Use the new helper to get the specific target wall(s)
            target_walls = self._get_target_walls(edge_constraint)
            
            if target_walls:
                # Get the normalized score [0, 1] based on distance to the correct wall(s)
                normalized_wall_score = self._calculate_wall_alignment_score(placement, target_walls)
                score += self.config.wall_alignment_weight * normalized_wall_score
        
        return score
    
    def _calculate_wall_alignment_score(
        self, placement: MotifPlacement, target_walls: List[LineString]
    ) -> float:
        """
        Calculate a NORMALIZED score based on proximity to a list of target walls.
        The score is 1.0 if flush with a wall, decaying to 0.0 at `wall_alignment_range`.
        """
        if not target_walls:
            return 0.0

        center = Point(placement.center_x, placement.center_y)
        
        # Find the distance to the closest of the target walls
        min_distance = min(wall.distance(center) for wall in target_walls)

        # Use linear decay for a normalized score [0, 1]
        if self.config.wall_alignment_range > self.config.epsilon:
            normalized_score = max(0.0, 1.0 - (min_distance / self.config.wall_alignment_range))
            return normalized_score

        return 0.0
    
    def _get_wall_segments(self) -> List[LineString]:
        """Get wall segments from surface polygon."""
        if not self.surface_poly:
            return []
        
        coords = list(self.surface_poly.exterior.coords)
        return [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]
    
    def _get_target_walls(self, constraint: Dict) -> List[LineString]:
        """Gets specific or all wall segments based on a constraint dictionary."""
        all_walls = self._get_wall_segments()
        if not all_walls:
            return []
            
        target_wall_id = constraint.get("wall_alignment_id")
        
        if target_wall_id is not None and 0 <= target_wall_id < len(all_walls):
            # A specific wall is targeted, return it as a single-item list
            return [all_walls[target_wall_id]]
        else:
            # No specific wall, so all walls are potential targets
            return all_walls
        
    def _add_to_spatial_index(self, motif_name: str, placement: MotifPlacement) -> None:
        """Add motif to spatial index."""
        if not self.spatial_index:
            return
        
        poly = Polygon(placement.bbox)
        idx = self._motif_idx_counter
        self.spatial_index.insert(idx, poly.bounds)
        self._idx_to_poly[idx] = (motif_name, poly)
        self._motif_name_to_idx[motif_name] = idx
        self._motif_idx_counter += 1
    
    def _remove_from_spatial_index(self, motif_name: str) -> None:
        """Remove motif from spatial index."""
        if not self.spatial_index or motif_name not in self._motif_name_to_idx:
            return
        
        idx = self._motif_name_to_idx.pop(motif_name)
        poly_info = self._idx_to_poly.pop(idx, None)
        if poly_info:
            _, poly = poly_info
            self.spatial_index.delete(idx, poly.bounds)
    
    def _create_rotated_bbox(
        self, 
        cx: float, 
        cy: float, 
        width: float, 
        depth: float, 
        angle_deg: float
    ) -> List[Tuple[float, float]]:
        """Create rotated bounding box."""
        angle_rad = math.radians(angle_deg)
        hw, hd = width / 2, depth / 2
        
        # Corner points (not rotated)
        corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
        
        # Rotate and translate
        rotated_bbox = []
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        for x, y in corners:
            rotated_x = x * cos_a - y * sin_a + cx
            rotated_y = x * sin_a + y * cos_a + cy
            rotated_bbox.append((rotated_x, rotated_y))
        
        return rotated_bbox
    
    def _is_placement_valid(
        self, 
        placement: MotifPlacement, 
        current_placement: Dict[str, MotifPlacement], 
        ignore_collision: bool = False
    ) -> bool:
        """Check if a placement is valid."""
        if not self._check_surface_containment(placement):
            return False
        
        if not ignore_collision and self._check_collision(placement, current_placement):
            return False
        
        return True
    
    def _extract_dimension(self, dim_value: Union[float, List, Tuple], default: float = 1.0) -> float:
        """Safely extract numeric dimension value."""
        current_val = dim_value
        
        # Unwrap nested sequences
        while isinstance(current_val, (list, tuple, np.ndarray)):
            if len(current_val) == 0:
                return default
            elif len(current_val) == 1:
                current_val = current_val[0]
            else:
                current_val = current_val[0]  # Take first element
                break
        
        try:
            return float(current_val)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Could not convert dimension to float: {dim_value}. Error: {e}")
    
    def format_input(
        self, 
        input_motifs: List[Dict], 
        expand_extent: float = 1.0
    ) -> Tuple[List, Dict, Dict]:
        """Convert input motifs to solver format."""
        motifs_list = []
        constraints = {}
        initial_placements = {}
        name_counts = defaultdict(int)
        
        for motif in input_motifs:
            # Create unique identifier
            original_name = motif.get("id", motif.get("name"))
            if original_name in name_counts:
                name_counts[original_name] += 1
                unique_id = f"{original_name}_{name_counts[original_name]}"
            else:
                unique_id = original_name
            
            if original_name is not None and unique_id is not None:
                self.original_names[unique_id] = original_name
            
            # Process constraints
            motif_constraints = []
            if motif.get("wall_alignment") is True:
                edge_constraint = {"type": "global", "constraint": "edge"}
                if "wall_alignment_id" in motif:
                    edge_constraint["wall_alignment_id"] = motif["wall_alignment_id"]
                motif_constraints.append(edge_constraint)
            
            if motif.get("ignore_collision") is True:
                motif_constraints.append({"type": "global", "constraint": "ignore_collision"})
            
            if "rotation" in motif and motif["rotation"] % 90 != 0:
                motif_constraints.append({
                    "type": "global", 
                    "constraint": "rotation", 
                    "angle": float(motif["rotation"])
                })
            
            if motif.get("is_fixed", False) or motif.get("fixed", False):
                motif_constraints.append({"constraint": "is_fixed"})
            
            motif_constraints.extend(motif.get("constraints", []))
            
            # Process dimensions
            if len(motif["dimensions"]) >= 3:
                dimensions = (
                    self._extract_dimension(motif["dimensions"][0]),
                    self._extract_dimension(motif["dimensions"][1]),
                    self._extract_dimension(motif["dimensions"][2])
                )
            elif len(motif["dimensions"]) == 2:
                dimensions = (
                    self._extract_dimension(motif["dimensions"][0]),
                    self._extract_dimension(motif["dimensions"][1]),
                    self._extract_dimension(motif["dimensions"][1])
                )
            else:
                dim_val = self._extract_dimension(motif["dimensions"][0])
                dimensions = (dim_val, dim_val, dim_val)
            
            motifs_list.append((unique_id, dimensions, motif_constraints))
            constraints[unique_id] = motif_constraints
            
            # Handle existing placements
            if "position" in motif:
                x, z = motif["position"]
                rotation = motif.get("rotation", 0) % 360
                
                if "wall_id" in motif:
                    width = self._extract_dimension(motif["dimensions"][0])
                    depth = max(
                        self._extract_dimension(motif["dimensions"][1]),
                        self._extract_dimension(motif["dimensions"][2]) if len(motif["dimensions"]) > 2 else 0
                    )
                    bbox = self._create_rotated_bbox(x, z, width, depth, rotation)
                else:
                    if len(motif["dimensions"]) > 2:
                        dim0 = self._extract_dimension(motif["dimensions"][0])
                        dim2 = self._extract_dimension(motif["dimensions"][2])
                        bbox = self._create_rotated_bbox(x, z, dim0 * expand_extent, dim2 * expand_extent, rotation)
                    else:
                        dim0 = self._extract_dimension(motif["dimensions"][0])
                        dim1 = self._extract_dimension(motif["dimensions"][1])
                        bbox = self._create_rotated_bbox(x, z, dim0 * expand_extent, dim1 * expand_extent, rotation)
                
                initial_placements[unique_id] = [(float(x), float(z)), float(rotation), bbox, 1.0]
        
        return motifs_list, constraints, initial_placements
    
    def format_solution(
        self, 
        initial_placements: Dict, 
        solutions: List[Dict], 
        fallback: bool = False
    ) -> List[Dict]:
        """Convert solutions to output format."""
        if not solutions and fallback and initial_placements:
            # Use initial placements as fallback
            fallback_solution = {}
            for motif_name, placement in initial_placements.items():
                if motif_name in {motif[0] for motif in self.motifs}:
                    fallback_solution[motif_name] = placement
            return self.format_solution(initial_placements, [fallback_solution])
        elif not solutions:
            return []
        
        best_solution = solutions[0]
        formatted_motifs = []
        
        for motif_name, placement in best_solution.items():
            if motif_name not in {motif[0] for motif in self.motifs}:
                continue
            
            # Handle different placement formats
            if isinstance(placement, MotifPlacement):
                center_x = placement.center_x
                center_z = placement.center_y
                rotation = placement.rotation
                bbox = placement.bbox
            elif isinstance(placement, dict):
                center_x = float(placement["center_x"])
                center_z = float(placement["center_y"])
                rotation = float(placement["rotation"])
                bbox = placement["bbox"]
            else:
                # Legacy format
                if isinstance(placement[0], (tuple, list)):
                    center_x = float(placement[0][0])
                    center_z = float(placement[0][1])
                    rotation = float(placement[1] % 360)
                    bbox = placement[2]
                else:
                    center_x = float(placement[0])
                    center_z = float(placement[1])
                    rotation = float(placement[3] % 360)
                    bbox = placement[2]
            
            # use dimension from initial placements
            motif_data = next((o for o in self.motifs if o[0] == motif_name), None)
            width = motif_data[1][0] if motif_data else 0.0
            depth = motif_data[1][2] if motif_data else 0.0
            
            original_name = self.original_names.get(motif_name, motif_name)
            # print(initial_placements)
            formatted_motif = {
                "id": original_name,
                "position": [center_x, center_z],
                "rotation": rotation,
                "dimensions": [width, depth],
                "bbox": [list(point) for point in bbox],
            }
            
            # Check for ignore_collision flag
            for motif in self.motifs:
                if motif[0] == motif_name:
                    for constraint in motif[2]:
                        if constraint.get("constraint") == "ignore_collision":
                            formatted_motif["ignore_collision"] = True
                            break
            
            formatted_motifs.append(formatted_motif)
        
        return formatted_motifs
    
    def calculate_occupancy(self, solution: Dict) -> float:
        """Calculate space occupancy ratio."""
        if not self.surface_poly:
            return 0.0
        
        total_area = 0.0
        for key, placement in solution.items():
            if key not in {motif[0] for motif in self.motifs}:
                continue
            
            try:
                if isinstance(placement, MotifPlacement):
                    bbox = placement.bbox
                elif isinstance(placement, dict):
                    bbox = placement["bbox"]
                elif isinstance(placement, (list, tuple)) and len(placement) >= 3:
                    bbox = placement[2]
                else:
                    continue
                
                poly = Polygon(bbox)
                total_area += poly.area
            except Exception as e:
                print(f"Error calculating area for {key}: {e}")
                continue
        
        surface_area = self.surface_poly.area
        return total_area / surface_area if surface_area > 0 else 0.0
    
    def visualize_solution(
        self,
        solution: Dict,
        initial_placements: Dict,
        input_motifs: Optional[List[Dict]] = None,
        output_path: Optional[str] = None
    ) -> None:
        """Visualize the solution."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw surface
        if self.surface_poly:
            surface_x, surface_y = self.surface_poly.exterior.xy
            ax.plot(surface_x, surface_y, "k-", linewidth=2, label="Surface")
        
        # Draw solution motifs
        for motif_name, placement in solution.items():
            if motif_name == "door_clearance":
                continue
            
            try:
                if isinstance(placement, MotifPlacement):
                    center = (placement.center_x, placement.center_y)
                    bbox = placement.bbox
                    rotation = placement.rotation
                elif isinstance(placement, dict):
                    center = placement["world_position"]
                    bbox = placement["bbox"]
                    rotation = placement["rotation"]
                elif isinstance(placement, (list, tuple)):
                    if len(placement) == 4 and isinstance(placement[0], (list, tuple)):
                        center = placement[0]
                        rotation = placement[1]
                        bbox = placement[2]
                    else:
                        center = (placement[0], placement[1])
                        bbox = placement[2]
                        rotation = placement[3]
                else:
                    continue
                
                # Draw motif polygon
                poly = Polygon(bbox)
                x, y = poly.exterior.xy
                artist = ax.fill(x, y, alpha=0.5, label=motif_name, edgecolor="black", linewidth=1)
                
                # Draw orientation arrow
                center_x, center_y = float(center[0]), float(center[1])
                bbox_width = max(abs(bbox[1][0] - bbox[0][0]), abs(bbox[2][0] - bbox[3][0]))
                bbox_height = max(abs(bbox[1][1] - bbox[0][1]), abs(bbox[2][1] - bbox[3][1]))
                arrow_length = max(bbox_width, bbox_height) * 0.5
                
                # Convert world rotation (0°=South) to plot rotation (0°=East) - consistent with custom.py
                plot_rotation = (rotation + 270.0) % 360
                angle_rad = math.radians(plot_rotation)
                dx = arrow_length * math.cos(angle_rad)
                dy = arrow_length * math.sin(angle_rad)
                
                head_width = arrow_length * 0.2
                head_length = arrow_length * 0.2
                
                ax.arrow(center_x, center_y, dx, dy,
                        head_width=head_width,
                        head_length=head_length,
                        fc="red", ec="red")
                
                # Add label
                ax.text(center_x, center_y, motif_name, fontsize=8, ha="center", va="center")
                
            except Exception as e:
                print(f"Error drawing motif {motif_name}: {e}")
                continue
        
        ax.set_title("DFS Solver Solution")
        ax.set_aspect("equal")
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            print(f"Solution saved to {output_path}")
        
        plt.close()

def _handle_ignore_collision(
    solver: DFSSolver,
    ignore_collision_motifs: List[Dict],
    initial_placements: Dict,
    raw_solutions: List[Dict],
    fallback: bool = False
) -> List[Dict]:
    """Handle ignore collision motifs."""
    # Create or get the best solution
    if raw_solutions:
        best_solution = dict(raw_solutions[0])  # Create a copy
    else:
        # Create new solution if no regular motifs were solved
        best_solution = {}
    
    for motif in ignore_collision_motifs:
        motif_name = motif.get("id", motif.get("name"))
        if motif_name is None:
            continue  # Skip motifs without valid names
            
        dimensions = tuple(motif["dimensions"])
        motif_constraints = [{"type": "global", "constraint": "ignore_collision"}]
        
        # Add to solver's motif list
        solver.motifs.append((motif_name, dimensions, motif_constraints))
        solver.constraints[motif_name] = motif_constraints
        
        # Add to solution if positioned
        if "position" in motif and "rotation" in motif:
            x, z = motif["position"]
            rotation = motif["rotation"] % 360
            bbox = solver._create_rotated_bbox(x, z, motif["dimensions"][0], motif["dimensions"][2], rotation)
            
            placement = MotifPlacement(
                center_x=float(x),
                center_y=float(z),
                rotation=float(rotation),
                bbox=bbox
            )
            best_solution[motif_name] = placement
            
            # Also add to initial_placements for format_solution
            initial_placements[motif_name] = [
                (float(x), float(z)),
                float(rotation),
                bbox,
                1.0
            ]
    
    # Return the modified solutions list
    if raw_solutions:
        # Replace the first solution with our modified version
        return [best_solution] + raw_solutions[1:]
    else:
        # Return new solution list
        return [best_solution] if best_solution else []

def run_solver(
    surface_motifs: List[Dict],
    surface_geometry: Polygon,
    grid_size: float = 0.1,
    expand_extent: float = 1.00,
    output_dir: str = "",
    subfix: str = "",
    fallback: bool = True,
    verbose: bool = False,
    enable: bool = True,
) -> Tuple[List[Dict], float]:
    """
    Run the DFS solver on surface motifs.
    
    Args:
        surface_motifs: List of motif dictionaries
        surface_geometry: Surface polygon
        grid_size: Grid resolution in meters
        expand_extent: motif expansion factor
        output_dir: Output directory for visualization
        subfix: Filename suffix
        fallback: Use initial positions if no solution found
        verbose: Enable verbose logging
        enable: Enable solver (if False, use initial positions)
        
    Returns:
        Tuple of (placed_motifs, occupancy_ratio)
    """
    # Separate motifs by collision handling
    motifs_to_solve = []
    ignore_collision_motifs = []
    
    for motif in surface_motifs:
        if motif.get("ignore_collision"):
            ignore_collision_motifs.append(motif)
        else:
            motifs_to_solve.append(motif)
    
    if enable:
        config = DFSSolverConfig(grid_size=grid_size)
        solver = DFSSolver(config)
        
        if verbose:
            # print(f"motifs to solve: {len(motifs_to_solve)}")
            print(f"motifs to solve: {len(motifs_to_solve)}: {round_nested_values(surface_motifs)}")
            print(f"motifs with ignore_collision: {len(ignore_collision_motifs)}")
        
        if not motifs_to_solve and not ignore_collision_motifs:
            print("No motifs to solve")
            return [], 0.0
        
        # Process solvable motifs
        raw_solutions = []
        if motifs_to_solve:
            motifs_list, constraints, initial_placements = solver.format_input(motifs_to_solve, expand_extent)
            raw_solutions = solver.solve(surface_geometry, motifs_list, initial_placements, verbose)
            print(f"solver raw solutions: {raw_solutions}")
        else:
            # No regular motifs to solve, create empty solution
            initial_placements = {}
        
        # Handle ignore_collision motifs
        if ignore_collision_motifs:
            raw_solutions = _handle_ignore_collision(solver, ignore_collision_motifs, initial_placements, raw_solutions, fallback)
        
        # Format solution
        placed_motifs = solver.format_solution(initial_placements, raw_solutions, fallback)
        
        occupancy = 0.0
        if raw_solutions:
            best_raw_solution = raw_solutions[0]
            occupancy = solver.calculate_occupancy(best_raw_solution)
            print(f"Occupancy: {occupancy:.2%}")
            
            # Visualize solution
            output_path = os.path.join(output_dir, f"solution_{subfix}.png") if output_dir else None
            solver.visualize_solution(best_raw_solution, initial_placements, output_path=output_path)
    
    else:
        # Ablation mode: use initial positions
        if verbose:
            print("Ablation: skipping DFS solver...")
        
        placed_motifs = []
        for motif in surface_motifs:
            if motif.get("is_fixed") is True:
                continue
            
            position = (
                [motif["position"][0], motif["position"][2]]
                if len(motif["position"]) == 3
                else [motif["position"][0], motif["position"][1]]
            )
            
            placed_motifs.append({
                "id": motif["id"],
                "position": position,
                "rotation": motif["rotation"],
                "dimensions": [motif["dimensions"][0], motif["dimensions"][2]],
            })
        
        occupancy = 0.0
    
    print(f"placed_motifs: {round_nested_values(placed_motifs)}")
    
    return placed_motifs, occupancy