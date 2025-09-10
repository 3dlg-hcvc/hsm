from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from hsm_core.utils import get_logger

from .arrangement import Arrangement
from .obj import Obj

logger = get_logger('scene_motif.core.hierarchy')


@dataclass
class HierarchyNode:
    """Represents a node in the motif hierarchy."""
    motif_type: str
    description: str
    arrangement_call: str
    arrangement: Optional[Arrangement] = None
    parent: Optional['HierarchyNode'] = None
    children: List['HierarchyNode'] = field(default_factory=list)
    depth: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class MotifHierarchy:
    """
    Manages the hierarchical structure of scene motifs.
    """
    
    def __init__(self):
        self.root: Optional[HierarchyNode] = None
        self.nodes_by_type: Dict[str, List[HierarchyNode]] = {}
        self.execution_order: List[HierarchyNode] = []
        
    def build_hierarchy(self, arrangement_json_data: dict, arrangements: List[Tuple[str, str]]) -> HierarchyNode:
        """
        Build the hierarchy from arrangement JSON and sub-arrangements.
        
        Args:
            arrangement_json_data: Parsed JSON containing the main arrangement structure
            arrangements: List of (motif_type, arrangement_call) tuples for sub-arrangements
            
        Returns:
            The root node of the built hierarchy
        """
        # Create sub-arrangement nodes first (bottom-up)
        sub_nodes = []
        for motif_type, arrangement_call in arrangements:
            node = HierarchyNode(
                motif_type=motif_type,
                description=f"{motif_type} sub-arrangement",
                arrangement_call=arrangement_call,
                depth=1  # Sub-arrangements are at depth 1
            )
            sub_nodes.append(node)
            self._add_node_to_index(node)
        
        # Create root node
        self.root = HierarchyNode(
            motif_type=arrangement_json_data["type"],
            description=arrangement_json_data["description"],
            arrangement_call="",  # Will be set later
            depth=0
        )
        
        # Link sub-arrangements as children of root
        for sub_node in sub_nodes:
            self.add_child(self.root, sub_node)
        
        self._add_node_to_index(self.root)
        self._compute_execution_order()
        
        return self.root
    
    def add_child(self, parent: HierarchyNode, child: HierarchyNode) -> None:
        """Add a child node to a parent node."""
        child.parent = parent
        child.depth = parent.depth + 1
        parent.children.append(child)
        
    def get_nodes_at_depth(self, depth: int) -> List[HierarchyNode]:
        """Get all nodes at a specific depth level."""
        return [node for node in self.execution_order if node.depth == depth]
    
    def get_leaf_nodes(self) -> List[HierarchyNode]:
        """Get all leaf nodes (nodes with no children)."""
        return [node for node in self.execution_order if not node.children]
    
    def get_nodes_by_type(self, motif_type: str) -> List[HierarchyNode]:
        """Get all nodes of a specific motif type."""
        return self.nodes_by_type.get(motif_type, [])
    
    def traverse_bottom_up(self) -> List[HierarchyNode]:
        """Traverse the hierarchy from leaves to root (bottom-up)."""
        return list(reversed(self.execution_order))
    
    def traverse_top_down(self) -> List[HierarchyNode]:
        """Traverse the hierarchy from root to leaves (top-down)."""
        return self.execution_order
    
    def get_siblings(self, node: HierarchyNode) -> List[HierarchyNode]:
        """Get all sibling nodes of the given node."""
        if node.parent is None:
            return []
        return [child for child in node.parent.children if child != node]
    
    def get_subtree_nodes(self, node: HierarchyNode) -> List[HierarchyNode]:
        """Get all nodes in the subtree rooted at the given node."""
        result = [node]
        for child in node.children:
            result.extend(self.get_subtree_nodes(child))
        return result
    
    def set_arrangement(self, node: HierarchyNode, arrangement: Arrangement) -> None:
        """Set the arrangement for a node."""
        node.arrangement = arrangement
        
    def get_arrangement_context(self, node: HierarchyNode) -> List[Tuple[str, str, List[Obj]]]:
        """
        Get the arrangement context for a node (its children's arrangements).
        
        Returns:
            List of (motif_type, arrangement_call, objects) for each child
        """
        context = []
        for child in node.children:
            if child.arrangement:
                context.append((
                    child.motif_type,
                    child.arrangement_call,
                    child.arrangement.objs if child.arrangement else []
                ))
        return context
    
    def _add_node_to_index(self, node: HierarchyNode) -> None:
        """Add a node to the type-based index."""
        if node.motif_type not in self.nodes_by_type:
            self.nodes_by_type[node.motif_type] = []
        self.nodes_by_type[node.motif_type].append(node)
    
    def _compute_execution_order(self) -> None:
        """Compute the execution order (depth-first, children before parents)."""
        if not self.root:
            return
            
        self.execution_order = []
        self._dfs_execution_order(self.root)
    
    def _dfs_execution_order(self, node: HierarchyNode) -> None:
        """Depth-first traversal to compute execution order."""
        # Process children first (bottom-up)
        for child in node.children:
            self._dfs_execution_order(child)
        # Then process current node
        self.execution_order.append(node)
    
    def __repr__(self) -> str:
        if not self.root:
            return "MotifHierarchy(empty)"
        
        def _repr_node(node: HierarchyNode, indent: int = 0) -> str:
            prefix = "  " * indent
            result = f"{prefix}{node.motif_type} (depth={node.depth})\n"
            for child in node.children:
                result += _repr_node(child, indent + 1)
            return result
        
        return f"MotifHierarchy:\n{_repr_node(self.root)}" 