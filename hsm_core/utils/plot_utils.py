from matplotlib import pyplot as plt
import numpy as np
import io
from PIL import Image
from hsm_core.utils import get_logger

def get_scene_transforms(scene):
    """Get transforms for all objects in scene."""
    transforms = {}
    for name in scene.geometry.keys():
        try:
            transforms[name] = scene.graph.get(name)[0]
        except (ValueError, KeyError):
            try:
                path = scene.graph.transforms.shortest_path(scene.graph.base_frame, name)
                transform = np.eye(4)
                for i in range(len(path) - 1):
                    transform = np.dot(transform, 
                                    scene.graph.transforms.edge_data[(path[i], path[i + 1])])
                transforms[name] = transform
            except (ValueError, KeyError):
                logger = get_logger('hsm_core.utils.plot_utils')
                logger.warning(f"Could not get transform for {name}, using identity matrix")
                transforms[name] = np.eye(4)
    return transforms

def combine_figures(figures, num_cols=2, figsize=(10, 10), dpi=100, output_path=None):
    """Combine multiple matplotlib figures into a single figure.
    
    Args:
        figures (list): List of matplotlib figures to combine
        num_cols (int): Number of columns in the combined figure
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Dots per inch for the output figure
        output_path (str, optional): Path to save the combined figure
    
    Returns:
        matplotlib.figure.Figure: Combined figure
    """
    num_figs = len(figures)
    if not num_figs:
        return None
        
    num_rows = (num_figs + num_cols - 1) // num_cols  # Ceiling division
    
    # Create figure with explicit size and DPI
    combined_fig = plt.figure(figsize=figsize, dpi=dpi)
    
    for idx, fig in enumerate(figures):
        if fig is not None:
            # Create subplot
            ax = plt.subplot(num_rows, num_cols, idx + 1)
            
            # Render figure to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            
            # Load as image and display
            img = Image.open(buf)
            ax.imshow(img)
            ax.axis('off')  # Hide axes
            buf.close()
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    return combined_fig