import argparse
from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf

class HSMArgumentParser:
    """
    Argument parser for HSM (Hierarchical Scene Motifs).

    HSM generates realistic 3D indoor scenes from natural language descriptions
    using a hierarchical framework that organizes objects into meaningful motifs
    across multiple scales (large furniture, wall/ceiling objects, small items).
    """

    def __init__(self, project_root: Path | str, description: str = None):
        """
        Initializes the argument parser.

        Args:
            project_root: The root directory of the project.
            description: The description of the parser (uses default if None).
        """
        if description is None:
            description = (
                "HSM: Hierarchical Scene Motifs for Multi-Scale Indoor Scene Generation\n\n"
                "Generate realistic 3D indoor scenes from natural language descriptions.\n"
                "Supports hierarchical object organization, spatial optimization, and LLM-driven\n"
                "scene decomposition. Results include 3D models, visualizations, and scene state.\n\n"
                "Example: python main.py -d 'modern living room with sofa and coffee table' --output results/living_room"
            )
        self.parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self.project_root = Path(project_root)
        self._add_common_arguments()

    def _add_common_arguments(self) -> None:
        """
        Adds arguments common to both single and batch processing.
        """
        self.parser.add_argument(
            "-d",
            "--desc",
            "--description",
            type=str,
            help=(
                "Natural language description of the indoor scene to generate.\n"
                "Examples:\n"
                "  'small modern living room with L-shaped sofa and coffee table'\n"
                "  'cozy bedroom with king bed and nightstands'\n"
                "  'a large kitchen with island counter and dining area'"
            ),
            required=True
        )

        self.parser.add_argument(
            "--output",
            type=str,
            help=(
                "Directory where generated scene files will be saved.\n"
                "Includes 3D models (.glb), visualizations (.png), scene state (.json),\n"
                "and LLM session logs. Default: results/single_run"
            ),
            default="results/single_run"
        )

        self.parser.add_argument(
            "-t",
            "--types",
            nargs="+",
            help=(
                "Object types to generate and arrange in the scene.\n"
                "Available types:\n"
                "  large    - Main furniture (sofas, tables, beds, cabinets)\n"
                "  wall     - Wall-mounted objects (pictures, shelves, lights)\n"
                "  ceiling  - Ceiling fixtures (lights, fans)\n"
                "  small    - Small objects placed on surfaces (lamps, decor, appliances)\n"
                "Default: all types enabled"
            ),
            default=["large", "wall", "ceiling", "small"],
            choices=["large", "wall", "ceiling", "small"],
        )

        self.parser.add_argument(
            "-et",
            "--extra_types",
            nargs="+",
            help=(
                "Additional object types to generate beyond the main types.\n"
                "Used for generating extra small objects or specialized items.\n"
                "Same choices as --types. Default: all types enabled"
            ),
            default=["large", "wall", "ceiling", "small"],
            choices=["large", "wall", "ceiling", "small"],
        )

        self.parser.add_argument(
            "--skip-scene-motifs",
            action="store_true",
            help=(
                "Disable scene motifs (ablation).\n"
            )
        )

        self.parser.add_argument(
            "--skip-solver",
            action="store_true",
            help=(
                "Disable the spatial solver (ablation).\n"
            )
        )

        self.parser.add_argument(
            "--skip-spatial-optimization",
            action="store_true",
            help=(
                "Disable spatial optimization (ablation).\n"
            )
        )
        
    def parse_args(self):
        """
        Parses the command line arguments.

        Returns:
            The parsed arguments.
        """
        return self.parser.parse_args()

    def get_config(self, args) -> DictConfig|ListConfig:
        """
        Creates a configuration object from the parsed arguments.

        Args:
            args: The parsed arguments.

        Returns:
            The configuration object.
        """
        config_path = self.project_root / "configs" / "scene" / "scene_config.yaml"
        cfg = OmegaConf.load(config_path)

        # Override config with command line arguments
        cfg.room.room_description = args.desc
        cfg.execution.result_dir = args.output
        cfg.mode.use_scene_motifs = not args.skip_scene_motifs
        cfg.mode.use_solver = not args.skip_solver
        cfg.mode.enable_spatial_optimization = not args.skip_spatial_optimization
        cfg.mode.object_types = args.types
        cfg.mode.extra_types = args.extra_types
        
        return cfg