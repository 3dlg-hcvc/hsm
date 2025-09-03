import argparse
from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf

class HSMArgumentParser:
    """
    Argument parser for HSM.
    """

    def __init__(self, project_root: Path | str, description: str = "HSM: text to indoor scene generation"):
        """
        Initializes the argument parser.

        Args:
            project_root: The root directory of the project.
            description: The description of the parser.
        """
        self.parser = argparse.ArgumentParser(description=description)
        self.project_root = Path(project_root)
        self._add_common_arguments()

    def _add_common_arguments(self) -> None:
        """
        Adds arguments common to both single and batch processing.
        """
        self.parser.add_argument(
            "--desc", "--description", type=str, help="Room description", required=True
        )
        self.parser.add_argument(
            "--output", type=str, help="Output directory", default="results/batch"
        )
        self.parser.add_argument(
            "-t",
            "--types",
            nargs="+",
            help="Object types to process (large, wall, ceiling, small)",
            default=["large", "ceiling", "wall", "small"],
            choices=["large", "wall", "ceiling", "small"],
        )
        self.parser.add_argument(
            "-et",
            "--extra_types",
            nargs="+",
            help="Generate extra objects types (large, wall, ceiling, small)",
            default=["large", "ceiling", "wall", "small"],
            choices=["large", "wall", "ceiling", "small"],
        )
        self.parser.add_argument(
            "--skip-scene-motifs", action="store_true", help="Ablation: without scene motifs"
        )
        self.parser.add_argument("--skip-solver", action="store_true", help="Ablation: without DFS solver")
        self.parser.add_argument(
            "--skip-spatial-optimization",
            action="store_true",
            help="Ablation: without spatial optimization",
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