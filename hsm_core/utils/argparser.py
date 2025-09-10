# hsm_core/utils/config_parser.py
import argparse
from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf
from hsm_core.utils import get_logger

class ArgumentParser:
    """
    Argument parser for the HSM project.
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
            "-p", "--previous", action="store_true", help="Use previous result", default=False
        )
        self.parser.add_argument("-i", "--load_iter", type=int, help="Load iteration", default=0)
        self.parser.add_argument(
            "--prompt-path", type=str, help="Prompt CSV file", default="data/annotations.csv"
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
            "--skip-scene-motifs", action="store_true", help="Ablation: Skip scene motifs"
        )
        self.parser.add_argument("--skip-solver", action="store_true", help="Ablation: Skip solver")
        self.parser.add_argument(
            "--skip-spatial-optimization",
            action="store_true",
            help="Ablation: Skip spatial optimization",
        )
        self.parser.add_argument(
            "--export", action="store_true", help="(debugging) Only export scene", default=False
        )
        self.parser.add_argument(
            "-l",
            "--load-object-types",
            nargs="+",
            help="Object types to load (large, wall, ceiling, small)",
            default=None,
            choices=["large", "wall", "ceiling", "small"],
        )
        # Ray Serve arguments
        self.parser.add_argument(
            "--use-ray-serve",
            action="store_true",
            help="Use Ray Serve for CLIP model serving",
            default=True
        )
        self.parser.add_argument(
            "--ray-serve-url",
            type=str,
            help="Ray Serve server URL",
            default="http://localhost:8000"
        )
        self.parser.add_argument(
            "--ray-serve-timeout",
            type=float,
            help="Ray Serve request timeout in seconds",
            default=300.0
        )
        self.parser.add_argument(
            "--auto-start-ray-server",
            action="store_true",
            help="Automatically start Ray Serve server if not running",
            default=False
        )
        # LLM arguments
        self.parser.add_argument(
            "--qwen",
            action="store_true",
            help="Use Qwen model instead of GPT (quick switch)",
            default=False
        )

    def add_batch_arguments(self) -> None:
        """
        Adds arguments specific to batch processing.
        """
        self.parser.add_argument("-s", "--start", type=int, help="First index", default=None)
        self.parser.add_argument("-e", "--end", type=int, help="Last index (exclusive)", default=None)
        self.parser.add_argument("-list", "--list_ids", nargs='+', help="List of IDs to process", type=int, default=None)
        self.parser.add_argument(
            "--max-workers",
            type=int,
            help="Maximum number of parallel workers",
            default=2,
        )
        self.parser.add_argument(
            "--memory-per-process",
            type=int,
            help="Memory required per process in MiB",
            default=4096,
        )
        self.parser.add_argument(
            "--retry-count", type=int, help="Number of retries per prompt", default=1
        )
        self.parser.add_argument(
            "--multiprocessing", 
            action="store_true", 
            help="Enable multiprocessing and run in parallel", 
            default=True
        )
        self.parser.add_argument(
            "-se",
            "--skip-existing",
            action="store_true",
            help="Automatically skip existing prompts",
            default=False
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
        cfg.execution.load_iter = args.load_iter
        cfg.execution.export_only = args.export
        cfg.execution.use_previous_result = args.previous
        cfg.execution.result_dir = args.output
        cfg.mode.use_scene_motifs = not args.skip_scene_motifs
        cfg.mode.use_solver = not args.skip_solver
        cfg.mode.enable_spatial_optimization = not args.skip_spatial_optimization
        cfg.mode.object_types = args.types
        cfg.mode.load_object_types = args.load_object_types
        cfg.mode.extra_types = args.extra_types
        
        # Handle --qwen argument
        if args.qwen:
            # Create or update LLM configuration
            if not hasattr(cfg, 'llm'):
                cfg.llm = OmegaConf.create({})
            cfg.llm.model_type = "qwen"
            # Set default Qwen model if not already specified
            if not hasattr(cfg.llm, 'model_name'):
                cfg.llm.model_name = "Qwen2.5-VL-7B-Instruct"
            if not hasattr(cfg.llm, 'use_quantized'):
                cfg.llm.use_quantized = True
        
        # Config logging can be enabled if needed
        return cfg