import json
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
# LIB_PROGRAMS_DIR = PROJECT_ROOT / "data" / "motif_library" / "programs"
LIB_META_PROGRAMS_DIR = PROJECT_ROOT / "data" / "motif_library" / "meta_programs"

# os.makedirs(LIB_PROGRAMS_DIR, exist_ok=True)
LIB_META_PROGRAMS_DIR.mkdir(parents=True, exist_ok=True)

from ..programs.program import Program


def load(motif_type: str, program_id: int | None = None, is_meta: bool = False) -> list[Program]:
    '''
    Load a program from the program library.

    Args:
        motif_type: string, the motif type of the programs
        program_id: int, the ID of a specific program to load (if None, load all programs)
        is_meta: bool, whether to load a meta-program
    
    Returns:
        programs: list[Program], the loaded programs
    '''

    programs = []
    file_paths = [LIB_META_PROGRAMS_DIR / f"{motif_type}.json"]
    
    for file_path in file_paths:
        if file_path.exists():
            with open(file_path, "r") as file:
                program_json: dict = json.load(file)
                description: str = program_json["description"]
                code_string: str = program_json["code_string"]
                program = Program(code_string.split("\n"), description)
                programs.append(program)
    
    return programs

def length(motif_type: str, is_meta: bool = False) -> int:
    '''
    Return the number of programs in the program library.

    Args:
        motif_type: string, the motif type of the programs
        is_meta: bool, whether to return the number of meta-programs
    
    Returns:
        length: int, the number of programs in the program library
    '''
    return int((LIB_META_PROGRAMS_DIR / f"{motif_type}.json").exists())

if __name__ == "__main__":
    programs = load("row", is_meta=True)
    print(programs[0].definition_with_docstring())