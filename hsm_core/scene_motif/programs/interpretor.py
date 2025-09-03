from .program import Program
from .language import *
from ..utils import resolve_sub_arrangements

def execute(program: Program, variables: dict = {}) -> dict:
    """
    Execute the program and return the local variables.

    Args:
        program: Program, the program to execute
        variables: dict, optional initial variables to pass to the program
    
    Returns:
        locals: dict, the local variables of the executed program
    """
    locals = variables.copy() if variables else {}
    exec(program.code_string, globals(), locals)
    return locals

def execute_with_context(
    meta_program: Program,
    call_string: str | None,
    execute_results: list,
    extra_globals: dict | None = None
):
    """Execute a meta-program function call with sub-arrangement context.

    Args:
        meta_program: Program to execute (meta-program definition)
        call_string: The function call string (possibly referencing sub_arrangements[i])
        execute_results: Previously executed sub-arrangements (used for reference resolution)
        extra_globals: Optional extra globals to expose during execution

    Returns:
        tuple[dict, str]: (locals_dict returned by execute, modified_call_string actually executed)
    """
    from copy import copy

    modified_call = resolve_sub_arrangements(call_string or "", execute_results)

    meta_program_with_call = copy(meta_program)
    meta_program_with_call.append_code(f"objs = {modified_call}")

    execution_globals = {
        "execute_results": execute_results,
        **{f"arrangement_{i}": arr for i, arr in enumerate(execute_results)}
    }
    if extra_globals:
        execution_globals.update(extra_globals)

    locals_dict = execute(meta_program_with_call, execution_globals)
    return locals_dict, modified_call