"""
Scene Motif decomposition
"""

import asyncio
import json
import logging
import traceback
import yaml
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import hsm_core.vlm.gpt as gpt
from hsm_core.config import PROMPT_DIR

if TYPE_CHECKING:
    from hsm_core.scene.motif import SceneMotif

logger = logging.getLogger(__name__)


async def decompose_motif_async(
    motif: "SceneMotif",
    max_attempts: int = 3,
    session: Optional[gpt.Session] = None,
    custom_logger: Optional[logging.Logger] = None
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Asynchronously decompose a single motif into arrangement JSON.

    Args:
        motif: The motif to decompose
        max_attempts: Maximum number of decomposition attempts
        session: Optional existing session to use (will create new if None)
        custom_logger: Optional logger for detailed logging (will use module logger if None)

    Returns:
        Tuple of (arrangement_json, validation_response)
    """
    from ..utils import send_llm_async, send_llm_with_validation_async
    from ..utils.validation import validate_remaining_arrangements, validate_compositional_json

    try:
        if session is not None:
            decompose_session = session
        else:
            decompose_session = gpt.Session(
                str(PROMPT_DIR / "sm_prompts_decompose.yaml"),
                prompt_info={"MOTIF_DEFINITIONS": yaml.safe_load(open(PROMPT_DIR / "motif_definitions.yaml"))["motifs"]}
            )

        log = custom_logger if custom_logger is not None else logging.getLogger(__name__)

        furniture_count = {spec.name: spec.amount for spec in motif.object_specs}

        if custom_logger:
            log.info(f"Starting decomposition for motif {motif.id}")
            log.info(f"Furniture count: {furniture_count}")

        for attempt in range(max_attempts):
            if custom_logger:
                log.info(f"Decomposition attempt {attempt + 1}/{max_attempts}")

            if attempt > 0:
                decompose_session.add_feedback(f"RETRY ATTEMPT {attempt + 1}: Previous attempts failed. You MUST change your approach.")

            # Step 1: Primary arrangement
            if custom_logger:
                log.info(f"Step 1: Identifying primary arrangement...")

            primary_arrangement = await send_llm_async(
                decompose_session, "identify_primary_arrangement",
                {"description": motif.description, "object_counts": str(furniture_count), "furniture_info": ([spec.to_gpt_furniture_info() for spec in motif.object_specs])},
                verbose=True
            )

            # get remaining objects
            primary_json = json.loads(gpt.extract_json(primary_arrangement))

            if custom_logger:
                log.info(f"Primary arrangement identified: {primary_json.get('motif_type', 'unknown')} with objects {list(primary_json.get('objects', {}).keys())}")

            primary_objects = set(item.lower() for item in primary_json.get("objects", {}).keys())
            if not primary_objects:
                error_msg = "Primary arrangement must include furniture"
                if custom_logger:
                    log.error(error_msg)
                raise ValueError(error_msg)
            remaining_objects = list(set(spec.name.lower() for spec in motif.object_specs) - primary_objects)

            if custom_logger:
                log.info(f"Remaining objects after primary: {remaining_objects}")

            # Step 2: Secondary arrangements
            if custom_logger:
                log.info(f"Step 2: Processing secondary arrangements...")

            if not remaining_objects:
                if custom_logger:
                    log.info(f"All objects used in primary arrangement")
                secondary_arrangements = json.dumps({"message": "All objects used in primary arrangement"})
                # Can start compositional JSON generation immediately
                arrangement_json_task = asyncio.create_task(send_llm_with_validation_async(
                    decompose_session, "generate_compositional_json",
                    {"description": motif.description, "primary_arrangement": primary_arrangement, "secondary_arrangements": secondary_arrangements},
                    validate_compositional_json, json=True
                ))
                arrangement_json = await arrangement_json_task
            else:
                if custom_logger:
                    log.info(f"Identifying arrangements for remaining objects: {remaining_objects}")
                secondary_arrangements = await send_llm_with_validation_async(
                    decompose_session, "identify_remaining_arrangements",
                    {"description": motif.description, "primary_arrangement": primary_arrangement, "remaining_objects": str(remaining_objects)},
                    lambda r: validate_remaining_arrangements(r, remaining_objects), json=True
                )

                if custom_logger:
                    log.info(f"Generating final compositional JSON...")
                arrangement_json = await send_llm_with_validation_async(
                    decompose_session, "generate_compositional_json",
                    {"description": motif.description, "primary_arrangement": primary_arrangement, "secondary_arrangements": secondary_arrangements},
                    validate_compositional_json, json=True
                )
            
            # Step 3: Validation
            if custom_logger:
                log.info(f"Step 3: Validating arrangement...")

            validate_response = json.loads(await send_llm_async(
                decompose_session, "validate_arrangement",
                {"arrangement_json": arrangement_json}, json=True
            ))

            if custom_logger:
                log.info(f"Validation result: {'PASSED' if validate_response.get('is_valid') else 'FAILED'}")

            if not validate_response.get("is_valid"):
                checks = validate_response["checks"]
                feedback = []
                if checks.get("motifs", {}).get("issues"):
                    feedback.append(f"MOTIF ERRORS: {checks['motifs']['issues']}")
                if checks.get("hierarchy", {}).get("issues"):
                    feedback.append(f"HIERARCHY ERRORS: {checks['hierarchy']['issues']}")
                if checks.get("completeness", {}).get("missing_items"):
                    feedback.append(f"COMPLETENESS ERRORS: {checks['completeness']['missing_items']}")
                if validate_response.get("fixes"):
                    feedback.append(f"REQUIRED FIXES: {validate_response['fixes']}")

                feedback_message = f"Decomposition validation failed - ATTEMPT {attempt + 1}: {' | '.join(feedback)}."
                decompose_session.add_feedback(feedback_message)
                if custom_logger:
                    log.warning(f"Added validation feedback to session: {feedback_message}")
                continue
            
            if custom_logger:
                log.info(f"Successfully completed decomposition for motif {motif.id}")
            return arrangement_json, validate_response

        if custom_logger:
            log.error(f"All {max_attempts} decomposition attempts failed for motif {motif.id}")
        return None, None

    except Exception as e:
        if custom_logger:
            log.error(f"Exception in decomposition for motif {motif.id}: {str(e)}")
            log.error(f"Traceback: {traceback.format_exc()}")
        else:
            module_logger = logging.getLogger(__name__)
            module_logger.error(f"Error decomposing motif {motif.id}: {str(e)}")
            module_logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None


async def decompose_motif_with_session(
    motif: "SceneMotif",
    custom_logger: logging.Logger,
    decompose_session: gpt.Session,
    max_decompose_attempts: int = 2
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Legacy function that calls the unified decompose_motif_async function.

    This function is kept for backward compatibility but internally uses
    the unified decompose_motif_async function.
    """
    return await decompose_motif_async(
        motif=motif,
        max_attempts=max_decompose_attempts,
        session=decompose_session,
        custom_logger=custom_logger
    )