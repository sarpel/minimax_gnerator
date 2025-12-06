"""
Configuration API Router

This module provides endpoints for managing WakeGen configuration files.
It allows the Web UI to create, load, validate, and save YAML configurations.

    ENDPOINTS:
    ==========
    GET  /template           - Get a blank configuration template
    POST /validate           - Validate configuration JSON/YAML
    POST /save               - Save configuration to a file
    GET  /load               - Load an existing configuration file
    GET  /presets            - List built-in configuration presets

    HOW THIS INTEGRATES:
    ====================
    Uses the existing configuration system from:
        - wakegen.config.yaml_loader (load_config, get_template_config)
        - wakegen.config.settings (validation)
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import yaml

from wakegen.config.yaml_loader import load_config, get_template_config, WakegenConfig
from wakegen.core.exceptions import ConfigError

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class ConfigTemplate(BaseModel):
    """Response containing a blank configuration template."""
    yaml_content: str = Field(..., description="YAML template content")
    sections: List[str] = Field(..., description="Available configuration sections")


class ConfigValidationRequest(BaseModel):
    """Request to validate configuration content."""
    content: str = Field(..., description="YAML content to validate")


class ConfigValidationResponse(BaseModel):
    """Response from configuration validation."""
    valid: bool = Field(..., description="Whether the configuration is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")


class ConfigSaveRequest(BaseModel):
    """Request to save configuration to a file."""
    content: str = Field(..., description="YAML content to save")
    path: str = Field(..., description="File path to save to")
    overwrite: bool = Field(False, description="Whether to overwrite existing file")


class ConfigSaveResponse(BaseModel):
    """Response from saving configuration."""
    success: bool = Field(..., description="Whether save was successful")
    path: str = Field(..., description="Path where file was saved")
    message: str = Field(..., description="Result message")


class ConfigLoadResponse(BaseModel):
    """Response containing loaded configuration."""
    content: str = Field(..., description="YAML content")
    path: str = Field(..., description="Path the file was loaded from")
    valid: bool = Field(..., description="Whether the loaded config is valid")


class ConfigPreset(BaseModel):
    """A built-in configuration preset."""
    name: str = Field(..., description="Preset name")
    description: str = Field(..., description="What this preset is for")
    filename: str = Field(..., description="Preset filename")


# =============================================================================
# ROUTER
# =============================================================================


router = APIRouter()


@router.get(
    "/template",
    response_model=ConfigTemplate,
    summary="Get configuration template"
)
async def get_template() -> ConfigTemplate:
    """
    Get a blank YAML configuration template.

    Returns a well-commented template that users can customize.
    This is the starting point for creating a new configuration.
    """
    try:
        # Get template from existing function
        template_dict = get_template_config()

        # Convert to YAML with nice formatting
        yaml_content = yaml.dump(
            template_dict,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True
        )

        # List of top-level sections
        sections = list(template_dict.keys())

        return ConfigTemplate(
            yaml_content=yaml_content,
            sections=sections
        )

    except Exception as e:
        logger.error(f"Error generating template: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating template: {str(e)}"
        )


@router.post(
    "/validate",
    response_model=ConfigValidationResponse,
    summary="Validate configuration"
)
async def validate_config(request: ConfigValidationRequest) -> ConfigValidationResponse:
    """
    Validate YAML configuration content.

    Checks that the configuration is syntactically correct YAML and
    contains all required fields with valid values.

        REQUEST BODY:
        =============
        content: The YAML string to validate

        RETURNS:
        ========
        ConfigValidationResponse with valid flag and any errors
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        # Step 1: Parse YAML
        try:
            config_dict = yaml.safe_load(request.content)
        except yaml.YAMLError as e:
            return ConfigValidationResponse(
                valid=False,
                errors=[f"YAML syntax error: {str(e)}"],
                warnings=[]
            )

        if not isinstance(config_dict, dict):
            return ConfigValidationResponse(
                valid=False,
                errors=["Configuration must be a YAML mapping (dictionary)"],
                warnings=[]
            )

        # Step 2: Check required sections
        required_sections = ["project", "generation"]
        for section in required_sections:
            if section not in config_dict:
                errors.append(f"Missing required section: '{section}'")

        # Step 3: Validate specific fields
        if "project" in config_dict:
            project = config_dict["project"]
            if not isinstance(project, dict):
                errors.append("'project' must be a mapping")
            else:
                if "name" not in project:
                    errors.append("'project.name' is required")

        if "generation" in config_dict:
            gen = config_dict["generation"]
            if not isinstance(gen, dict):
                errors.append("'generation' must be a mapping")
            else:
                if "wake_words" not in gen:
                    warnings.append("'generation.wake_words' not specified")
                elif not isinstance(gen["wake_words"], list):
                    errors.append("'generation.wake_words' must be a list")

        # Step 4: Try to create WakegenConfig object for full validation
        if not errors:
            try:
                # Save to temp file and load
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.yaml',
                    delete=False,
                    encoding='utf-8'
                ) as f:
                    f.write(request.content)
                    temp_path = f.name

                try:
                    config = load_config(Path(temp_path))
                except ConfigError as e:
                    errors.append(str(e))
                finally:
                    import os
                    os.unlink(temp_path)

            except Exception as e:
                errors.append(f"Validation error: {str(e)}")

        return ConfigValidationResponse(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return ConfigValidationResponse(
            valid=False,
            errors=[f"Unexpected error: {str(e)}"],
            warnings=[]
        )


@router.post(
    "/save",
    response_model=ConfigSaveResponse,
    summary="Save configuration to file"
)
async def save_config(request: ConfigSaveRequest) -> ConfigSaveResponse:
    """
    Save configuration content to a YAML file.

    Validates the configuration before saving to ensure it's correct.

        REQUEST BODY:
        =============
        content: YAML string to save
        path: File path to save to
        overwrite: Whether to overwrite if file exists
    """
    try:
        path = Path(request.path)

        # Check if file exists
        if path.exists() and not request.overwrite:
            return ConfigSaveResponse(
                success=False,
                path=str(path),
                message=f"File already exists: {path}. Set overwrite=true to replace."
            )

        # Validate before saving
        validation = await validate_config(
            ConfigValidationRequest(content=request.content)
        )
        if not validation.valid:
            return ConfigSaveResponse(
                success=False,
                path=str(path),
                message=f"Invalid configuration: {'; '.join(validation.errors)}"
            )

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        path.write_text(request.content, encoding='utf-8')

        return ConfigSaveResponse(
            success=True,
            path=str(path),
            message=f"Configuration saved to {path}"
        )

    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return ConfigSaveResponse(
            success=False,
            path=request.path,
            message=f"Error saving: {str(e)}"
        )


@router.get(
    "/load",
    response_model=ConfigLoadResponse,
    summary="Load configuration from file"
)
async def load_config_file(
    path: str = Query(..., description="Path to configuration file")
) -> ConfigLoadResponse:
    """
    Load a configuration file and return its contents.

        QUERY PARAMETERS:
        =================
        path: Absolute or relative path to the YAML file
    """
    try:
        file_path = Path(path)

        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {path}"
            )

        # Read the file
        content = file_path.read_text(encoding='utf-8')

        # Validate it
        validation = await validate_config(
            ConfigValidationRequest(content=content)
        )

        return ConfigLoadResponse(
            content=content,
            path=str(file_path.absolute()),
            valid=validation.valid
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading file: {str(e)}"
        )


@router.get(
    "/presets",
    response_model=List[ConfigPreset],
    summary="List configuration presets"
)
async def list_presets() -> List[ConfigPreset]:
    """
    List available built-in configuration presets.

    These are example configurations for common use cases.
    """
    # Define some built-in presets
    presets = [
        ConfigPreset(
            name="Basic English",
            description="Simple configuration for English wake words",
            filename="basic_english.yaml"
        ),
        ConfigPreset(
            name="Multi-language",
            description="Configuration for multiple languages",
            filename="multi_language.yaml"
        ),
        ConfigPreset(
            name="High Quality",
            description="Maximum quality with multiple providers and augmentation",
            filename="high_quality.yaml"
        ),
    ]

    return presets
