from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any
from jinja2 import Template

# We use 'logging' to print messages to the console in a structured way.
logger = logging.getLogger(__name__)

# This is a template for the training script.
# It's like a "fill-in-the-blanks" form for code.
# We use {{ variable_name }} to mark the blanks.
TRAINING_SCRIPT_TEMPLATE = """
import openwakeword
import torch
from openwakeword.model import Model
from openwakeword.train import Train

# 1. Configuration
# These values were filled in by the generator
TARGET_MODEL_NAME = "{{ model_name }}"
TRAIN_DATA_PATH = "{{ train_data_path }}"
VAL_DATA_PATH = "{{ val_data_path }}"
STEPS = {{ steps }}
BATCH_SIZE = {{ batch_size }}
LEARNING_RATE = {{ learning_rate }}

def train_model():
    print(f"Starting training for {TARGET_MODEL_NAME}...")
    
    # 2. Initialize the training process
    # This is where we tell OpenWakeWord how to train our model
    trainer = Train(
        model_name=TARGET_MODEL_NAME,
        train_data_path=TRAIN_DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        steps=STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # 3. Start training
    # This might take a while depending on your hardware!
    trainer.train()
    
    print("Training complete!")
    print(f"Model saved to {TARGET_MODEL_NAME}.onnx")

if __name__ == "__main__":
    train_model()
"""

async def generate_training_script(
    export_dir: str,
    output_script_path: str,
    model_name: str = "my_wakeword",
    steps: int = 10000,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> None:
    """
    Generates a Python script to train an OpenWakeWord model.

    Instead of running the training directly (which might be long and complex),
    we generate a script that the user can run separately. This gives the user
    more control and allows them to run it on a different machine if needed.

    Args:
        export_dir: The directory containing the exported dataset (with train.json/val.json).
        output_script_path: Where to save the generated Python script.
        model_name: The name to give the trained model.
        steps: How many training steps to run.
        batch_size: How many samples to process at once.
        learning_rate: How fast the model should learn (too fast = unstable, too slow = takes forever).
    """
    export_path = Path(export_dir)
    
    # We assume the standard names created by our splitter
    train_json = export_path / "train.json"
    val_json = export_path / "val.json"

    if not train_json.exists() or not val_json.exists():
        raise FileNotFoundError(
            f"Training data not found in {export_dir}. Did you run the splitter?"
        )

    # 1. Prepare the values for the template
    # We need to use absolute paths so the script works from anywhere
    context: Dict[str, Any] = {
        "model_name": model_name,
        "train_data_path": str(train_json.absolute()).replace("\\", "/"), # Fix for Windows paths in Python strings
        "val_data_path": str(val_json.absolute()).replace("\\", "/"),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }

    # 2. Render the template
    # This replaces the {{ variable_name }} placeholders with actual values
    template = Template(TRAINING_SCRIPT_TEMPLATE)
    script_content = template.render(context)

    # 3. Save the script
    output_path = Path(output_script_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    logger.info(f"Training script generated at {output_path}")
    logger.info("You can run it with: python " + output_path.name)