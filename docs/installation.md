# Installation Guide

Welcome to the **WakeGen** installation guide! This document will help you set up the wake word generator on your system.

## Prerequisites

Before we start, make sure you have the following:

1.  **Python 3.10 or higher**: This is the programming language we use.
    *   Check your version: `python --version`
2.  **pip**: The Python package installer.
3.  **Git**: To download the code.

## Standard Installation (Windows/Linux/Mac)

### 1. Clone the Repository

First, download the code from GitHub:

```bash
git clone https://github.com/yourusername/wakegen.git
cd wakegen
```

### 2. Create a Virtual Environment

It's best practice to keep project dependencies separate from your system.

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

Install the required libraries using `pip`:

```bash
pip install -e .
```

This installs `wakegen` in "editable" mode, so changes to the code are reflected immediately.

## Raspberry Pi Installation

Running on a Raspberry Pi (Zero 2 W, 4, or 5) requires a few extra steps due to hardware limitations.

### 1. System Dependencies

Install system libraries needed for audio processing:

```bash
sudo apt-get update
sudo apt-get install -y libsndfile1 portaudio19-dev python3-dev
```

### 2. Install WakeGen

Follow the standard installation steps above.

### 3. Optimization for Pi Zero

If you are using a Pi Zero, we recommend using **Piper TTS** instead of Edge TTS or Coqui, as it is much faster on low-power devices.

## Verifying Installation

To check if everything is working, run the help command:

```bash
wakegen --help
```

You should see a list of available commands.

## Troubleshooting

### "Command not found: wakegen"

Make sure your virtual environment is activated. If it is, try reinstalling the package:

```bash
pip install -e .
```

### Audio Driver Issues

If you see errors related to `portaudio` or `libsndfile`, ensure you installed the system dependencies listed in the Raspberry Pi section (these are often needed on Linux too).