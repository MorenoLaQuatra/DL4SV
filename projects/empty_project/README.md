# Deep Learning Project Template

This directory contains a starting template for your course project. It is structured to follow best practices in deep learning research and development, utilizing configuration files and experiment tracking.

## Structure

*   `configs/`: Contains YAML configuration files.
    *   `config.yaml`: Default configuration parameters.
*   `data/`: Contains dataset class definitions.
    *   `dataset.py`: Implements `CustomDataset` with placeholders for Audio and Vision tasks.
*   `models/`: Contains model definitions.
    *   `model.py`: Implements `get_model` factory, `CustomCNN`, and a placeholder `CustomTransformer`.
*   `utils/`: Utility scripts.
    *   `exp_manager.py`: Handles experiment tracking (CometML or Console).
*   `train.py`: Main training script.
*   `test.py`: Evaluation script.
*   `requirements.txt`: Python dependencies.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    The project uses `yaml_config_manager`. Edit `configs/config.yaml` to set your parameters (paths, hyperparameters, model choice).
    
    You can override parameters from the command line:
    ```bash
    python train.py --config configs/config.yaml --training.lr 0.0005 --training.epochs 20
    ```

3.  **Data Loading:**
    Edit `data/dataset.py`.
    *   Select your modality in `config.yaml` (`vision` or `audio`).
    *   Implement the file loading logic in `__init__`.
    *   Implement the data loading and transformation logic in `__getitem__`.
    *   Ensure `__getitem__` returns a dictionary (e.g., `{'input': ..., 'target': ...}`).

4.  **Model Implementation:**
    Edit `models/model.py`.
    *   Modify `CustomCNN` or implement `CustomTransformer`.
    *   Update `get_model` if adding new architectures.

5.  **Experiment Tracking:**
    *   **Console**: Default mode. Prints metrics to stdout.
    *   **CometML**: Set `experiment.tracking: "comet"` in config. Provide your API key via config or `COMET_API_KEY` env var.

## Usage

**Train:**
```bash
python train.py --config configs/config.yaml
```

**Test:**
```bash
python test.py --config configs/config.yaml
```

## Hints

*   **HuggingFace**: The `dataset.py` is set up to accept a `processor`. Instantiate it in `train.py` (e.g., `AutoProcessor.from_pretrained(...)`) and pass it to the dataset.
*   **Audio**: Use `torchaudio.load()` and `torchaudio.transforms`.
*   **Vision**: Use `PIL.Image` and `torchvision.transforms`.
*   **Placeholders**: Look for comments like `# --- PLACEHOLDER LOGIC ---` or `TODO` to find where to add your code.