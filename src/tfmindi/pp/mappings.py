"""Helper functions for creating mappings between example indices and metadata."""

from __future__ import annotations

import os
import re

import numpy as np


def create_cell_type_mapping(
    cell_type_files: list[str] | dict[str, str],
    file_pattern: str = "{cell_type}_oh.npz",
) -> dict[int, str]:
    """
    Create mapping from example_idx to cell types based on file organization.

    This function assumes that data files are organized by cell type, where each file
    contains a consecutive range of examples. It determines the mapping by loading
    file sizes and creating sequential index assignments.

    Parameters
    ----------
    cell_type_files
        Either a list of file paths or a dictionary {cell_type: file_path}.
        Files should contain numpy arrays with shape (n_examples, ...).
    file_pattern
        Pattern for extracting cell type from filenames when using list input.
        Use {cell_type} as placeholder (default: "{cell_type}_oh.npz").

    Returns
    -------
    Dictionary mapping example_idx to cell type names.

    Examples
    --------
    >>> import tfmindi as tm
    >>> # From list of file paths - extracts cell types from filenames
    >>> cell_files = ["Neuron_oh.npz", "Astrocyte_oh.npz", "Microglia_oh.npz"]
    >>> mapping = tm.pp.create_cell_type_mapping(cell_files)
    >>> # Result: {0: 'Neuron', 1: 'Neuron', ..., 1000: 'Astrocyte', ...}
    >>> # From dictionary - explicit cell type specification
    >>> cell_files = {
    ...     "Neuron": "data/Neuron_oh.npz",
    ...     "Astrocyte": "data/Astrocyte_oh.npz",
    ...     "Microglia": "data/Microglia_oh.npz",
    ... }
    >>> mapping = tm.pp.create_cell_type_mapping(cell_files)
    """
    if isinstance(cell_type_files, dict):
        # Direct mapping provided
        cell_type_mapping = {}
        current_idx = 0

        for cell_type, file_path in cell_type_files.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Load and determine size
            data = np.load(file_path)
            if "arr_0" in data:
                oh_data = data["arr_0"]
            else:
                # Try to find the main array
                arrays = list(data.keys())
                if len(arrays) == 1:
                    oh_data = data[arrays[0]]
                else:
                    raise ValueError(f"Could not determine main array in {file_path}. Available: {arrays}")

            n_examples = oh_data.shape[0]

            # Assign this cell type to the next n_examples indices
            for i in range(current_idx, current_idx + n_examples):
                cell_type_mapping[i] = cell_type

            current_idx += n_examples

        return cell_type_mapping

    else:
        # Extract cell types from file patterns
        pattern_regex = file_pattern.replace("{cell_type}", r"(.+?)")

        cell_type_mapping = {}
        current_idx = 0

        for file_path in sorted(cell_type_files):  # Sort for reproducible order
            # Extract cell type from filename
            filename = os.path.basename(file_path)
            match = re.match(pattern_regex, filename)
            if not match:
                raise ValueError(f"File '{filename}' does not match pattern '{file_pattern}'")

            cell_type = match.group(1)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Load and determine size
            data = np.load(file_path)
            if "arr_0" in data:
                oh_data = data["arr_0"]
            else:
                # Try to find the main array
                arrays = list(data.keys())
                if len(arrays) == 1:
                    oh_data = data[arrays[0]]
                else:
                    raise ValueError(f"Could not determine main array in {file_path}. Available: {arrays}")

            n_examples = oh_data.shape[0]

            # Assign this cell type to the next n_examples indices
            for i in range(current_idx, current_idx + n_examples):
                cell_type_mapping[i] = cell_type

            current_idx += n_examples

        return cell_type_mapping
