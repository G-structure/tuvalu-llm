"""Shared dataset naming conventions for source building and generation."""


def dataset_name_to_filename(dataset_name: str) -> str:
    """Convert a HuggingFace dataset name to a safe filesystem name.

    E.g., "tasksource/tasksource-instruct-v0" -> "tasksource__tasksource-instruct-v0"
    """
    return dataset_name.replace("/", "__")


def filename_to_dataset_name(filename: str) -> str:
    """Reverse of dataset_name_to_filename.

    E.g., "tasksource__tasksource-instruct-v0" -> "tasksource/tasksource-instruct-v0"
    """
    return filename.replace("__", "/", 1)
