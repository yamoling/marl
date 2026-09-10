def deslugify(text: str) -> str:
    """Convert a slug (e.g., 'my-experiment') to a more human-readable format (e.g., 'My Experiment')."""
    return text.replace("-", " ").capitalize()
