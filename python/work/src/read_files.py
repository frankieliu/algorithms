import os

def list_files(directory_path):
    """Reads all files in a directory using os.listdir()."""
    out = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):  # Ensure it's a file
            out.append(file_path)
    return out
