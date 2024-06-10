import os

def print_directory_contents(path, level=0):
    if level == 0:
        print(f"Root directory: {path}")
    prefix = "    " * level
    try:
        for item in os.listdir(path):
            if item == '.git':
                continue
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(f"{prefix}|-- {item}/")
                print_directory_contents(item_path, level + 1)
            else:
                print(f"{prefix}|-- {item}")
    except PermissionError as e:
        print(f"{prefix}|-- [PermissionError] {e}")
    except FileNotFoundError as e:
        print(f"{prefix}|-- [FileNotFoundError] {e}")

# Ganti 'your_project_directory' dengan path direktori proyek Anda yang sebenarnya
project_directory = 'C:\\Code Qorina'
print_directory_contents(project_directory)
