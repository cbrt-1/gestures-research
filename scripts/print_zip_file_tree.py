import sys
import zipfile
from collections import defaultdict

def print_tree(d, indent=''):
    for i, (key, val) in enumerate(sorted(d.items())):
        is_last = i == len(d) - 1
        connector = '└── ' if is_last else '├── '
        print(f'{indent}{connector}{key}')
        if isinstance(val, dict):
            child_indent = '    ' if is_last else '│   '
            print_tree(val, indent + child_indent)

def print_zip_tree(zip_path):
    try:
        with zipfile.ZipFile(zip_path) as zf:
            tree = lambda: defaultdict(tree)
            root = tree()
            for path in zf.namelist():
                node = root
                for part in path.split('/'):
                    if part:
                        node = node[part]


            print_tree(root)

    except (FileNotFoundError, zipfile.BadZipFile):
        print(f"Error: Cannot open or read '{zip_path}'", file=sys.stderr)


if __name__ == "__main__":
    print_zip_tree(input("path to zip file: "))