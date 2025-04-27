import os
from datetime import datetime

def list_project_structure(base_path='.', max_depth=5, ignore_dirs=None):
    if ignore_dirs is None:
        ignore_dirs = {'.git', '__pycache__', '.ipynb_checkpoints', 'venv', '.venv', 'build', 'dist', '.idea', '.vscode'}

    lines = []

    def walk(dir_path, depth):
        if depth > max_depth:
            return
        for entry in sorted(os.listdir(dir_path)):
            full_path = os.path.join(dir_path, entry)
            if os.path.isdir(full_path):
                if entry in ignore_dirs:
                    continue
                lines.append('    ' * depth + f'- {entry}/')
                walk(full_path, depth + 1)
            else:
                lines.append('    ' * depth + f'- {entry}')

    walk(base_path, 0)
    return lines

def save_to_file(lines, output_file='project_structure.txt'):
    # 加入时间戳标记，避免混乱
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n--- Project Structure generated at {timestamp} ---\n")
        for line in lines:
            f.write(line + '\n')

if __name__ == "__main__":
    project_lines = list_project_structure()
    save_to_file(project_lines)
    print("✅ Project structure has been saved to 'project_structure.txt'")