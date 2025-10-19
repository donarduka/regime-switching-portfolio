import sys, pathlib
# Add repository root (parent of tests/) to sys.path so `import src.*` works
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
