# UV Workflow Documentation

This project has been transitioned from Poetry to UV for dependency management.

## Project Structure Changes

### Files Kept/Modified:
- `pyproject.toml` - Kept minimal with only project metadata (no tool.poetry section)
- `.venv/` - Virtual environment created with uv

### Files Created:
- `requirements.in` - High-level dependencies (what you want to install)
- `requirements.txt` - Locked dependencies (generated automatically)
- `UV_WORKFLOW.md` - This documentation

### Files Removed:
- `poetry.lock` - No longer needed (replaced by requirements.txt)

## UV Workflow Commands

### 1. Virtual Environment Management
```bash
# Create virtual environment
uv venv .venv

# Activate virtual environment (fish shell)
source .venv/bin/activate.fish

# For bash/zsh users:
source .venv/bin/activate
```

### 2. Dependency Management
```bash
# Install dependencies from requirements.txt
uv pip install -r requirements.txt

# Add new dependency to requirements.in, then compile
echo "new-package>=1.0.0" >> requirements.in
uv pip compile requirements.in -o requirements.txt

# Install the newly compiled requirements
uv pip install -r requirements.txt
```

### 3. Lock Dependencies
```bash
# Compile requirements.in to requirements.txt (lock dependencies)
uv pip compile requirements.in -o requirements.txt

# For development dependencies (if needed in future)
# uv pip compile requirements-dev.in -o requirements-dev.txt
```

### 4. Package Installation (if needed)
```bash
# Install the project package in development mode
uv pip install -e .
```

## Key Benefits of UV

1. **Speed**: Much faster than pip/poetry for dependency resolution
2. **Simplicity**: Clear separation between high-level (requirements.in) and locked (requirements.txt) dependencies
3. **Compatibility**: Works with existing pip ecosystem
4. **Reproducibility**: requirements.txt provides exact version locks

## Migration Summary

- ✅ Removed Poetry-specific sections from pyproject.toml
- ✅ Created requirements.in with all dependencies from poetry.lock
- ✅ Generated locked requirements.txt with uv pip compile
- ✅ Created .venv virtual environment with uv venv
- ✅ Removed poetry.lock file
- ✅ Maintained project metadata in minimal pyproject.toml

The project is now ready to use the UV workflow for dependency management!
