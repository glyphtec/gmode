# Soundboard Lab

Blender add-on for luthier soundboard eigenmode experiments. See [docs/CLAUDE.md](docs/CLAUDE.md) for the full project context and decisions.

## Engine dev setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Set the CalculiX executable path for tests:

```powershell
$env:SOUNDBOARD_LAB_CCX_PATH = "C:\Program Files\FreeCAD 1.1\bin\ccx.exe"
pytest
```
