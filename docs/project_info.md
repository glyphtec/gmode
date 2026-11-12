# Project info

- **Project:** Soundboard Lab
- **Repository:** <https://github.com/glyphtec/soundboard-lab>
- **Landing page:** <https://glyphstone.studio/projects/soundboard-lab>
- **Maintainer contact:** breanna@glyphstone.net
- **License:** GPL-3.0-or-later; the add-on stays free (paid work is consulting and data packs only)

The maintainer and website values are also set in
`addon/soundboard_lab/blender_manifest.toml`; keep the two in sync.

## Building the add-on zip

With the Blender executable on hand (any version that supports extensions):

```powershell
New-Item -ItemType Directory dist -Force
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --command extension build `
    --source-dir addon\soundboard_lab --output-dir dist
```

This writes `dist\soundboard_lab-<version>.zip` (`dist/` is git-ignored). Users install
it with Preferences > Get Extensions > Install from Disk, then set the CalculiX and
engine Python paths in the add-on preferences.

The zip contains only the add-on. The engine (`src/soundboard_lab`) and CalculiX are
not bundled yet, so pre-release users still need the engine venv
(`pip install .` from the repo) and a `ccx.exe`.
