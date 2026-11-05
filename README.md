# Dynamix Three Space Blender Extension

Import Dynamix Three Space (DTS) models from Torque3D and BeamNG.drive, and BeamNG cache Collada (`.cdae`, format version 31), into Blender.

The importer reads two UV channels, vertex colors, node translation and rotation, level of detail, collision meshes, skinned meshes, and sequence keyframes. Material names are imported. Material parameters such as color are not.

## Version support

DTS versions 19 through 26, BeamNG DTS version 29, and CDAE version 31. Version 30 is not a readable shape format.

Blender 4.2 or newer. The extension is built for 64-bit x86 and arm64 on Windows, Linux, and macOS.

## Installation

1. Download the extension zip for your platform from the [releases](https://github.com/PrzemekWolw/DynamixThreeSpaceBlenderAddon/releases).
2. In Blender, open Edit > Preferences > Get Extensions.
3. Open the dropdown at the top right and choose Install from Disk.
4. Select the zip.

## Building from source

CDAE files are a MessagePack stream, compressed with Zstandard once the body is large. Those two libraries are not bundled in Blender, and the wheel files are not committed. `tools/fetch_wheels.py` downloads pinned `msgpack` and `zstandard` wheels for all six platforms into `io_scene_dtst3d/wheels/` and writes them into `blender_manifest.toml`.

```
python tools/fetch_wheels.py
blender --command extension build --source-dir io_scene_dtst3d --output-dir dist --split-platforms
```

The default is CPython 3.11, which Blender 4.2 through 4.5 embed. For a newer Blender, pass the Python version it embeds:

```
python tools/fetch_wheels.py --python-version 3.13
```

`--split-platforms` writes one zip per OS and architecture, each containing only the wheels that platform can load. A macOS universal2 wheel, when a package publishes one, is shared by both macOS zips.

To check that a folder of shapes parses, without opening Blender:

```
python tools/parse_check.py path\to\shapes
```
