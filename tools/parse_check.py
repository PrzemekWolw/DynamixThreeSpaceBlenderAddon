"""Parse every .dts and .cdae file under a folder and report failures.

The readers do not need Blender. CDAE files need msgpack and zstandard
installed for the Python that runs this script (the wheels under
io_scene_dtst3d/wheels are built for Blender's Python, not for this one):

    python -m pip install msgpack==1.2.4 zstandard==0.25.0
    python tools/parse_check.py path\\to\\shapes
"""

import argparse
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from io_scene_dtst3d.import_dts import file_version, read_shape  # noqa: E402


def _summarize(shape):
    meshes = len(shape.meshes)
    skinned = sum(1 for mesh in shape.meshes if mesh.__class__.__name__ == "TSSkinnedMesh")
    return (
        f"v{shape.version}  nodes={len(shape.nodes)}  objects={len(shape.objects)}  "
        f"meshes={meshes}  skinned={skinned}  sequences={len(shape.sequences)}  "
        f"materials={len(shape.materials)}"
    )


def check_tree(folder):
    paths = []
    for dirpath, _, filenames in os.walk(folder):
        for name in filenames:
            if name.lower().endswith((".dts", ".cdae")):
                paths.append(os.path.join(dirpath, name))
    paths.sort()
    if not paths:
        print(f"No .dts or .cdae files under {folder}")
        return 1

    failed = 0
    for path in paths:
        try:
            version = file_version(path)
            if version == 30:
                print(f"SKIP  v30  {path}")
                continue
            shape = read_shape(path, version)
            print(f"OK    {_summarize(shape)}  {path}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {path}\n      {exc}")
            traceback.print_exc()
    print(f"\n{len(paths) - failed} ok, {failed} failed, {len(paths)} files")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Directory to scan for .dts and .cdae files")
    args = parser.parse_args()
    if not os.path.isdir(args.folder):
        print(f"Not a directory: {args.folder}")
        return 2
    return check_tree(args.folder)


if __name__ == "__main__":
    sys.exit(main())
