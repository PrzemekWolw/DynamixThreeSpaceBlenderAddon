import struct
import time

from .tsshape import TSShape


def file_version(path: str) -> int:
    with open(path, "rb") as handle:
        data = handle.read(4)
    if len(data) != 4:
        raise ValueError("file is too small to contain a DTS version")
    return struct.unpack("<i", data)[0] & 0xFF


def read_shape(path: str, version: int):
    if version >= 31:
        from .cdae import read_shape as read_cdae
        return read_cdae(path)
    shape = TSShape()
    shape.read_from_path(path)
    return shape


def load(operator, context, filepath="", merge_verts=True):
    del context
    try:
        version = file_version(filepath)
    except OSError as exc:
        operator.report({"ERROR"}, f"Cannot read {filepath}: {exc}")
        return {"CANCELLED"}
    except ValueError as exc:
        operator.report({"ERROR"}, str(exc))
        return {"CANCELLED"}

    if hasattr(operator, "merge_verts"):
        merge_verts = bool(operator.merge_verts)

    if version == 30:
        operator.report({"ERROR"}, "Version 30 is not a shape. Delete the cached file so the game can rewrite it.")
        return {"CANCELLED"}
    if version < 19:
        operator.report({"ERROR"}, f"DTS version {version} is too old")
        return {"CANCELLED"}

    started = time.perf_counter()
    try:
        shape = read_shape(filepath, version)
        from .import_shape import build_scene
        build_scene(shape, merge_verts=merge_verts)
    except Exception as exc:
        operator.report({"ERROR"}, f"Import failed: {exc}")
        return {"CANCELLED"}

    print(f"imported {filepath} ({version}) in {time.perf_counter() - started:.4f} sec")
    return {"FINISHED"}
