"""Download the msgpack and zstandard wheels this extension imports.

Blender 4.2 has no 32-bit builds, so each desktop OS is covered by its
64-bit x86 build and its arm64 build. A macOS universal2 wheel satisfies
both macOS platforms. The script rewrites the wheels list in
io_scene_dtst3d/blender_manifest.toml and exits if any platform is missing
a wheel for either package.

    python tools/fetch_wheels.py
    python tools/fetch_wheels.py --python-version 3.13
"""

import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDON = os.path.join(ROOT, "io_scene_dtst3d")
WHEELS = os.path.join(ADDON, "wheels")
MANIFEST = os.path.join(ADDON, "blender_manifest.toml")

# Pinned to releases that publish a CPython 3.11 binary wheel for every
# platform below, including Windows arm64.
PACKAGES = (
    ("msgpack", "1.2.4"),
    ("zstandard", "0.25.0"),
)

# Blender platform -> pip platform tags, most specific first.
# Extra tags are older ABIs pip would otherwise refuse when --platform is set.
PLATFORMS = (
    ("windows-x64", ("win_amd64",)),
    ("windows-arm64", ("win_arm64",)),
    ("linux-x64", ("manylinux_2_28_x86_64", "manylinux2014_x86_64", "manylinux_2_17_x86_64")),
    ("linux-arm64", ("manylinux_2_28_aarch64", "manylinux2014_aarch64", "manylinux_2_17_aarch64")),
    ("macos-x64", ("macosx_11_0_x86_64", "macosx_10_9_x86_64", "macosx_11_0_universal2")),
    ("macos-arm64", ("macosx_11_0_arm64", "macosx_11_0_universal2")),
)


def wheel_tags(filename):
    """Return the platform tags of a wheel filename."""
    stem = filename[:-4] if filename.endswith(".whl") else filename
    # {distribution}-{version}(-{build})?-{python}-{abi}-{platform}
    parts = stem.split("-")
    if len(parts) < 5:
        return []
    return parts[-1].split(".")


def wheel_matches(filename, pip_tags):
    tags = wheel_tags(filename)
    if "any" in tags:
        return True
    if "universal2" in tags and any("macosx" in tag for tag in pip_tags):
        return True
    return any(tag in pip_tags for tag in tags)


def download(python_version):
    os.makedirs(WHEELS, exist_ok=True)
    py = python_version.replace(".", "")
    abi = f"cp{py}"
    covered = {name: {dist: [] for dist, _ in PACKAGES} for name, _ in PLATFORMS}

    for blender_name, pip_tags in PLATFORMS:
        for dist, version in PACKAGES:
            before = set(os.listdir(WHEELS))
            cmd = [
                sys.executable, "-m", "pip", "download",
                f"{dist}=={version}",
                "--dest", WHEELS,
                "--only-binary=:all:",
                "--no-deps",
                "--implementation", "cp",
                "--python-version", py,
                "--abi", abi,
                "--exists-action", "i",
            ]
            for tag in pip_tags:
                cmd.extend(["--platform", tag])
            print(" ".join(cmd))
            subprocess.check_call(cmd)
            fresh = [name for name in os.listdir(WHEELS) if name not in before and name.endswith(".whl")]
            if not fresh:
                fresh = [
                    name for name in os.listdir(WHEELS)
                    if name.endswith(".whl") and name.startswith(dist.replace("_", "-"))
                    and wheel_matches(name, pip_tags)
                ]
            matches = [name for name in fresh if wheel_matches(name, pip_tags)]
            if not matches:
                raise SystemExit(f"No {dist} wheel for {blender_name} ({', '.join(pip_tags)})")
            covered[blender_name][dist] = matches

    missing = []
    for blender_name, packages in covered.items():
        for dist, wheels in packages.items():
            if not wheels:
                missing.append(f"{blender_name}: {dist}")
    if missing:
        raise SystemExit("Missing wheels:\n  " + "\n  ".join(missing))
    return sorted(glob.glob(os.path.join(WHEELS, "*.whl")))


def rewrite_manifest(wheel_paths):
    with open(MANIFEST, "r", encoding="utf-8") as handle:
        text = handle.read()
    rels = ["./wheels/" + os.path.basename(path) for path in wheel_paths]
    block = "wheels = [\n" + "".join(f'    "{rel}",\n' for rel in rels) + "]"
    updated, count = re.subn(
        r"# BEGIN WHEELS\nwheels = \[.*?\]\n# END WHEELS",
        "# BEGIN WHEELS\n" + block + "\n# END WHEELS",
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise SystemExit("Could not find the wheels block in blender_manifest.toml")
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(updated)
    print(f"Wrote {len(rels)} wheels into {MANIFEST}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-version", default="3.11", help="CPython version Blender embeds (default: 3.11)")
    args = parser.parse_args()
    rewrite_manifest(download(args.python_version))


if __name__ == "__main__":
    main()
