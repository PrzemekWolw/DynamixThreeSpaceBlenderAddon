from typing import List

from .tsalloc import *


class TSDrawPrimitiveType:
    Triangles = 0 << 30
    Strip = 1 << 30
    Fan = 2 << 30

    Indexed = 1 << 29
    NoMaterial = 1 << 28

    TypeMask = Strip | Fan | Triangles
    # Bits 28-31 are type flags. Keep this masked to 32 bits: Python's ~
    # would otherwise sign-extend and corrupt fan primitives.
    MaterialMask = (~(Strip | Fan | Triangles | Indexed | NoMaterial)) & 0xFFFFFFFF


class TSMeshFlags:
    # graphics/ts/tsMesh.h. These live in the high bits of the mesh flags word.
    UseEncodedNormals = 1 << 28
    BillboardZAxis = 1 << 29
    HasDetailTexture = 1 << 30
    Billboard = 1 << 31


class TSDrawPrimitive:
    def __init__(self, start, num_elements, material_and_flags):
        self.start = start
        self.num_elements = num_elements
        material_and_flags &= 0xFFFFFFFF
        self.has_no_material = (material_and_flags & TSDrawPrimitiveType.NoMaterial) != 0
        self.type = material_and_flags & TSDrawPrimitiveType.TypeMask
        self.material_index = material_and_flags & TSDrawPrimitiveType.MaterialMask


class TSNullMesh:
    pass


class TSDecalMesh(TSNullMesh):
    """Decal meshes carry no geometry the importer uses; this only moves past them."""

    def assemble(self, ts_alloc, version: int):
        if version < 20:
            ts_alloc.check_guard()
            ts_alloc.skip32(15)
        primitive_count = ts_alloc.read32()
        ts_alloc.skip16(primitive_count * 2)
        ts_alloc.skip32(primitive_count)
        ts_alloc.skip16(ts_alloc.read32())
        if version < 20:
            ts_alloc.skip32(3)
            ts_alloc.check_guard()
        start_count = ts_alloc.read32()
        ts_alloc.skip32(start_count)
        if version >= 19:
            ts_alloc.skip32(start_count * 8)
        ts_alloc.read32()  # material index
        ts_alloc.check_guard()


def _triples(buffer):
    return list(zip(buffer[0::3], buffer[1::3], buffer[2::3]))


def _pairs(buffer):
    return list(zip(buffer[0::2], buffer[1::2]))


class TSMesh:
    def __init__(self):
        self._vertices: List[tuple] = []
        self._tvertices: List[tuple] = []
        self._t2vertices: List[tuple] = []
        self._colors: List[tuple] = []
        self._normals: List[tuple] = []
        self._tangents: List[tuple] = []
        self._primitives: List[TSDrawPrimitive] = []
        self._indices: List[int] = []
        self._parent_mesh: int = -1
        self._flags: int = 0
        self._num_frames: int = 1
        self._num_mat_frames: int = 1
        self._verts_per_frame: int = 0

    @property
    def vertices(self):
        return self._vertices

    @property
    def normals(self):
        return self._normals

    @property
    def tangents(self):
        return self._tangents

    @property
    def tvertices(self):
        return self._tvertices

    @property
    def t2vertices(self):
        return self._t2vertices

    @property
    def colors(self):
        return self._colors

    @property
    def primitives(self):
        return self._primitives

    @property
    def indices(self):
        return self._indices

    @property
    def parent_mesh(self) -> int:
        return self._parent_mesh

    @property
    def flags(self) -> int:
        return self._flags

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_mat_frames(self) -> int:
        return self._num_mat_frames

    @property
    def verts_per_frame(self) -> int:
        return self._verts_per_frame

    def copy_vertex_data_from(self, other):
        self._vertices = other._vertices.copy()
        self._tvertices = other._tvertices.copy()
        self._t2vertices = other._t2vertices.copy()
        self._colors = other._colors.copy()
        self._normals = other._normals.copy()
        self._tangents = other._tangents.copy()

    def _read_owned(self, ts_alloc, count, width, unpack):
        """Vertex arrays are in the buffer only for meshes with no parent."""
        if self._parent_mesh < 0 and count:
            return unpack(ts_alloc.read_float_list(count * width))
        return []

    def assemble(self, ts_alloc: TSAlloc, version: int):
        ts_alloc.check_guard()

        self._num_frames = ts_alloc.read32()
        self._num_mat_frames = ts_alloc.read32()
        self._parent_mesh = ts_alloc.read32()

        # bounds (6), center (3), radius
        ts_alloc.skip32(10)

        num_verts = ts_alloc.read32()
        self._vertices = self._read_owned(ts_alloc, num_verts, 3, _triples)

        num_tverts = ts_alloc.read32()
        self._tvertices = self._read_owned(ts_alloc, num_tverts, 2, _pairs)

        if version > 25:
            num_t2verts = ts_alloc.read32()
            self._t2vertices = self._read_owned(ts_alloc, num_t2verts, 2, _pairs)

            num_colors = ts_alloc.read32()
            if self._parent_mesh < 0 and num_colors:
                for packed in ts_alloc.read_u32_list(num_colors):
                    self._colors.append((
                        (packed & 0xFF) / 255.0,
                        ((packed >> 8) & 0xFF) / 255.0,
                        ((packed >> 16) & 0xFF) / 255.0,
                        ((packed >> 24) & 0xFF) / 255.0,
                    ))

        self._normals = self._read_owned(ts_alloc, num_verts, 3, _triples)
        if version > 21 and self._parent_mesh < 0:
            ts_alloc.skip8(num_verts)

        starts = []
        counts = []
        materials = []
        if version > 25:
            prim_count = ts_alloc.read32()
            for _ in range(prim_count):
                starts.append(ts_alloc.read32())
                counts.append(ts_alloc.read32())
                materials.append(ts_alloc.read32() & 0xFFFFFFFF)
            index_count = ts_alloc.read32()
            self._indices = ts_alloc.read_u32_list(index_count)
        else:
            prim_count = ts_alloc.read32()
            for _ in range(prim_count):
                starts.append(ts_alloc.read16())
                counts.append(ts_alloc.read16())
            for _ in range(prim_count):
                materials.append(ts_alloc.read32() & 0xFFFFFFFF)
            index_count = ts_alloc.read32()
            # Unsigned: a signed 16-bit read turns indices past 32767 negative.
            self._indices = ts_alloc.read_u16_list(index_count)

        for start, count, material in zip(starts, counts, materials):
            self._primitives.append(TSDrawPrimitive(start, count, material))

        merge_count = ts_alloc.read32()
        ts_alloc.skip16(merge_count)
        ts_alloc.align32()

        self._verts_per_frame = ts_alloc.read32()
        self._flags = ts_alloc.read32() & 0xFFFFFFFF

        # Point4F tangents, present from version 28 onward. Owned by the parent mesh.
        if version >= 28:
            tangent_count = ts_alloc.read32()
            if self._parent_mesh < 0 and tangent_count:
                raw = ts_alloc.read_float_list(tangent_count * 4)
                self._tangents = list(zip(raw[0::4], raw[1::4], raw[2::4], raw[3::4]))

        ts_alloc.check_guard()


class TSSortedMesh(TSMesh):
    """A standard mesh followed by translucency cluster data the importer does not use."""

    def assemble(self, ts_alloc: TSAlloc, version: int):
        super().assemble(ts_alloc, version)
        ts_alloc.skip32(ts_alloc.read32() * 8)  # clusters
        for _ in range(4):  # startCluster, firstVerts, numVerts, firstTVerts
            ts_alloc.skip32(ts_alloc.read32())
        ts_alloc.read32()  # alwaysWriteDepth
        ts_alloc.check_guard()


class TSSkinnedMesh(TSMesh):
    def __init__(self):
        super().__init__()
        self._initial_transforms: List[List[float]] = []
        self._vertex_index: List[int] = []
        self._bone_index: List[int] = []
        self._weight: List[float] = []
        self._node_index: List[int] = []

    @property
    def initial_transforms(self):
        return self._initial_transforms

    @property
    def vertex_index(self):
        return self._vertex_index

    @property
    def bone_index(self):
        return self._bone_index

    @property
    def weight(self):
        return self._weight

    @property
    def node_index(self):
        return self._node_index

    def copy_skin_from(self, other):
        self._initial_transforms = [row.copy() for row in other._initial_transforms]
        self._vertex_index = other._vertex_index.copy()
        self._bone_index = other._bone_index.copy()
        self._weight = other._weight.copy()
        self._node_index = other._node_index.copy()

    def assemble(self, ts_alloc: TSAlloc, version: int):
        super().assemble(ts_alloc, version)
        # Same layout as TSSkinMesh::assemble_v29, which reads every version before 30.
        # Every count is present; the arrays themselves are only stored when parentMesh < 0.
        initial_verts = ts_alloc.read32()
        if self._parent_mesh < 0:
            ts_alloc.skip32(initial_verts * 3)
            if version > 21:
                ts_alloc.skip32(initial_verts * 3)
                ts_alloc.skip8(initial_verts)
            else:
                ts_alloc.skip32(initial_verts * 3)

        transform_count = ts_alloc.read32()
        if self._parent_mesh < 0 and transform_count:
            raw = ts_alloc.read_float_list(transform_count * 16)
            self._initial_transforms = [raw[i * 16:(i + 1) * 16] for i in range(transform_count)]

        influence_count = ts_alloc.read32()
        if self._parent_mesh < 0:
            self._vertex_index = ts_alloc.read_u32_list(influence_count)
            self._bone_index = ts_alloc.read_u32_list(influence_count)
            self._weight = ts_alloc.read_float_list(influence_count)

        node_count = ts_alloc.read32()
        if self._parent_mesh < 0:
            self._node_index = ts_alloc.read_u32_list(node_count)

        ts_alloc.check_guard()
