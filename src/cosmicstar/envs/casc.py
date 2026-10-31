"""Reading files out of the game's local CASC storage by name, through CascLib.

StarCraft II's TACT root is an MNDX file: a compressed trie of file names (envs/tact.py can
read the root's bytes from the storage but does not parse the trie). CascLib implements it,
so this module loads libcasc.so (built from https://github.com/ladislav-zezula/CascLib, see
the README) with ctypes and exposes three things: open a storage at a build, list files by a
wildcard mask, read a file.

    from cosmicstar.envs.casc import Storage
    with Storage(install_dir, build_key="05cce99544c8b80c07f6eb3250b957a9") as st:
        for name in st.find("*GameData/UpgradeData.xml"):
            data = st.read(name)

The library path comes from $CASCLIB (a libcasc.so) or the repo's `lib/libcasc.so`.
"""
import ctypes
import os

MAX_PATH = 1024
MD5 = 16
CASC_LOCALE_ALL = 0xFFFFFFFF
CASC_OPEN_BY_NAME = 0
LIB_CANDIDATES = (os.environ.get("CASCLIB", ""), os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib", "libcasc.so"),
                  "libcasc.so")


class FindData(ctypes.Structure):
    _fields_ = [("szFileName", ctypes.c_char * MAX_PATH), ("CKey", ctypes.c_ubyte * MD5), ("EKey", ctypes.c_ubyte * MD5),
                ("TagBitMask", ctypes.c_ulonglong), ("FileSize", ctypes.c_ulonglong), ("szPlainName", ctypes.c_char_p),
                ("dwFileDataId", ctypes.c_uint32), ("dwLocaleFlags", ctypes.c_uint32), ("dwContentFlags", ctypes.c_uint32),
                ("dwSpanCount", ctypes.c_uint32), ("bFileAvailable", ctypes.c_uint32), ("NameType", ctypes.c_int)]


class OpenArgs(ctypes.Structure):
    _fields_ = [("Size", ctypes.c_size_t), ("szLocalPath", ctypes.c_char_p), ("szCodeName", ctypes.c_char_p), ("szRegion", ctypes.c_char_p),
                ("PfnProgressCallback", ctypes.c_void_p), ("PtrProgressParam", ctypes.c_void_p), ("PfnProductCallback", ctypes.c_void_p),
                ("PtrProductParam", ctypes.c_void_p), ("dwLocaleMask", ctypes.c_uint32), ("dwFlags", ctypes.c_uint32),
                ("szBuildKey", ctypes.c_char_p), ("szCdnHostUrl", ctypes.c_char_p)]


def load_library():
    last = None
    for p in LIB_CANDIDATES:
        if not p:
            continue
        try:
            lib = ctypes.CDLL(p)
            break
        except OSError as e:
            last = e
    else:
        raise OSError(f"libcasc.so not found (set $CASCLIB or build CascLib into lib/libcasc.so): {last}")
    H = ctypes.c_void_p
    lib.CascOpenStorageEx.argtypes = [ctypes.c_char_p, ctypes.POINTER(OpenArgs), ctypes.c_bool, ctypes.POINTER(H)]
    lib.CascOpenStorageEx.restype = ctypes.c_bool
    lib.CascCloseStorage.argtypes = [H]
    lib.CascOpenFile.argtypes = [H, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(H)]
    lib.CascOpenFile.restype = ctypes.c_bool
    lib.CascGetFileSize64.argtypes = [H, ctypes.POINTER(ctypes.c_ulonglong)]
    lib.CascGetFileSize64.restype = ctypes.c_bool
    lib.CascReadFile.argtypes = [H, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    lib.CascReadFile.restype = ctypes.c_bool
    lib.CascCloseFile.argtypes = [H]
    lib.CascFindFirstFile.argtypes = [H, ctypes.c_char_p, ctypes.POINTER(FindData), ctypes.c_char_p]
    lib.CascFindFirstFile.restype = H
    lib.CascFindNextFile.argtypes = [H, ctypes.POINTER(FindData)]
    lib.CascFindNextFile.restype = ctypes.c_bool
    lib.CascFindClose.argtypes = [H]
    lib.GetCascError.restype = ctypes.c_uint32
    return lib


class Storage:
    """A local CASC storage opened at one build (the active one, or `build_key`, a BuildConfig
    hash present in the install's .build.info / SC2Data/config)."""

    def __init__(self, install, build_key=None, lib=None, code_name=None):
        self.lib = lib or load_library()
        self.handle = ctypes.c_void_p()
        args = OpenArgs()
        args.Size = ctypes.sizeof(OpenArgs)
        args.szLocalPath = install.encode()
        args.szCodeName = code_name.encode() if code_name else None  # the Lutris install's .build.info has no Product column
        args.dwLocaleMask = CASC_LOCALE_ALL
        args.szBuildKey = build_key.encode() if build_key else None
        if not self.lib.CascOpenStorageEx(None, ctypes.byref(args), False, ctypes.byref(self.handle)):
            raise OSError(f"CascOpenStorageEx({install}, build {build_key}) failed: error {self.lib.GetCascError()}")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        if self.handle:
            self.lib.CascCloseStorage(self.handle)
            self.handle = ctypes.c_void_p()

    def find(self, mask="*", available_only=True):
        """File names matching a wildcard mask (case-insensitive in CascLib)."""
        fd = FindData()
        h = self.lib.CascFindFirstFile(self.handle, mask.encode(), ctypes.byref(fd), None)
        if not h:
            return []
        out = []
        try:
            while True:
                if not available_only or fd.bFileAvailable:
                    out.append(fd.szFileName.decode(errors="replace"))
                if not self.lib.CascFindNextFile(h, ctypes.byref(fd)):
                    break
        finally:
            self.lib.CascFindClose(h)
        return out

    def read(self, name):
        hf = ctypes.c_void_p()
        if not self.lib.CascOpenFile(self.handle, name.encode(), CASC_LOCALE_ALL, CASC_OPEN_BY_NAME, ctypes.byref(hf)):
            raise FileNotFoundError(f"{name}: CascOpenFile error {self.lib.GetCascError()}")
        try:
            size = ctypes.c_ulonglong()
            if not self.lib.CascGetFileSize64(hf, ctypes.byref(size)):
                raise OSError(f"{name}: CascGetFileSize64 error {self.lib.GetCascError()}")
            buf = ctypes.create_string_buffer(int(size.value))
            got = ctypes.c_uint32()
            if not self.lib.CascReadFile(hf, buf, int(size.value), ctypes.byref(got)) or got.value != size.value:
                raise OSError(f"{name}: CascReadFile read {got.value} of {size.value} bytes, error {self.lib.GetCascError()}")
            return buf.raw
        finally:
            self.lib.CascCloseFile(hf)
