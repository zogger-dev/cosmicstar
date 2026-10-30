"""pysc2 run config for the Windows StarCraft II client under Wine/Proton on Linux.

Blizzard's headless Linux builds stop at 4.10 and every 5.0.x client is Windows only, so on
Linux the client runs inside a Wine prefix (a Lutris / Battle.net-under-Proton install).
pysc2's own Linux config expects a native binary; this one launches
`<launcher> Versions/Base<build>/SC2_x64.exe -listen ... -displayMode 0` from the install's
Support64 folder, with -dataDir / -tempDir rewritten to Windows paths the client can read.

Importing this module registers the config: pysc2's run_configs.get() picks it on Linux
whenever an install is found (priority 2, above the native-Linux config). Configuration is
by environment variable, all optional:

  SC2PATH      the install, e.g. ~/Games/battlenet/drive_c/Program Files (x86)/StarCraft II
               (default: the first of DEFAULT_INSTALLS that exists)
  WINEPREFIX   the prefix (default: the directory above the install's drive_c)
  SC2_WINE     the launcher: umu-run (Lutris' Proton launcher, the default when it is
               installed) or a wine binary
  PROTONPATH   the Proton build umu-run should use (default: the one Lutris recorded in
               <prefix>/config_info, else umu-run's own default)

The client has no headless mode (-displayMode 0 is windowed) and needs a display: DISPLAY
defaults to :0; point it at an Xvfb server for unattended runs.

Measured (RTX 5090 box, GE-Proton 11-11, 2026-10-23): API port open 6 s after launch, a
5.0.17 replay played as observer with fog off at ~1500 game loops/s.
"""
import glob
import os
import platform
import subprocess

from pysc2.lib import sc_process
from pysc2.run_configs import platforms

DEFAULT_INSTALLS = [
    "~/Games/battlenet/drive_c/Program Files (x86)/StarCraft II",
    "~/Games/starcraft-ii/drive_c/Program Files (x86)/StarCraft II",
    "~/.wine/drive_c/Program Files (x86)/StarCraft II",
]
# Older builds live in a side copy of the install (cosmicstar/envs/tact.py; worldmodel/README.md
# "Older builds from the CDN"): the engine for a base build comes from the first install that
# has its Versions/Base<build>, the real install first. SC2PATH overrides everything.
SIDE_INSTALLS = ["~/Games/battlenet/drive_c/Program Files (x86)/StarCraft II old"]
UMU_RUN = "~/.local/share/lutris/runtime/umu/umu-run"
PROTON_ROOTS = ["~/.local/share/Steam/compatibilitytools.d", "~/.steam/root/compatibilitytools.d"]


def sc2_install_dir():
    """The StarCraft II install: $SC2PATH, else the platform default (Linux: the first of
    DEFAULT_INSTALLS with a Versions folder). None when nothing is found."""
    p = os.environ.get("SC2PATH")
    if p:
        return os.path.expanduser(p)
    if platform.system() == "Windows":
        return r"C:\Program Files (x86)\StarCraft II"
    for cand in DEFAULT_INSTALLS:
        cand = os.path.expanduser(cand)
        if os.path.isdir(os.path.join(cand, "Versions")):
            return cand
    return None


def installs():
    """Every install to look in, the real one first: just $SC2PATH when it is set, else the
    default install and the SIDE_INSTALLS that exist."""
    if os.environ.get("SC2PATH"):
        return [os.path.expanduser(os.environ["SC2PATH"])]
    out = [d for d in [sc2_install_dir()] if d]
    for cand in SIDE_INSTALLS:
        cand = os.path.expanduser(cand)
        if os.path.isdir(os.path.join(cand, "Versions")) and cand not in out:
            out.append(cand)
    return out


def install_for_build(build):
    """The install whose Versions/Base<build>/SC2_x64.exe exists (the real one first), else the
    default install."""
    for inst in installs():
        if os.path.exists(os.path.join(inst, "Versions", "Base%05d" % build, "SC2_x64.exe")):
            return inst
    return sc2_install_dir()


def installed_builds():
    """Base builds with an SC2_x64.exe in any install."""
    out = set()
    for inst in installs():
        for d in glob.glob(os.path.join(inst, "Versions", "Base*")):
            if os.path.isfile(os.path.join(d, "SC2_x64.exe")):
                out.add(int(os.path.basename(d)[4:]))
    return out


def wine_prefix(install=None):
    """The Wine prefix holding the install: $WINEPREFIX, else the directory above drive_c."""
    p = os.environ.get("WINEPREFIX")
    if p:
        return os.path.expanduser(p)
    install = install or sc2_install_dir()
    if install and "/drive_c/" in install:
        return install.split("/drive_c/")[0]
    return None


def battlenet_cache_dirs(install=None):
    """Where the client caches maps and mods (*.s2ma), on either platform."""
    install = install or sc2_install_dir()
    if platform.system() == "Windows":
        roots = [r"C:\ProgramData\Blizzard Entertainment\Battle.net\Cache"]
    else:
        pfx = wine_prefix(install)
        roots = [os.path.join(pfx, "drive_c/ProgramData/Blizzard Entertainment/Battle.net/Cache")] if pfx else []
    if install:
        roots.append(os.path.join(install, "Battle.net", "Cache"))
    return roots


def windows_path(path, prefix):
    """A Unix path as the Wine client sees it: C:\\ inside the prefix's drive_c, Z:\\ (Wine's
    view of /) elsewhere. A trailing separator is kept (pysc2 passes -dataDir and -tempDir
    with one)."""
    trailing = path.endswith("/")
    path = os.path.abspath(path) + ("/" if trailing else "")
    drive_c = os.path.join(prefix, "drive_c")
    if path == drive_c or path.startswith(drive_c + "/"):
        win = "C:" + path[len(drive_c):]
    else:
        win = "Z:" + path
    return win.replace("/", "\\")


def launcher():
    """The command that runs a Windows exe in the prefix: $SC2_WINE, else Lutris' umu-run when
    it is installed, else wine."""
    w = os.environ.get("SC2_WINE") or (UMU_RUN if os.path.exists(os.path.expanduser(UMU_RUN)) else "wine")
    return os.path.expanduser(w)


def proton_path(prefix):
    """The Proton build for umu-run: $PROTONPATH, else the one named on the first line of the
    prefix's config_info (written by Lutris), looked up under Steam's compatibilitytools.d."""
    p = os.environ.get("PROTONPATH")
    if p:
        return p
    info = os.path.join(prefix, "config_info")
    if os.path.exists(info):
        name = open(info).readline().strip()
        for root in PROTON_ROOTS:
            for cand in sorted(glob.glob(os.path.join(os.path.expanduser(root), name + "*"))):
                return cand
    return None


def documents_dir(prefix=None):
    """The Windows user's Documents folder in the prefix (where the game and Editor write
    "StarCraft II/..." logs and replays); the real ~/Documents on Windows."""
    if platform.system() == "Windows":
        return os.path.expanduser("~/Documents")
    prefix = prefix or wine_prefix()
    return os.path.join(prefix, "drive_c/users", os.environ.get("USER", "steamuser"), "Documents")


def wine_env(prefix, launcher_path=None):
    """Environment for running a Windows exe in the prefix with launcher(): WINEPREFIX, a
    display, and for umu-run its game id, store and Proton build (never a runtime update)."""
    launcher_path = launcher_path or launcher()
    env = dict(os.environ, WINEPREFIX=prefix)
    env.setdefault("DISPLAY", ":0")
    if os.path.basename(launcher_path).startswith("umu"):
        env.setdefault("GAMEID", "umu-default")
        env.setdefault("STORE", "none")
        env.setdefault("UMU_RUNTIME_UPDATE", "0")  # never block a launch on a runtime download
        # Proton's default verb, waitforexitandrun, first waits for every Wine process in the
        # prefix to exit (Steam's "one instance" rule): a second engine, or the Battle.net app,
        # would block the launch forever. "run" just runs.
        env.setdefault("PROTON_VERB", "run")
        pp = proton_path(prefix)
        if pp:
            env.setdefault("PROTONPATH", pp)
    return env


def wine_command(exe, args=(), prefix=None):
    """(argv, env) to run a Windows exe from the install under Wine/Proton, e.g. the Editor:
    wine_command(f"{install}/Support64/SC2Editor_x64.exe", ["-exportreplay", windows_path(...)]).
    Path arguments must already be Windows paths (windows_path())."""
    prefix = prefix or wine_prefix()
    lp = launcher()
    return [lp, exe, *args], wine_env(prefix, lp)


class WineProcess(sc_process.StarcraftProcess):
    """pysc2's process wrapper with the launch step done through the Wine launcher."""

    def _launch(self, run_config, args, **kwargs):
        del kwargs
        args = list(args)
        for flag in ("-dataDir", "-tempDir"):
            if flag in args:
                i = args.index(flag) + 1
                args[i] = windows_path(args[i], run_config.prefix)
        if "-displayMode" not in args:  # pysc2 leaves these out on Linux (its Linux build is headless)
            args += ["-displayMode", "0", "-windowwidth", "640", "-windowheight", "480", "-windowx", "0", "-windowy", "0"]
        cmd = [run_config.launcher] + args
        try:
            return subprocess.Popen(cmd, cwd=run_config.cwd, env=run_config.env)
        except OSError as e:
            raise sc_process.SC2LaunchError(f"Failed to launch: {cmd}") from e


class WineLinux(platforms.LocalBase):
    """The Windows client under Wine/Proton on Linux (module doc)."""

    def __init__(self, version=None):
        build = getattr(version, "build_version", None)
        install = install_for_build(build) if build else sc2_install_dir()
        if not install:
            raise sc_process.SC2LaunchError(
                "No StarCraft II install found; set SC2PATH to the 'StarCraft II' folder inside the Wine prefix.")
        self.prefix = wine_prefix(install)
        if not self.prefix:
            raise sc_process.SC2LaunchError(f"{install} is not inside a Wine prefix (no drive_c); set WINEPREFIX.")
        self.launcher = launcher()
        super().__init__(install, "SC2_x64.exe", version=version, cwd="Support64",
                         env=wine_env(self.prefix, self.launcher))

    @classmethod
    def priority(cls):
        if platform.system() == "Linux" and sc2_install_dir():
            return 2  # above pysc2's native-Linux config (1)

    def start(self, want_rgb=True, **kwargs):
        del want_rgb  # the Windows client always renders; nothing to choose
        exec_path = os.path.join(self.data_dir, "Versions/Base%05d" % self.version.build_version, self._exec_name)
        if not os.path.exists(exec_path):
            raise sc_process.SC2LaunchError("No SC2 binary found at: %s" % exec_path)
        return WineProcess(self, exec_path=exec_path, version=self.version, **kwargs)
