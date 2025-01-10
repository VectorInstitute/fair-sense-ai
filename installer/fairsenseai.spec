# -*- mode: python ; coding: utf-8 -*-
import sys
from os import path
site_packages = next(p for p in sys.path if 'site-packages' in p)

a = Analysis(
    ['../fairsenseai.py'],
    pathex=[],
    binaries=[],
    datas=[
        (path.join(site_packages,'gradio'),'gradio'),
        (path.join(site_packages,'gradio_client'),'gradio_client'),
    ],
    hiddenimports=['gradio', 'gradio_client'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='fairsenseai',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='fairsenseai',
)
