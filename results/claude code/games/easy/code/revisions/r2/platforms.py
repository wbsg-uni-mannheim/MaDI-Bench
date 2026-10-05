"""Platform alias map: lowercase-alnum key -> canonical (taxonomy name where one clearly applies).
PC family kept as 'PC' (target schema example); DOS/other legacy computers kept distinct."""
import re
def pkey(x): return re.sub(r'[^a-z0-9]', '', str(x).lower())
_A = {
 "PlayStation": "playstation sonyplaystation ps ps1 psx",
 "PlayStation 2": "ps2 playstation2",
 "PlayStation 3": "ps3 playstation3",
 "PlayStation 4": "ps4 playstation4",
 "PlayStation 5": "ps5 playstation5",
 "PlayStation Portable (PSP)": "playstationportable psp playstationpotable",
 "PlayStation Vita": "playstationvita psvita playstationvitar",
 "PlayStation VR": "playstationvr",
 "PlayStation VR2": "playstationvr2",
 "Xbox": "xbox",
 "Xbox 360": "xbox360 x360 xbox360console",
 "Xbox One": "xboxone xone",
 "Xbox Series X": "xboxseriesx xboxseriesxs xboxseriesxandseriess xboxseries",
 "Nintendo Switch": "switch nintendoswitch ns",
 "Nintendo Switch 2": "nintendoswitch2",
 "Wii": "wii",
 "Wii U": "wiiu",
 "Nintendo DS": "ds nintendods",
 "Nintendo DSi": "dsi",
 "Nintendo 3DS": "3ds nintendo3ds 3dsfamily",
 "New Nintendo 3DS": "new3ds",
 "Nintendo 2DS": "nintendo2ds",
 "Nintendo GameCube": "gamecube gc nintendogamecube",
 "Nintendo 64": "nintendo64 n64",
 "Game Boy": "gameboy gb gameboyline",
 "Game Boy Color": "gameboycolor gbc",
 "Game Boy Advance": "gameboyadvance gba",
 "Nintendo Entertainment System (NES)": "nes nintendoentertainmentsystem snintendoentertainmentsystem familycomputer famicom famicomnintendoentertainmentsystem",
 "Super Nintendo (SNES)": "supernintendo snes superfamicom supernintendoentertainmentsystem",
 "Sega Dreamcast": "dreamcast dc segadreamcast",
 "Sega Saturn": "segasaturn saturn",
 "Sega Genesis / Mega Drive": "segagenesis genesis megadrive segamegadrive megadrivegenesis segamegadrivegenesis segagenesismegadrive segamegadrivesegagenesis",
 "Sega Master System": "mastersystem segamastersystem segamarkiii",
 "Sega Game Gear": "gamegear segagamegear",
 "Atari 2600": "atari2600", "Atari 5200": "atari5200", "Atari 7800": "atari7800",
 "Atari Jaguar": "atarijaguar",
 "PC": "pc microsoftwindows windows personalcomputer windowspc ibmpc ibmpccompatible ibmpersonalcomputer windows95 windows98 windowsxp microsoftwindowsxp microsoftwindows95 microsoftwindows98 windows2000 microsoftwindows2000 windowsvista windows7 windows8 windows81 windows10 microsoftwindows7 microsoftwindowsvista windowsme windows9x steamservice",
 "macOS": "macos osx macosx mac macintosh applemacintosh applemac classicmacos",
 "Linux / SteamOS": "linux steamos",
 "iOS (iPhone)": "iosapple iphone iphoneos appleios appleiphone",
 "iPadOS (iPad)": "ipados ipad ipad2",
 "Android": "androidoperatingsystem androidos",
 "Google Stadia": "googlestadia stadia",
 "Amazon Luna": "amazonluna",
 "Arcade Cabinet": "arcadecabinet arcadegame arcadevideogame arcadegames videoarcade amusementarcade videogamearcadecabinet",
 "Neo Geo MVS": "neogeomvs",
 "Oculus Quest": "oculusquest", "Oculus Quest 2": "oculusquest2", "Oculus Rift": "oculusrift", "Oculus Rift S": "oculusrifts",
 "Valve Index": "valveindex", "HTC Vive": "htcvive",
}
# Values that appear verbatim in the non-dbpedia sources in the taxonomy-like form and are
# ambiguous with respect to the taxonomy: keep as the source spells them.
_KEEP = {"ios": "iOS", "mobile": "Mobile", "arcade": "Arcade", "android": "Android"}
ALIAS = {}
for canon, keys in _A.items():
    for k in keys.split(): ALIAS[k] = canon
ALIAS.update(_KEEP)
def canon_platform(raw, fallback=None):
    if raw is None or (isinstance(raw, float)) or str(raw).strip() == "": return None
    k = pkey(raw)
    if k in ALIAS: return ALIAS[k]
    return fallback.get(k, re.sub(r'\s+', ' ', str(raw)).strip()) if fallback else re.sub(r'\s+', ' ', str(raw)).strip()
