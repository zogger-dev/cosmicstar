"""Extracts authentic StarCraft II button and UI icons from local CASC storage.

Converts Blizzard .dds textures to web-friendly .png images and builds a manifest
mapping canonical unit, structure, upgrade, and resource names to their icon files.

Usage:
    python scripts/extract_icons.py
    python scripts/extract_icons.py --out data/worldmodel/assets/icons
"""

import argparse
import io
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

from cosmicstar.envs import wine_run_config  # noqa: E402
from cosmicstar.envs.casc import Storage  # noqa: E402
from extract_gamedata import build_key  # noqa: E402

MODS = ["core", "liberty", "swarm", "void", "voidmulti"]

# Specific HUD & UI textures to always extract
HUD_TEXTURES = {
    "icon-mineral-nobg": "icon-mineral-nobg.dds",
    "icon-gas-terran-nobg": "icon-gas-terran-nobg.dds",
    "icon-gas-protoss-nobg": "icon-gas-protoss-nobg.dds",
    "icon-gas-zerg_nobg": "icon-gas-zerg_nobg.dds",
    "icon-supply-terran_nobg": "icon-supply-terran_nobg.dds",
    "icon-supply-protoss_nobg": "icon-supply-protoss_nobg.dds",
    "icon-supply-zerg_nobg": "icon-supply-zerg_nobg.dds",
    "ui_console_helperbutton_icon_worker_terr": "ui_console_helperbutton_icon_worker_terr.dds",
    "ui_console_helperbutton_icon_worker_prot": "ui_console_helperbutton_icon_worker_prot.dds",
    "ui_console_helperbutton_icon_worker_zerg": "ui_console_helperbutton_icon_worker_zerg.dds",
    "ui_console_helperbutton_icon_army_terr": "ui_console_helperbutton_icon_army_terr.dds",
    "ui_console_helperbutton_icon_army_prot": "ui_console_helperbutton_icon_army_prot.dds",
    "ui_console_helperbutton_icon_army_zerg": "ui_console_helperbutton_icon_army_zerg.dds",
    "icon-1v1-terran": "icon-1v1-terran.dds",
    "icon-1v1-protoss": "icon-1v1-protoss.dds",
    "icon-1v1-zerg": "icon-1v1-zerg.dds",
}

# Standard manual overrides / aliases to ensure 100% accurate match
KNOWN_UNIT_ALIASES = {
    "larva": "btn-unit-zerg-larvaex3.png",
    "egg": "btn-unit-zerg-egg.png",
    "drone": "btn-unit-zerg-drone.png",
    "droneburrowed": "btn-unit-zerg-drone.png",
    "overlord": "btn-unit-zerg-overlord.png",
    "overlordtransport": "btn-unit-zerg-overlordtransport.png",
    "overseer": "btn-unit-zerg-overseer.png",
    "overseersieged": "btn-unit-zerg-overseer.png",
    "zergling": "btn-unit-zerg-zergling.png",
    "zerglingburrowed": "btn-unit-zerg-zergling.png",
    "baneling": "btn-unit-zerg-baneling.png",
    "banelingburrowed": "btn-unit-zerg-baneling.png",
    "banelingcocoon": "btn-unit-zerg-baneling.png",
    "roach": "btn-unit-zerg-roach.png",
    "roachburrowed": "btn-unit-zerg-roach.png",
    "ravager": "btn-unit-zerg-ravager.png",
    "ravagerburrowed": "btn-unit-zerg-ravager.png",
    "ravagercocoon": "btn-unit-zerg-ravager.png",
    "hydralisk": "btn-unit-zerg-hydralisk.png",
    "hydraliskburrowed": "btn-unit-zerg-hydralisk.png",
    "lurkermp": "btn-unit-zerg-lurker.png",
    "lurker": "btn-unit-zerg-lurker.png",
    "lurkerburrowed": "btn-unit-zerg-lurker.png",
    "lurkerburrowedmp": "btn-unit-zerg-lurker.png",
    "lurkeregg": "btn-unit-zerg-lurker.png",
    "lurkermpcocoon": "btn-unit-zerg-lurker.png",
    "mutalisk": "btn-unit-zerg-mutalisk.png",
    "corruptor": "btn-unit-zerg-corruptor.png",
    "swarmhost": "btn-unit-zerg-swarmhost.png",
    "swarmhostmp": "btn-unit-zerg-swarmhost.png",
    "swarmhostburrowedmp": "btn-unit-zerg-swarmhost.png",
    "infestor": "btn-unit-zerg-infestor.png",
    "infestorburrowed": "btn-unit-zerg-infestor.png",
    "viper": "btn-unit-zerg-viper.png",
    "ultralisk": "btn-unit-zerg-ultralisk.png",
    "ultraliskburrowed": "btn-unit-zerg-ultralisk.png",
    "broodlord": "btn-unit-zerg-broodlord.png",
    "broodlordcocoon": "btn-unit-zerg-broodlord.png",
    "queen": "btn-unit-zerg-queen.png",
    "queenburrowed": "btn-unit-zerg-queen.png",
    "creeptumor": "btn-building-zerg-creeptumor.png",
    "creeptumorburrowed": "btn-building-zerg-creeptumor.png",
    "creeptumorqueen": "btn-building-zerg-creeptumor.png",
    "hatchery": "btn-building-zerg-hatchery.png",
    "lair": "btn-building-zerg-lair.png",
    "hive": "btn-building-zerg-hive.png",
    "spawningpool": "btn-building-zerg-spawningpool.png",
    "evolutionchamber": "btn-building-zerg-evolutionchamber.png",
    "hydraliskden": "btn-building-zerg-hydraliskden.png",
    "lurkerden": "btn-building-zerg-hydraliskden.png",
    "lurkerdenmp": "btn-building-zerg-hydraliskden.png",
    "spire": "btn-building-zerg-spire.png",
    "greaterspire": "btn-building-zerg-greaterspire.png",
    "ultraliskcavern": "btn-building-zerg-ultraliskcavern.png",
    "infestationpit": "btn-building-zerg-infestationpit.png",
    "nydusnetwork": "btn-building-zerg-nydusnetwork.png",
    "nyduscanal": "btn-building-zerg-nydusnetwork.png",
    "banelingnest": "btn-building-zerg-banelingnest.png",
    "roachwarren": "btn-building-zerg-roachwarren.png",
    "spinecrawler": "btn-building-zerg-spinecrawler.png",
    "spinecrawleruprooted": "btn-building-zerg-spinecrawler.png",
    "sporecrawler": "btn-building-zerg-sporecrawler.png",
    "sporecrawleruprooted": "btn-building-zerg-sporecrawler.png",
    "extractor": "btn-building-zerg-extractor.png",
    "extractorrich": "btn-building-zerg-extractor.png",

    # Terran
    "marine": "btn-unit-terran-marine.png",
    "marauder": "btn-unit-terran-marauder.png",
    "reaper": "btn-unit-terran-reaper.png",
    "ghost": "btn-unit-terran-ghost.png",
    "hellion": "btn-unit-terran-hellion.png",
    "hellbat": "btn-unit-terran-hellionbattlemode.png",
    "helliontank": "btn-unit-terran-hellionbattlemode.png",
    "siegetank": "btn-unit-terran-siegetank.png",
    "siegetanksieged": "btn-unit-terran-siegetank.png",
    "cyclone": "btn-unit-terran-cyclone.png",
    "widowmine": "btn-unit-terran-widowmine.png",
    "widowmineburrowed": "btn-unit-terran-widowmine.png",
    "thor": "btn-unit-terran-thor.png",
    "thorhighimpactmode": "btn-unit-terran-thor.png",
    "vikingfighter": "btn-unit-terran-vikingfighter.png",
    "vikingassault": "btn-unit-terran-vikingfighter.png",
    "medivac": "btn-unit-terran-medivac.png",
    "liberator": "btn-unit-terran-liberator.png",
    "liberatorag": "btn-unit-terran-liberator.png",
    "raven": "btn-unit-terran-raven.png",
    "banshee": "btn-unit-terran-banshee.png",
    "battlecruiser": "btn-unit-terran-battlecruiser.png",
    "scv": "btn-unit-terran-scv.png",
    "commandcenter": "btn-building-terran-commandcenter.png",
    "commandcenterflying": "btn-building-terran-commandcenter.png",
    "orbitalcommand": "btn-techupgrade-terran-orbitalcommand.png",
    "orbitalcommandflying": "btn-techupgrade-terran-orbitalcommand.png",
    "planetaryfortress": "btn-building-terran-planetaryfortress.png",
    "supplydepot": "btn-building-terran-supplydepot.png",
    "supplydepotlowered": "btn-building-terran-supplydepot.png",
    "refinery": "btn-building-terran-refinery.png",
    "refineryrich": "btn-building-terran-refinery.png",
    "barracks": "btn-building-terran-barracks.png",
    "barracksflying": "btn-building-terran-barracks.png",
    "engineeringbay": "btn-building-terran-engineeringbay.png",
    "bunker": "btn-building-terran-bunker.png",
    "sensortower": "btn-building-terran-sensordome.png",
    "missileturret": "btn-building-terran-missileturret.png",
    "factory": "btn-building-terran-factory.png",
    "factoryflying": "btn-building-terran-factory.png",
    "ghostacademy": "btn-building-terran-ghostacademy.png",
    "starport": "btn-building-terran-starport.png",
    "starportflying": "btn-building-terran-starport.png",
    "armory": "btn-building-terran-armory.png",
    "fusioncore": "btn-building-terran-fusioncore.png",
    "techlab": "btn-building-terran-techlab.png",
    "barrackstechlab": "btn-building-terran-techlab.png",
    "factorytechlab": "btn-building-terran-techlab.png",
    "starporttechlab": "btn-building-terran-techlab.png",
    "reactor": "btn-building-terran-reactor.png",
    "barracksreactor": "btn-building-terran-reactor.png",
    "factoryreactor": "btn-building-terran-reactor.png",
    "starportreactor": "btn-building-terran-reactor.png",

    # Protoss
    "probe": "btn-unit-protoss-probe.png",
    "zealot": "btn-unit-protoss-zealot.png",
    "stalker": "btn-unit-protoss-stalker.png",
    "sentry": "btn-unit-protoss-sentry.png",
    "adept": "btn-unit-protoss-adept.png",
    "hightemplar": "btn-unit-protoss-hightemplar.png",
    "darktemplar": "btn-unit-protoss-darktemplar.png",
    "archon": "btn-unit-protoss-archon.png",
    "immortal": "btn-unit-protoss-immortal.png",
    "colossus": "btn-unit-protoss-colossus.png",
    "disruptor": "btn-unit-protoss-disruptor.png",
    "observer": "btn-unit-protoss-observer.png",
    "observersieged": "btn-unit-protoss-observer.png",
    "warpprism": "btn-unit-protoss-warpprism.png",
    "warpprismphasing": "btn-unit-protoss-warpprism.png",
    "phoenix": "btn-unit-protoss-phoenix.png",
    "voidray": "btn-unit-collection-voidray-golden.png",
    "oracle": "btn-unit-protoss-oracle.png",
    "carrier": "btn-unit-protoss-carrier.png",
    "interceptor": "btn-unit-protoss-interceptor.png",
    "tempest": "btn-unit-protoss-tempest.png",
    "mothership": "btn-unit-protoss-mothership.png",
    "mothershipcore": "btn-unit-protoss-mothershipcore.png",
    "nexus": "btn-building-protoss-nexus.png",
    "pylon": "btn-building-protoss-pylon.png",
    "assimilator": "btn-building-protoss-assimilator.png",
    "assimilatorrich": "btn-building-protoss-assimilator.png",
    "gateway": "btn-building-protoss-gateway.png",
    "warpgate": "btn-building-protoss-warpgate.png",
    "forge": "btn-building-protoss-forge.png",
    "cyberneticscore": "btn-building-protoss-cyberneticscore.png",
    "photoncannon": "btn-building-protoss-photoncannon.png",
    "shieldbattery": "btn-building-protoss-shieldbattery.png",
    "twilightcouncil": "btn-building-protoss-twilightcouncil.png",
    "roboticsfacility": "btn-building-protoss-roboticsfacility.png",
    "stargate": "btn-building-protoss-stargate.png",
    "roboticsbay": "btn-building-protoss-roboticssupportbay.png",
    "fleetbeacon": "btn-building-protoss-fleetbeacon.png",
    "templararchive": "btn-building-protoss-templararchives.png",
    "templararchives": "btn-building-protoss-templararchives.png",
    "darkshrine": "btn-building-protoss-darkshrine.png",
    "oraclestasistrap": "btn-unit-protoss-stasisward.png",
    "stasistrap": "btn-unit-protoss-stasisward.png",
    "stasisward": "btn-unit-protoss-stasisward.png",
}

KNOWN_UPGRADE_ALIASES = {
    # Terran
    "stimpack": "btn-ability-terran-stimpack-color.png",
    "shieldwall": "btn-techupgrade-terran-combatshield-color.png",
    "combatshield": "btn-techupgrade-terran-combatshield-color.png",
    "punishergrenades": "btn-ability-terran-punishergrenade-color.png",
    "concussiveshells": "btn-ability-terran-punishergrenade-color.png",
    "hisecautotracking": "btn-upgrade-terran-hisecautotracking.png",
    "terranbuildingarmor": "btn-upgrade-terran-buildingarmor.png",
    "drillclaws": "btn-upgrade-terran-researchdrillingclaws.png",
    "smartservos": "btn-upgrade-terran-transformationconverters.png",
    "bansheecloak": "btn-ability-terran-cloak-color.png",
    "personalcloaking": "btn-ability-terran-cloak-color.png",
    "cloak": "btn-ability-terran-cloak-color.png",
    "liberatorrange": "btn-upgrade-terran-advanceballistics.png",
    "advancedballistics": "btn-upgrade-terran-advanceballistics.png",
    "terraninfantryweapons": "btn-upgrade-terran-infantryweaponslevel1.png",
    "terraninfantryweaponslevel1": "btn-upgrade-terran-infantryweaponslevel1.png",
    "terraninfantryweaponslevel2": "btn-upgrade-terran-infantryweaponslevel2.png",
    "terraninfantryweaponslevel3": "btn-upgrade-terran-infantryweaponslevel3.png",
    "terraninfantryarmors": "btn-upgrade-terran-infantryarmorlevel1.png",
    "terraninfantryarmorslevel1": "btn-upgrade-terran-infantryarmorlevel1.png",
    "terraninfantryarmorslevel2": "btn-upgrade-terran-infantryarmorlevel2.png",
    "terraninfantryarmorslevel3": "btn-upgrade-terran-infantryarmorlevel3.png",
    "terranvehicleweapons": "btn-upgrade-terran-vehicleweaponslevel1.png",
    "terranvehicleweaponslevel1": "btn-upgrade-terran-vehicleweaponslevel1.png",
    "terranvehicleweaponslevel2": "btn-upgrade-terran-vehicleweaponslevel2.png",
    "terranvehicleweaponslevel3": "btn-upgrade-terran-vehicleweaponslevel3.png",
    "terranvehicleandshiparmorslevel1": "btn-upgrade-terran-vehicleplatinglevel1.png",
    "terranvehicleandshiparmorslevel2": "btn-upgrade-terran-vehicleplatinglevel2.png",
    "terranvehicleandshiparmorslevel3": "btn-upgrade-terran-vehicleplatinglevel3.png",
    "terranshipweaponslevel1": "btn-upgrade-terran-shipweaponslevel1.png",
    "terranshipweaponslevel2": "btn-upgrade-terran-shipweaponslevel2.png",
    "terranshipweaponslevel3": "btn-upgrade-terran-shipweaponslevel3.png",

    # Zerg
    "zerglingmovementspeed": "btn-upgrade-zerg-metabolicboost.png",
    "metabolicboost": "btn-upgrade-zerg-metabolicboost.png",
    "zerglingattackspeed": "btn-upgrade-zerg-adrenalglands.png",
    "adrenalglands": "btn-upgrade-zerg-adrenalglands.png",
    "centrificalhooks": "btn-upgrade-zerg-centrifugalhooks.png",
    "centrifugalhooks": "btn-upgrade-zerg-centrifugalhooks.png",
    "glialreconstitution": "btn-upgrade-zerg-glialreconstitution.png",
    "tunnelingclaws": "btn-upgrade-zerg-tunnelingclaws.png",
    "evolvemuscularaugments": "btn-upgrade-zerg-evolvemuscularaugments.png",
    "muscularaugments": "btn-upgrade-zerg-evolvemuscularaugments.png",
    "evolvegroovedspines": "btn-upgrade-zerg-hotsgroovedspines.png",
    "groovedspines": "btn-upgrade-zerg-hotsgroovedspines.png",
    "anabolicsynthesis": "btn-upgrade-zerg-anabolicsynthesis.png",
    "chitinousplating": "btn-upgrade-zerg-chitinousplating.png",
    "burrow": "btn-ability-zerg-burrow-color.png",
    "overlordspeed": "btn-upgrade-zerg-pneumatizedcarapace.png",
    "pneumatizedcarapace": "btn-upgrade-zerg-pneumatizedcarapace.png",
    "zergmeleeweapons": "btn-upgrade-zerg-meleeattacks-level1.png",
    "zergmeleeweaponslevel1": "btn-upgrade-zerg-meleeattacks-level1.png",
    "zergmeleeweaponslevel2": "btn-upgrade-zerg-meleeattacks-level2.png",
    "zergmeleeweaponslevel3": "btn-upgrade-zerg-meleeattacks-level3.png",
    "zergmissileweapons": "btn-upgrade-zerg-missileattacks-level1.png",
    "zergmissileweaponslevel1": "btn-upgrade-zerg-missileattacks-level1.png",
    "zergmissileweaponslevel2": "btn-upgrade-zerg-missileattacks-level2.png",
    "zergmissileweaponslevel3": "btn-upgrade-zerg-missileattacks-level3.png",
    "zerggroundarmors": "btn-upgrade-zerg-groundcarapace-level1.png",
    "zerggroundarmorslevel1": "btn-upgrade-zerg-groundcarapace-level1.png",
    "zerggroundarmorslevel2": "btn-upgrade-zerg-groundcarapace-level2.png",
    "zerggroundarmorslevel3": "btn-upgrade-zerg-groundcarapace-level3.png",
    "zergflyerweaponslevel1": "btn-upgrade-zerg-airattacks-level1.png",
    "zergflyerweaponslevel2": "btn-upgrade-zerg-airattacks-level2.png",
    "zergflyerweaponslevel3": "btn-upgrade-zerg-airattacks-level3.png",
    "zergflyerarmorslevel1": "btn-upgrade-zerg-flyercarapace-level1.png",
    "zergflyerarmorslevel2": "btn-upgrade-zerg-flyercarapace-level2.png",
    "zergflyerarmorslevel3": "btn-upgrade-zerg-flyercarapace-level3.png",

    # Protoss
    "warpgateresearch": "btn-building-protoss-warpgate.png",
    "warpgate": "btn-building-protoss-warpgate.png",
    "charge": "btn-ability-protoss-charge-color.png",
    "blinktech": "btn-ability-protoss-blink-color.png",
    "blink": "btn-ability-protoss-blink-color.png",
    "adeptpiercingattack": "btn-upgrade-protoss-resonatingglaives-orange.png",
    "adeptpiercingupgrade": "btn-upgrade-protoss-resonatingglaives-orange.png",
    "resonatingglaives": "btn-upgrade-protoss-resonatingglaives-orange.png",
    "psistormtech": "btn-ability-protoss-psistorm-color.png",
    "psistorm": "btn-ability-protoss-psistorm-color.png",
    "storm": "btn-ability-protoss-psistorm-color.png",
    "darktemplarblinkupgrade": "btn-ability-protoss-stealth-blink.png",
    "shadowstride": "btn-ability-protoss-stealth-blink.png",
    "extendedthermallance": "btn-upgrade-protoss-extendedthermallance.png",
    "graviticdrive": "btn-upgrade-protoss-graviticdrive.png",
    "graviticbooster": "btn-upgrade-protoss-graviticbooster.png",
    "protossgroundweapons": "btn-upgrade-protoss-groundweaponslevel1.png",
    "protossgroundweaponslevel1": "btn-upgrade-protoss-groundweaponslevel1.png",
    "protossgroundweaponslevel2": "btn-upgrade-protoss-groundweaponslevel2.png",
    "protossgroundweaponslevel3": "btn-upgrade-protoss-groundweaponslevel3.png",
    "protossgroundarmors": "btn-upgrade-protoss-groundarmorlevel1.png",
    "protossgroundarmorslevel1": "btn-upgrade-protoss-groundarmorlevel1.png",
    "protossgroundarmorslevel2": "btn-upgrade-protoss-groundarmorlevel2.png",
    "protossgroundarmorslevel3": "btn-upgrade-protoss-groundarmorlevel3.png",
    "protossshields": "btn-upgrade-protoss-shieldslevel1.png",
    "protossshieldslevel1": "btn-upgrade-protoss-shieldslevel1.png",
    "protossshieldslevel2": "btn-upgrade-protoss-shieldslevel2.png",
    "protossshieldslevel3": "btn-upgrade-protoss-shieldslevel3.png",
    "protossairweaponslevel1": "btn-upgrade-protoss-airweaponslevel1.png",
    "protossairweaponslevel2": "btn-upgrade-protoss-airweaponslevel2.png",
    "protossairweaponslevel3": "btn-upgrade-protoss-airweaponslevel3.png",
    "protossairarmorslevel1": "btn-upgrade-protoss-airarmorlevel1.png",
    "protossairarmorslevel2": "btn-upgrade-protoss-airarmorlevel2.png",
    "protossairarmorslevel3": "btn-upgrade-protoss-airarmorlevel3.png",
}

HUD_ALIASES = {
    "minerals": "icon-mineral-nobg.png",
    "vespene": "icon-gas-terran-nobg.png",
    "vespene_terran": "icon-gas-terran-nobg.png",
    "vespene_protoss": "icon-gas-protoss-nobg.png",
    "vespene_zerg": "icon-gas-zerg_nobg.png",
    "supply_terran": "icon-supply-terran_nobg.png",
    "supply_protoss": "icon-supply-protoss_nobg.png",
    "supply_zerg": "icon-supply-zerg_nobg.png",
    "worker_terran": "ui_console_helperbutton_icon_worker_terr.png",
    "worker_protoss": "ui_console_helperbutton_icon_worker_prot.png",
    "worker_zerg": "ui_console_helperbutton_icon_worker_zerg.png",
    "army_terran": "ui_console_helperbutton_icon_army_terr.png",
    "army_protoss": "ui_console_helperbutton_icon_army_prot.png",
    "army_zerg": "ui_console_helperbutton_icon_army_zerg.png",
    "race_terran": "icon-1v1-terran.png",
    "race_protoss": "icon-1v1-protoss.png",
    "race_zerg": "icon-1v1-zerg.png",
}


def _clean_basename(p: str) -> str:
    return p.replace("\\", "/").split("/")[-1].lower()


def extract_icons(build: int = 98310, out_dir: str = "data/worldmodel/assets/icons"):
    os.makedirs(out_dir, exist_ok=True)
    install = wine_run_config.install_for_build(build)
    key = build_key(install, build)
    if not key:
        print(f"Error: Build key not found for build {build}")
        return

    print(f"Opening CASC storage at {install} (build {build})...", flush=True)
    t0 = time.time()
    with Storage(install, build_key=key) as st:
        print(f"Connected in {time.time()-t0:.2f}s", flush=True)

        # 1. Discover all candidate texture paths
        all_btns = st.find("*btn-*.dds")
        ladder_btns = [
            b for b in all_btns
            if any(f"mods\\{m}.sc2mod" in b.lower() for m in MODS)
        ]
        print(f"Found {len(ladder_btns)} button textures in ladder mods", flush=True)

        # Map base lowercased name -> full CASC path
        casc_map: dict[str, str] = {}
        for b in ladder_btns:
            base = _clean_basename(b)
            # If duplicate, prefer earlier in priority list
            if base not in casc_map:
                casc_map[base] = b

        # 2. Add HUD textures
        for target_name, target_file in HUD_TEXTURES.items():
            matches = st.find("*" + target_file)
            if matches:
                # Prefer core/liberty
                best = matches[0]
                for m in matches:
                    if "core.sc2mod" in m.lower() or "liberty.sc2mod" in m.lower():
                        best = m
                        break
                casc_map[_clean_basename(target_file)] = best

        # 3. Parse ButtonData and AbilData to build automated mappings
        btn_to_icon: dict[str, str] = {}
        upgrade_to_btn: dict[str, str] = {}

        for m in MODS:
            # ButtonData
            bp = f"mods\\{m}.sc2mod\\base.sc2data\\gamedata\\buttondata.xml"
            try:
                raw_b = st.read(bp)
                root_b = ET.fromstring(raw_b)
                for el in root_b:
                    b_id = el.get("id")
                    icon = el.find("Icon")
                    if b_id and icon is not None and icon.get("value"):
                        tex = _clean_basename(icon.get("value"))
                        btn_to_icon[b_id.lower()] = tex
            except Exception:
                pass

            # AbilData
            ap = f"mods\\{m}.sc2mod\\base.sc2data\\gamedata\\abildata.xml"
            try:
                raw_a = st.read(ap)
                root_a = ET.fromstring(raw_a)
                for el in root_a:
                    for c in el:
                        up = c.attrib.get("Upgrade")
                        btn = c.find("Button")
                        if up and btn is not None:
                            face = btn.attrib.get("DefaultButtonFace")
                            if face:
                                upgrade_to_btn[up.lower()] = face.lower()
            except Exception:
                pass

        # 4. Extract and convert textures to PNG
        print(f"Extracting {len(casc_map)} textures to {out_dir}...", flush=True)
        extracted_files = set()
        t1 = time.time()
        count = 0
        for base_dds, casc_path in casc_map.items():
            out_png = base_dds.replace(".dds", ".png")
            dst = os.path.join(out_dir, out_png)
            if not os.path.exists(dst):
                try:
                    raw = st.read(casc_path)
                    im = Image.open(io.BytesIO(raw))
                    im.save(dst, "PNG", optimize=True)
                    count += 1
                except Exception as e:
                    # Skip corrupt / non-standard DDS textures
                    continue
            extracted_files.add(out_png)

        print(
            f"Extracted {count} new textures ({len(extracted_files)} total) in "
            f"{time.time() - t1:.2f}s",
            flush=True,
        )

        # 5. Build unit and upgrade manifest
        unit_map: dict[str, str] = dict(KNOWN_UNIT_ALIASES)
        upgrade_map: dict[str, str] = dict(KNOWN_UPGRADE_ALIASES)

        # Merge automated ButtonData / AbilData mappings
        for up_name, btn_face in upgrade_to_btn.items():
            icon_file = btn_to_icon.get(btn_face)
            if icon_file:
                png_name = icon_file.replace(".dds", ".png")
                if png_name in extracted_files:
                    upgrade_map[up_name] = png_name

        for btn_id, icon_file in btn_to_icon.items():
            png_name = icon_file.replace(".dds", ".png")
            if png_name in extracted_files:
                if btn_id not in unit_map:
                    unit_map[btn_id] = png_name

        manifest = {
            "version": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hud": HUD_ALIASES,
            "units": unit_map,
            "upgrades": upgrade_map,
            "total_files": len(extracted_files),
        }

        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Wrote manifest with {len(unit_map)} units and {len(upgrade_map)} upgrades -> {manifest_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=int, default=98310)
    ap.add_argument("--out", default="data/worldmodel/assets/icons")
    args = ap.parse_args()

    extract_icons(build=args.build, out_dir=args.out)


if __name__ == "__main__":
    main()
