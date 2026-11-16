// ============================================================================
// MATCH HUD, PRODUCTION QUEUE & FORCES BREAKDOWN
// ============================================================================
function formatRace(rawRace, pKey) {
  if (!rawRace) {
    if (FRAME && FRAME.forces && FRAME.forces[pKey]) {
      const uNames = (FRAME.forces[pKey].army || []).map(x => x.name).concat((FRAME.forces[pKey].structures || []).map(x => x.name)).join(" ");
      if (/Drone|Zergling|Hatchery|Overlord|Roach/i.test(uNames)) return { code: "Z", name: "Zerg" };
      if (/Probe|Zealot|Nexus|Pylon|Stalker/i.test(uNames)) return { code: "P", name: "Protoss" };
      if (/SCV|Marine|CommandCenter|Barracks/i.test(uNames)) return { code: "T", name: "Terran" };
    }
    return { code: "T", name: "Terran" };
  }
  const s = String(rawRace).toLowerCase();
  if (s.includes("테란") || s.includes("terran")) return { code: "T", name: "Terran" };
  if (s.includes("저그") || s.includes("zerg")) return { code: "Z", name: "Zerg" };
  if (s.includes("프로토스") || s.includes("protoss")) return { code: "P", name: "Protoss" };
  return { code: s[0].toUpperCase(), name: rawRace };
}

let ICON_MANIFEST = null;
fetch("/assets/icons/manifest.json")
  .then(r => r.json())
  .then(m => {
    ICON_MANIFEST = m;
    if (FRAME && FRAME.hud) renderHUDAndForces();
  })
  .catch(e => console.warn("Icon manifest load skipped:", e));

function getUnitOrUpgradeIcon(rawName, category, raceHint) {
  if (!rawName) return "";
  const name = String(rawName).trim();
  const n = name.toLowerCase().replace(/[\s\-_]/g, "");

  const KNOWN_DIRECT_ALIASES = {
    mule: "btn-unit-terran-mule.png",
    calldownmule: "btn-unit-terran-mule.png",
    oraclestasistrap: "btn-unit-protoss-stasisward.png",
    stasistrap: "btn-unit-protoss-stasisward.png",
    stasisward: "btn-unit-protoss-stasisward.png",
  };

  let assetFile = KNOWN_DIRECT_ALIASES[n] || null;
  if (!assetFile && ICON_MANIFEST) {
    const isUp = category === "upgrade" || /weapon|armor|shield|speed|stimpack|drillclaws|centrifical|chitinous|anabolic|glial|punisher/i.test(n);
    if (isUp) {
      assetFile = (ICON_MANIFEST.upgrades && ICON_MANIFEST.upgrades[n]) || (ICON_MANIFEST.units && ICON_MANIFEST.units[n]);
      if (!assetFile && ICON_MANIFEST.upgrades) {
        const stripped = n.replace(/level[123]|mp/g, "");
        assetFile = ICON_MANIFEST.upgrades[stripped] || (ICON_MANIFEST.units && ICON_MANIFEST.units[stripped]);
      }
    } else {
      assetFile = (ICON_MANIFEST.units && ICON_MANIFEST.units[n]) || (ICON_MANIFEST.upgrades && ICON_MANIFEST.upgrades[n]);
      if (!assetFile && ICON_MANIFEST.units) {
        const stripped = n.replace(/flying|burrowed|sieged|mp|cocoon/g, "");
        assetFile = ICON_MANIFEST.units[stripped] || (ICON_MANIFEST.upgrades && ICON_MANIFEST.upgrades[stripped]);
      }
    }
  }

  const fallback = getFallbackSvg(rawName, category, raceHint);
  if (assetFile) {
    return `<img class="sc2-game-icon-img" src="/assets/icons/${assetFile}" alt="${name}" onerror="this.style.display='none'; if(this.nextElementSibling) this.nextElementSibling.style.display='block';"/><div style="display:none;width:100%;height:100%;">${fallback}</div>`;
  }
  return fallback;
}

function getFallbackSvg(rawName, category, raceHint) {
  if (!rawName) return "";
  const name = String(rawName);
  const n = name.toLowerCase();

  // 1. Upgrades detection
  if (category === "upgrade" || /weapon|armor|shield|speed|stimpack|drillclaws|centrifical|chitinous|anabolic|glial|punisher/i.test(n)) {
    if (/weapon|melee|missile|attack/i.test(n)) {
      const col = /zerg/i.test(n) ? "#ef4444" : (/protoss/i.test(n) ? "#38bdf8" : "#f59e0b");
      return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="36" height="36" rx="4" fill="#0f172a"/>
        <path d="M9 27L25 11M27 9L23 5L20 8L24 12L27 9ZM11 25L7 29L11 31L13 27L11 25Z" stroke="${col}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M27 27L11 11M9 9L13 5L16 8L12 12L9 9ZM25 25L29 29L25 31L23 27L25 25Z" stroke="${col}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>`;
    }
    if (/armor|carapace|plating/i.test(n)) {
      const col = /zerg/i.test(n) ? "#10b981" : (/protoss/i.test(n) ? "#38bdf8" : "#94a3b8");
      return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="36" height="36" rx="4" fill="#0f172a"/>
        <path d="M18 6L8 10V18C8 24.5 12.3 30.5 18 32C23.7 30.5 28 24.5 28 18V10L18 6Z" stroke="${col}" stroke-width="2" fill="rgba(16,185,129,0.15)" stroke-linejoin="round"/>
        <path d="M18 10V28" stroke="${col}" stroke-width="1.5"/>
      </svg>`;
    }
    if (/shield/i.test(n)) {
      return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="36" height="36" rx="4" fill="#0f172a"/>
        <circle cx="18" cy="18" r="10" stroke="#38bdf8" stroke-width="2" stroke-dasharray="4 2"/>
        <circle cx="18" cy="18" r="5" stroke="#818cf8" stroke-width="1.5" fill="rgba(56,189,248,0.2)"/>
      </svg>`;
    }
    if (/speed|boost|reconstitution|centrifical/i.test(n)) {
      return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="36" height="36" rx="4" fill="#0f172a"/>
        <path d="M9 18L18 9V15H27L18 27V21H9L18 12" stroke="#facc15" stroke-width="2" stroke-linejoin="round" fill="rgba(250,204,21,0.2)"/>
      </svg>`;
    }
    if (/stimpack/i.test(n)) {
      return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="36" height="36" rx="4" fill="#0f172a"/>
        <rect x="14" y="9" width="8" height="15" rx="2" stroke="#ef4444" stroke-width="2" fill="rgba(239,68,68,0.2)"/>
        <line x1="18" y1="24" x2="18" y2="30" stroke="#f87171" stroke-width="2"/>
        <line x1="16" y1="6" x2="20" y2="6" stroke="#94a3b8" stroke-width="2"/>
        <line x1="18" y1="6" x2="18" y2="9" stroke="#94a3b8" stroke-width="2"/>
      </svg>`;
    }
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <polygon points="18,6 28,14 28,26 18,32 8,26 8,14" stroke="#a78bfa" stroke-width="1.8" fill="rgba(167,139,250,0.15)"/>
      <path d="M14 18L18 14L22 18M18 15V23" stroke="#e0e7ff" stroke-width="1.8" stroke-linecap="round"/>
    </svg>`;
  }

  // 2. Structures
  if (category === "structure" || /commandcenter|hatchery|nexus|barracks|gateway|spawningpool|factory|roachwarren|roboticsfacility|starport|spire|stargate|supplydepot|pylon|overlord|refinery|assimilator|extractor|evolutionchamber|forge|engineeringbay|armory|twilightcouncil|cyberneticscore|bunker|cannon|battery|crawler/i.test(n)) {
    let col = "#94a3b8";
    let bg = "rgba(51, 65, 85, 0.4)";
    let tag = name.substring(0, 3).toUpperCase();
    if (/commandcenter|orbital|planetary/i.test(n)) { col = "#38bdf8"; tag = "CC"; }
    else if (/hatchery|lair|hive/i.test(n)) { col = "#c084fc"; tag = "HAT"; }
    else if (/nexus/i.test(n)) { col = "#facc15"; tag = "NEX"; }
    else if (/barracks/i.test(n)) { col = "#38bdf8"; tag = "RAX"; }
    else if (/factory/i.test(n)) { col = "#38bdf8"; tag = "FAC"; }
    else if (/starport/i.test(n)) { col = "#38bdf8"; tag = "SPT"; }
    else if (/spawningpool/i.test(n)) { col = "#c084fc"; tag = "POL"; }
    else if (/gateway|warpgate/i.test(n)) { col = "#facc15"; tag = "GTW"; }
    else if (/roachwarren/i.test(n)) { col = "#c084fc"; tag = "RCH"; }
    else if (/spire/i.test(n)) { col = "#c084fc"; tag = "SPI"; }
    else if (/stargate/i.test(n)) { col = "#facc15"; tag = "STG"; }
    else if (/robotics/i.test(n)) { col = "#facc15"; tag = "ROB"; }
    else if (/depot/i.test(n)) { col = "#94a3b8"; tag = "DEP"; }
    else if (/pylon/i.test(n)) { col = "#38bdf8"; tag = "PYL"; }

    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <rect x="6" y="9" width="24" height="20" rx="3" stroke="${col}" stroke-width="1.8" fill="${bg}"/>
      <line x1="6" y1="18" x2="30" y2="18" stroke="${col}" stroke-width="1" stroke-dasharray="2 2"/>
      <text x="18" y="23" fill="#ffffff" font-size="9" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">${tag}</text>
    </svg>`;
  }

  // 3. Units
  let col = "#38bdf8";
  let bg = "rgba(56, 189, 248, 0.15)";
  let tag = name.substring(0, 3).toUpperCase();

  // Workers
  if (/scv/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <path d="M12 11a4 4 0 0 1 7.8-1.3l-2.4 2.4 2.8 2.8 2.4-2.4A4 4 0 0 1 21 20l-9 9a2 2 0 0 1-2.8-2.8l9-9a4 4 0 0 1-6.2-6.2z" stroke="#facc15" stroke-width="1.8" fill="rgba(250,204,21,0.2)"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">SCV</text>
    </svg>`;
  }
  if (/drone/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <path d="M18 9c-3 0-5 3-5 7 0 5 5 11 5 11s5-6 5-11c0-4-2-7-5-7z" stroke="#c084fc" stroke-width="1.8" fill="rgba(192,132,252,0.2)"/>
      <circle cx="15" cy="13" r="1.5" fill="#f43f5e"/>
      <circle cx="21" cy="13" r="1.5" fill="#f43f5e"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">DRN</text>
    </svg>`;
  }
  if (/probe/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <circle cx="18" cy="16" r="7" stroke="#38bdf8" stroke-width="1.8" fill="rgba(56,189,248,0.2)"/>
      <circle cx="18" cy="16" r="3" fill="#facc15"/>
      <path d="M12 25l6-3 6 3" stroke="#38bdf8" stroke-width="1.5"/>
      <text x="18" y="33" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">PRB</text>
    </svg>`;
  }

  // Zerg units
  if (/zergling/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <path d="M9 25c4-1 8-5 11-10 2-4 5-6 7-6-1 3-2 6-5 9-3 4-8 7-13 7z" stroke="#ef4444" stroke-width="1.8" fill="rgba(239,68,68,0.2)"/>
      <path d="M15 17l4-5M11 21l3-4" stroke="#f87171" stroke-width="1.5"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">ZRG</text>
    </svg>`;
  }
  if (/baneling/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <circle cx="18" cy="17" r="9" stroke="#10b981" stroke-width="2" fill="rgba(16,185,129,0.3)"/>
      <circle cx="15" cy="15" r="2.5" fill="#facc15"/>
      <circle cx="21" cy="18" r="2" fill="#facc15"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">BNE</text>
    </svg>`;
  }
  if (/roach/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <path d="M10 22c0-6 4-11 8-11s8 5 8 11z" stroke="#f59e0b" stroke-width="2" fill="rgba(245,158,11,0.2)"/>
      <line x1="18" y1="11" x2="18" y2="22" stroke="#f59e0b" stroke-width="1.5"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">RCH</text>
    </svg>`;
  }
  if (/ravager/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <path d="M11 23l7-14 7 14z" stroke="#f97316" stroke-width="2" fill="rgba(249,115,22,0.2)"/>
      <circle cx="18" cy="17" r="2.5" fill="#facc15"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">RAV</text>
    </svg>`;
  }
  if (/hydralisk/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <path d="M8 24c4-4 8-12 10-15 2 3 6 11 10 15" stroke="#10b981" stroke-width="2" stroke-linecap="round"/>
      <path d="M13 14l5-5 5 5" stroke="#34d399" stroke-width="1.5"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">HYD</text>
    </svg>`;
  }
  if (/mutalisk/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <path d="M6 14c5 0 9 4 12 11 3-7 7-11 12-11-4 4-8 10-12 10S10 18 6 14z" stroke="#c084fc" stroke-width="1.8" fill="rgba(192,132,252,0.2)"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">MUT</text>
    </svg>`;
  }
  if (/ultralisk/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <path d="M7 21c2-6 6-12 11-12s9 6 11 12c-3-2-7-3-11-3s-8 1-11 3z" stroke="#e11d48" stroke-width="2" fill="rgba(225,29,72,0.25)"/>
      <path d="M5 24c3-1 6-4 8-9M31 24c-3-1-6-4-8-9" stroke="#fda4af" stroke-width="2" stroke-linecap="round"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">ULT</text>
    </svg>`;
  }
  if (/overlord/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <path d="M10 16c0-5 3.5-9 8-9s8 4 8 9c0 4-3 7-8 7s-8-3-8-7z" stroke="#a855f7" stroke-width="1.8" fill="rgba(168,85,247,0.2)"/>
      <path d="M13 23v4M18 23v5M23 23v4" stroke="#c084fc" stroke-width="1.5" stroke-linecap="round"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">OVL</text>
    </svg>`;
  }

  // Terran units
  if (/marine/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <circle cx="18" cy="15" r="7" stroke="#38bdf8" stroke-width="1.8" fill="rgba(56,189,248,0.2)"/>
      <rect x="13" y="13" width="10" height="3.5" rx="1.5" fill="#38bdf8"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">MAR</text>
    </svg>`;
  }
  if (/marauder/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <rect x="10" y="9" width="16" height="15" rx="3" stroke="#f59e0b" stroke-width="2" fill="rgba(245,158,11,0.2)"/>
      <circle cx="14" cy="16" r="2" fill="#ef4444"/>
      <circle cx="22" cy="16" r="2" fill="#ef4444"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">MRD</text>
    </svg>`;
  }
  if (/siegetank/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <rect x="8" y="16" width="20" height="8" rx="2" stroke="#94a3b8" stroke-width="1.8" fill="rgba(148,163,184,0.2)"/>
      <circle cx="18" cy="16" r="4" stroke="#f59e0b" stroke-width="1.8"/>
      <line x1="18" y1="14" x2="18" y2="7" stroke="#f59e0b" stroke-width="2.5" stroke-linecap="round"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">TNK</text>
    </svg>`;
  }
  if (/medivac/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <path d="M18 7l7 9h-4v9h-6v-9h-4l7-9z" stroke="#10b981" stroke-width="1.8" fill="rgba(16,185,129,0.2)"/>
      <line x1="18" y1="18" x2="18" y2="22" stroke="#ffffff" stroke-width="1.5"/>
      <line x1="16" y1="20" x2="20" y2="20" stroke="#ffffff" stroke-width="1.5"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">MED</text>
    </svg>`;
  }
  if (/widowmine/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <polygon points="18,8 26,17 18,26 10,17" stroke="#ef4444" stroke-width="2" fill="rgba(239,68,68,0.2)"/>
      <circle cx="18" cy="17" r="2.5" fill="#f87171"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">MINE</text>
    </svg>`;
  }

  // Protoss units
  if (/zealot/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <path d="M12 25l4-15 2 4M24 25l-4-15-2 4" stroke="#38bdf8" stroke-width="2" stroke-linecap="round"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">ZLT</text>
    </svg>`;
  }
  if (/stalker/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <polygon points="18,10 24,18 18,22 12,18" stroke="#a78bfa" stroke-width="2" fill="rgba(167,139,250,0.2)"/>
      <path d="M12 18l-4 6M24 18l4 6" stroke="#818cf8" stroke-width="1.8"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">STK</text>
    </svg>`;
  }
  if (/immortal/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <circle cx="18" cy="16" r="6" stroke="#facc15" stroke-width="2" fill="rgba(250,204,21,0.2)"/>
      <line x1="8" y1="16" x2="12" y2="16" stroke="#38bdf8" stroke-width="2.5"/>
      <line x1="24" y1="16" x2="28" y2="16" stroke="#38bdf8" stroke-width="2.5"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">IMM</text>
    </svg>`;
  }
  if (/colossus/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <path d="M18 8v10M18 18l-8 9M18 18l8 9" stroke="#f59e0b" stroke-width="2" stroke-linecap="round"/>
      <circle cx="18" cy="8" r="3" fill="#facc15"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">COL</text>
    </svg>`;
  }
  if (/carrier/i.test(n)) {
    return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="4" fill="#0f172a"/>
      <polygon points="18,6 27,22 18,19 9,22" stroke="#facc15" stroke-width="2" fill="rgba(250,204,21,0.2)"/>
      <text x="18" y="32" fill="#ffffff" font-size="8" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">CAR</text>
    </svg>`;
  }

  // Fallback unit card
  return `<svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="36" height="36" rx="4" fill="#0f172a"/>
    <circle cx="18" cy="16" r="8" stroke="${col}" stroke-width="1.8" fill="${bg}"/>
    <text x="18" y="20" fill="#ffffff" font-size="9" font-weight="800" font-family="system-ui,sans-serif" text-anchor="middle">${tag}</text>
  </svg>`;
}

function renderObserverPanel() {
  const canvas = document.getElementById("observerFanCanvas");
  const btnMap = document.getElementById("btnToggleThreatMap");
  const bgSel = document.getElementById("mapBgMode");
  if (btnMap && bgSel) {
    const isThreat = bgSel.value.startsWith("threat_");
    btnMap.textContent = isThreat ? "✓ Leverage Map" : "Leverage Map";
    btnMap.style.background = isThreat ? "#4f46e5" : "";
    btnMap.style.color = isThreat ? "#ffffff" : "#e0e7ff";
  }
  if (!canvas) return;
  const { ctx, w, h } = setupHiDPICanvas(canvas);
  ctx.clearRect(0, 0, w, h);

  if (!OBSERVER_DATA || !OBSERVER_DATA.quantiles) {
    ctx.fillStyle = "#64748b";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Computing 32-branch +120s Theory-of-Mind rollout…", w * 0.5, h * 0.52);
    return;
  }

  const d = OBSERVER_DATA;
  const p1NowProb = (d.prob_lead_now_p1 !== undefined && d.prob_lead_now_p1 !== null)
    ? d.prob_lead_now_p1
    : d.prob_stronger_hand_p1;
  const p1NowPct = Math.round(p1NowProb * 100);
  const p2NowPct = 100 - p1NowPct;
  const p1StrongPct = Math.round(d.prob_stronger_hand_p1 * 100);
  const p2StrongPct = 100 - p1StrongPct;
  const p1InitPct = Math.round(d.prob_improve_p1 * 100);
  const p2InitPct = 100 - p1InitPct;

  const p1NameShort = (FRAME && FRAME.hud && FRAME.hud.p1 && FRAME.hud.p1.name)
    ? unescapeHtml(FRAME.hud.p1.name).slice(0, 9)
    : "P1";
  const p2NameShort = (FRAME && FRAME.hud && FRAME.hud.p2 && FRAME.hud.p2.name)
    ? unescapeHtml(FRAME.hud.p2.name).slice(0, 9)
    : "P2";

  const elSP1 = document.getElementById("obsStrongP1");
  const elSCtr = document.getElementById("obsLeadCenterLbl");
  const elSP2 = document.getElementById("obsStrongP2");
  const elSBar = document.getElementById("obsStrongBarP1");
  if (elSP1) elSP1.textContent = `${p1NameShort}: ${p1NowPct}% (→${p1StrongPct}%)`;
  if (elSCtr) elSCtr.textContent = d.mode === "counterplay_p1"
    ? "Omni • P1 Opt +120s"
    : (d.mode === "counterplay_p2" ? "Omni • P2 Opt +120s" : "Omni Now • +120s");
  if (elSP2) elSP2.textContent = `(${p2StrongPct}%←) ${p2NowPct}% :${p2NameShort}`;
  if (elSBar) elSBar.style.width = `${Math.max(2, Math.min(98, p1NowPct))}%`;

  const fmtRes = v => `${v >= 0 ? "+" : ""}${Math.round(v)}`;
  const elA0 = document.getElementById("obsKpiA0");
  if (elA0) {
    elA0.textContent = fmtRes(d.A0);
    elA0.style.color = d.A0 >= 0 ? "var(--p1)" : "var(--p2)";
  }
  const elDr = document.getElementById("obsKpiDrift");
  if (elDr) {
    elDr.textContent = fmtRes(d.expected_drift);
    elDr.style.color = d.expected_drift >= 0 ? "var(--p1)" : "var(--p2)";
  }
  const elIn = document.getElementById("obsKpiInit");
  if (elIn) {
    if (p1InitPct >= 50) {
      elIn.textContent = `P1 ${p1InitPct}%`;
      elIn.style.color = "var(--p1)";
    } else {
      elIn.textContent = `P2 ${p2InitPct}%`;
      elIn.style.color = "var(--p2)";
    }
  }
  const elVol = document.getElementById("obsKpiVol");
  const elVolCell = document.getElementById("obsKpiVolCell");
  const pm1 = (d.plan_misalignment && d.plan_misalignment.p1 !== undefined) ? d.plan_misalignment.p1 : 0;
  const pm2 = (d.plan_misalignment && d.plan_misalignment.p2 !== undefined) ? d.plan_misalignment.p2 : 0;
  if (elVol) {
    elVol.textContent = `${pm1.toFixed(3)}|${pm2.toFixed(3)}`;
    elVol.style.color = Math.max(pm1, pm2) > 0.025 ? "#fbbf24" : "#cbd5e1";
  }
  if (elVolCell) {
    elVolCell.title =
      `Plan Misalignment ||v(h_omni) - v(h_fog)||: P1=${pm1.toFixed(3)}, P2=${pm2.toFixed(3)} • ` +
      `Volatility σ(ΔA)=±${Math.round(d.std_drift)} • CVaR₁₀%: P1 ${fmtRes(d.cvar10_p1)} | P2 ${fmtRes(-d.cvar10_p2)}`;
  }

  const elVolSum = document.getElementById("obsVolSummary");
  if (elVolSum) {
    elVolSum.textContent = `σ=±${Math.round(d.std_drift)} • CVaR₁₀ ${fmtRes(d.cvar10_p1)}|${fmtRes(-d.cvar10_p2)}`;
  }

  // Render Theory-of-Mind Unscouted Threat Attribution Pills
  const elUnsc = document.getElementById("obsUnscoutedList");
  if (elUnsc) {
    elUnsc.innerHTML = "";
    const ut = d.unscouted_threats || { p1: [], p2: [] };
    const items = [];
    (ut.p1 || []).slice(0, 2).forEach(th => items.push({ viewer: 1, ...th }));
    (ut.p2 || []).slice(0, 2).forEach(th => items.push({ viewer: 2, ...th }));
    items.sort((a, b) => (b.attr || 0) - (a.attr || 0));

    if (items.length === 0) {
      const emptySpan = document.createElement("span");
      emptySpan.style.color = "var(--text-muted)";
      emptySpan.textContent = "No high-attribution unscouted threats right now.";
      elUnsc.appendChild(emptySpan);
    } else {
      items.slice(0, 3).forEach(th => {
        const pill = document.createElement("div");
        pill.className = "obs-threat-pill";
        const isActive = HIGHLIGHT_THREAT_XY &&
          Math.abs(HIGHLIGHT_THREAT_XY.x - th.x) < 0.2 &&
          Math.abs(HIGHLIGHT_THREAT_XY.y - th.y) < 0.2;
        if (isActive) {
          pill.style.borderColor = "#fbbf24";
          pill.style.background = "rgba(120, 53, 15, 0.45)";
        }
        const vTagCol = th.viewer === 1 ? "#fca5a5" : "#93c5fd";
        const shimBadge = (th.shimmer && th.shimmer >= 0.15)
          ? `<span style="color:#fde047; font-weight:700;" title="Cloaked/burrowed shimmer visible on-screen!">✦Shimmer</span> `
          : "";
        pill.innerHTML =
          `<span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">` +
          `<b style="color:${vTagCol};">P${th.viewer} Blind:</b> ${shimBadge}${escapeHtml(th.units)}</span>` +
          `<span style="color:#fbbf24; white-space:nowrap;">(${Math.round(th.x)},${Math.round(th.y)}) Δπ=${(Number(th.attr) * 1000).toFixed(1)}m</span>`;
        pill.title = `Click to highlight on 2D map • Enemy P${th.enemy_owner} squad at (${th.x}, ${th.y}) • Unscouted=${Math.round((th.unscouted_frac || 0) * 100)}% • Shimmer=${Math.round((th.shimmer || 0) * 100)}%`;
        pill.addEventListener("click", () => {
          if (HIGHLIGHT_THREAT_XY &&
              Math.abs(HIGHLIGHT_THREAT_XY.x - th.x) < 0.2 &&
              Math.abs(HIGHLIGHT_THREAT_XY.y - th.y) < 0.2) {
            HIGHLIGHT_THREAT_XY = null;
          } else {
            HIGHLIGHT_THREAT_XY = {
              x: th.x,
              y: th.y,
              viewer: th.viewer,
              enemy_owner: th.enemy_owner,
              units: th.units,
              shimmer: th.shimmer || 0
            };
          }
          renderObserverPanel();
          renderMap();
        });
        elUnsc.appendChild(pill);
      });
    }
  }

  // Draw Multi-Horizon Advantage Fan Chart (Now .. +120s)
  const q = d.quantiles;
  const secs = d.step_secs || [0, 30, 60, 90, 120];
  const K = secs.length;
  const padL = 38, padR = 10, padT = 8, padB = 16;
  const cw = Math.max(20, w - padL - padR);
  const ch = Math.max(20, h - padT - padB);

  let yMin = Math.min(0, ...q.p10);
  let yMax = Math.max(0, ...q.p90);
  (d.branches || []).forEach(br => {
    br.forEach(v => {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    });
  });
  const span = Math.max(50.0, yMax - yMin);
  yMin -= span * 0.08;
  yMax += span * 0.08;

  const xOf = i => padL + (i / Math.max(1, K - 1)) * cw;
  const yOf = v => padT + (1.0 - (v - yMin) / Math.max(1e-6, yMax - yMin)) * ch;

  // Background zero line (A = 0)
  const yZero = yOf(0);
  if (yZero >= padT && yZero <= padT + ch) {
    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(padL, yZero);
    ctx.lineTo(padL + cw, yZero);
    ctx.stroke();
    ctx.restore();
  }

  // Shaded p10..p90 outer fan
  const leadEnd = q.p50[K - 1] >= 0;
  const bandOuter = leadEnd ? "rgba(239, 68, 68, 0.16)" : "rgba(59, 130, 246, 0.16)";
  const bandInner = leadEnd ? "rgba(239, 68, 68, 0.30)" : "rgba(59, 130, 246, 0.30)";
  const lineCol = leadEnd ? "#f87171" : "#60a5fa";

  const drawBand = (loArr, hiArr, fillStyle) => {
    ctx.fillStyle = fillStyle;
    ctx.beginPath();
    for (let i = 0; i < K; i++) {
      const x = xOf(i), y = yOf(hiArr[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    for (let i = K - 1; i >= 0; i--) {
      ctx.lineTo(xOf(i), yOf(loArr[i]));
    }
    ctx.closePath();
    ctx.fill();
  };

  drawBand(q.p10, q.p90, bandOuter);
  drawBand(q.p25, q.p75, bandInner);

  // Individual stochastic branch trajectories (spaghetti lines)
  (d.branches || []).slice(0, 14).forEach(br => {
    ctx.strokeStyle = br[K - 1] >= 0 ? "rgba(248, 113, 113, 0.22)" : "rgba(96, 165, 250, 0.22)";
    ctx.lineWidth = 0.85;
    ctx.beginPath();
    for (let i = 0; i < K; i++) {
      const x = xOf(i), y = yOf(br[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  // Median trajectory p50
  ctx.strokeStyle = lineCol;
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  for (let i = 0; i < K; i++) {
    const x = xOf(i), y = yOf(q.p50[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Origin dot at tau_0
  ctx.fillStyle = "#f8fafc";
  ctx.beginPath();
  ctx.arc(xOf(0), yOf(q.p50[0]), 2.8, 0, Math.PI * 2);
  ctx.fill();

  // Y-axis ticks & X-axis time horizon labels
  ctx.fillStyle = "#94a3b8";
  ctx.font = "9px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.fillText(fmtRes(yMax), padL - 4, padT + 7);
  if (yZero >= padT + 9 && yZero <= padT + ch - 7) {
    ctx.fillText("0", padL - 4, yZero + 3);
  }
  ctx.fillText(fmtRes(yMin), padL - 4, padT + ch);

  ctx.textAlign = "center";
  const tickIdxs = Array.from(new Set([
    0,
    Math.floor((K - 1) * 0.25),
    Math.floor((K - 1) * 0.50),
    Math.floor((K - 1) * 0.75),
    K - 1
  ]));
  tickIdxs.forEach(idx => {
    if (idx >= 0 && idx < K) {
      const lbl = idx === 0 ? "Now" : `+${Math.round(secs[idx])}s`;
      ctx.fillText(lbl, xOf(idx), h - 3);
    }
  });
}

function renderHUDAndForces() {
  if (!FRAME || !FRAME.hud) return;
  const h = FRAME.hud;
  const p = FRAME.production || { p1: [], p2: [] };
  const f = FRAME.forces || { p1: { army: [], workers: [], structures: [] }, p2: { army: [], workers: [], structures: [] } };

  // 1. Leaderboard HUD
  for (const pKey of ["p1", "p2"]) {
    const d = h[pKey];
    if (!d) continue;
    const pUpper = pKey.toUpperCase();

    const elName = document.getElementById(`hud${pUpper}Name`);
    if (elName) elName.textContent = unescapeHtml(d.name || `Player ${pKey === 'p1' ? 1 : 2}`);

    const elRace = document.getElementById(`hud${pUpper}Race`);
    if (elRace) {
      const r = formatRace(d.race, pKey);
      let crest = "icon-1v1-terran.png";
      if (r.code === "Z") crest = "icon-1v1-zerg.png";
      else if (r.code === "P") crest = "icon-1v1-protoss.png";
      elRace.innerHTML = `<img class="esl-race-crest-img" src="/assets/icons/${crest}" alt="${r.name}" onerror="this.style.display='none'; this.nextElementSibling.style.display='inline-block';"/><span style="display:none; font-size:10px; font-weight:800;">${r.code}</span>`;
      elRace.title = r.name;
    }

    const elSup = document.getElementById(`hud${pUpper}Supply`);
    if (elSup) elSup.textContent = `${d.supply_used}/${d.supply_cap}`;

    const elWrk = document.getElementById(`hud${pUpper}Workers`);
    if (elWrk) elWrk.textContent = d.workers;

    const elArmy = document.getElementById(`hud${pUpper}Army`);
    if (elArmy) elArmy.textContent = d.army_supply;

    const elMin = document.getElementById(`hud${pUpper}Minerals`);
    if (elMin) elMin.textContent = d.minerals;
    const elMinRate = document.getElementById(`hud${pUpper}MinRate`);
    if (elMinRate) elMinRate.textContent = `+${d.minerals_rate}`;

    const elVesp = document.getElementById(`hud${pUpper}Vespene`);
    if (elVesp) elVesp.textContent = d.vespene;
    const elVespRate = document.getElementById(`hud${pUpper}VespRate`);
    if (elVespRate) elVespRate.textContent = `+${d.vespene_rate}`;

    const elLost = document.getElementById(`hud${pUpper}Lost`);
    if (elLost) elLost.textContent = (d.resources_lost || 0).toLocaleString();
  }

  // 2. Production Queue (Image 1 style)
  for (const pKey of ["p1", "p2"]) {
    const elProd = document.getElementById(`prod${pKey.toUpperCase()}Cards`);
    if (!elProd) continue;
    const items = p[pKey] || [];
    if (items.length === 0) {
      elProd.innerHTML = `<span style="font-size:11px; color:var(--text-muted); font-style:italic;">No active production</span>`;
    } else {
      elProd.innerHTML = items.map(item => {
        const progPct = Math.min(100, Math.max(0, Math.round((item.progress || 0) * 100)));
        const icon = getUnitOrUpgradeIcon(item.name, item.category);
        return `
          <div class="hud-card ${pKey}" title="${item.name}: ${item.count} in production (${progPct}% complete)">
            <div class="hud-card-icon">${icon}</div>
            <div class="hud-card-badge">${item.count}</div>
            <div class="hud-card-progress-track">
              <div class="hud-card-progress-bar" style="width:${progPct}%;"></div>
            </div>
          </div>
        `;
      }).join("");
    }
  }

  // 3. Units & Forces (Workers & Army)
  for (const pKey of ["p1", "p2"]) {
    const elForces = document.getElementById(`forces${pKey.toUpperCase()}Cards`);
    if (!elForces) continue;
    const fData = f[pKey] || { army: [], workers: [] };
    const allUnits = [...(fData.workers || []), ...(fData.army || [])];
    if (allUnits.length === 0) {
      elForces.innerHTML = `<span style="font-size:11px; color:var(--text-muted); font-style:italic;">No units alive</span>`;
    } else {
      elForces.innerHTML = allUnits.map(item => {
        const groupKey = `${pKey}_${item.name}`;
        const isSel = SELECTED_GROUP_KEY === groupKey;
        const icon = getUnitOrUpgradeIcon(item.name, "unit");
        return `
          <div class="hud-card ${pKey} ${isSel ? 'selected' : ''}" 
               data-group-key="${groupKey}"
               onclick="selectUnitGroup('${pKey}', '${item.name}')"
               title="${pKey.toUpperCase()} ${item.name}: ${item.count} alive (Click to isolate on map & inspect top SAE features)">
            <div class="hud-card-icon">${icon}</div>
            <div class="hud-card-badge">${item.count}</div>
          </div>
        `;
      }).join("");
    }
  }

  // 4. Structures
  for (const pKey of ["p1", "p2"]) {
    const elStructs = document.getElementById(`structs${pKey.toUpperCase()}Cards`);
    if (!elStructs) continue;
    const sData = (f[pKey] && f[pKey].structures) || [];
    if (sData.length === 0) {
      elStructs.innerHTML = `<span style="font-size:11px; color:var(--text-muted); font-style:italic;">No structures alive</span>`;
    } else {
      elStructs.innerHTML = sData.map(item => {
        const groupKey = `${pKey}_${item.name}`;
        const isSel = SELECTED_GROUP_KEY === groupKey;
        const icon = getUnitOrUpgradeIcon(item.name, "structure");
        return `
          <div class="hud-card ${pKey} ${isSel ? 'selected' : ''}" 
               data-group-key="${groupKey}"
               onclick="selectUnitGroup('${pKey}', '${item.name}')"
               title="${pKey.toUpperCase()} ${item.name}: ${item.count} structures (Click to isolate on map & inspect top SAE features)">
            <div class="hud-card-icon">${icon}</div>
            <div class="hud-card-badge">${item.count}</div>
          </div>
        `;
      }).join("");
    }
  }

  // 4. Completed Upgrades (Row Groups)
  for (const pKey of ["p1", "p2"]) {
    const elUps = document.getElementById(`upgrades${pKey.toUpperCase()}Cards`);
    if (!elUps) continue;
    const d = h[pKey] || {};
    const ups = d.upgrades || [];
    if (ups.length === 0) {
      elUps.innerHTML = `<span style="font-size:11px; color:var(--text-muted); font-style:italic;">No upgrades completed</span>`;
    } else {
      elUps.innerHTML = ups.map(u => {
        const icon = getUnitOrUpgradeIcon(u.name, "upgrade", d.race);
        const lvlBadge = (u.level && u.level > 1) ? `<div class="hud-card-badge" style="background:#4338ca;">+${u.level}</div>` : "";
        return `
          <div class="hud-card ${pKey}" title="${u.name} (Level ${u.level || 1})">
            <div class="hud-card-icon">${icon}</div>
            ${lvlBadge}
          </div>
        `;
      }).join("");
    }
  }

  // 5. Units Lost & Resources Lost Breakdown (Row Groups)
  for (const pKey of ["p1", "p2"]) {
    const d = h[pKey] || {};
    const hdr = document.getElementById(`${pKey}LostHeader`);
    if (hdr) {
      const minLost = (d.minerals_lost || 0).toLocaleString();
      const vespLost = (d.vespene_lost || 0).toLocaleString();
      const totLost = (d.resources_lost || 0).toLocaleString();
      hdr.innerHTML = `
        <span style="display:inline-flex; align-items:center; gap:3px;" title="Minerals Lost"><img src="/assets/icons/icon-mineral-nobg.png" style="width:11px; height:11px;" alt="Minerals"/> ${minLost}</span>
        <span style="display:inline-flex; align-items:center; gap:3px;" title="Vespene Gas Lost"><img src="/assets/icons/icon-gas-terran-nobg.png" style="width:11px; height:11px;" alt="Gas"/> ${vespLost}</span>
        <span style="color:var(--text-muted); font-weight:400;" title="Total Resources Lost">(Total: ${totLost})</span>
      `;
    }

    const elLost = document.getElementById(`lost${pKey.toUpperCase()}Cards`);
    if (!elLost) continue;
    const deadList = (FRAME && FRAME.units_lost && FRAME.units_lost[pKey]) || [];
    if (deadList.length === 0) {
      elLost.innerHTML = `<span style="font-size:11px; color:var(--text-muted); font-style:italic;">No losses</span>`;
    } else {
      elLost.innerHTML = deadList.map(item => {
        const icon = getUnitOrUpgradeIcon(item.name, "unit");
        return `
          <div class="hud-card ${pKey}" style="opacity:0.9;" title="${pKey.toUpperCase()}: ${item.count} ${item.name} lost">
            <div class="hud-card-icon">${icon}</div>
            <div class="hud-card-badge" style="background:#dc2626;">${item.count}</div>
          </div>
        `;
      }).join("");
    }
  }
}

function selectUnitGroup(playerKey, unitName) {
  const groupKey = `${playerKey}_${unitName}`;
  if (SELECTED_GROUP_KEY === groupKey) {
    SELECTED_GROUP_KEY = null;
    SELECTED_GROUP_ROWS.clear();
    renderHUDAndForces();
    renderReverseSaeAttribution(null, null, []);
    renderMap();
    return;
  }

  SELECTED_GROUP_KEY = groupKey;
  const fData = (FRAME && FRAME.forces && FRAME.forces[playerKey]) || {};
  const allGroups = [...(fData.workers || []), ...(fData.army || []), ...(fData.structures || [])];
  const grp = allGroups.find(x => x.name === unitName);
  const rows = grp ? (grp.rows || []) : [];

  SELECTED_GROUP_ROWS = new Set(rows);
  SELECTED_UNIT_ROW = null;
  SELECTED_TOK_IDX = null;
  USER_DESELECTED_UNIT = true;

  renderHUDAndForces();
  renderReverseSaeAttribution(playerKey, unitName, rows);
  renderMap();
}

function renderReverseSaeAttribution(playerKey, unitName, rows) {
  const pnlTitle = document.getElementById("reverseSaeTitle");
  const pnlBody = document.getElementById("reverseSaeBody");
  const btnDeselect = document.getElementById("btnDeselectGroup");

  if (!playerKey || !unitName || !rows || rows.length === 0 || !FRAME || !FRAME.units) {
    if (pnlTitle) pnlTitle.textContent = "Reverse SAE Feature Attribution";
    if (btnDeselect) btnDeselect.style.display = "none";
    if (pnlBody) {
      pnlBody.innerHTML = `Click any unit or structure card above (e.g. <b>56 SCVs</b> or <b>6 Marauders</b>) to highlight them across the battlefield and inspect which Sparse Autoencoder features activate the strongest.`;
    }
    return;
  }

  const pName = unescapeHtml(playerKey === "p1" ? (FRAME.hud ? FRAME.hud.p1.name : "Player 1") : (FRAME.hud ? FRAME.hud.p2.name : "Player 2"));
  const pCol = playerKey === "p1" ? "var(--p1)" : "var(--p2)";
  const groupIcon = getUnitOrUpgradeIcon(unitName, "unit");
  if (pnlTitle) {
    pnlTitle.innerHTML = `<span style="display:inline-flex; align-items:center; gap:6px;">
      <span style="width:20px; height:20px; display:inline-block; border-radius:3px; overflow:hidden; vertical-align:middle;">${groupIcon}</span>
      <span style="color:${pCol};font-weight:800;">${escapeHtml(pName)}</span> • ${escapeHtml(unitName)} (${rows.length} units)
    </span>`;
  }
  if (btnDeselect) {
    btnDeselect.style.display = "inline-block";
    btnDeselect.onclick = () => {
      SELECTED_GROUP_KEY = null;
      SELECTED_GROUP_ROWS.clear();
      renderHUDAndForces();
      renderReverseSaeAttribution(null, null, []);
      renderMap();
    };
  }

  const matchingUnits = FRAME.units.filter(u => rows.includes(u.row));
  if (matchingUnits.length === 0) {
    pnlBody.innerHTML = `<span style="color:var(--text-muted);">No units currently alive in this group.</span>`;
    return;
  }

  const featMap = new Map();
  matchingUnits.forEach(u => {
    if (!u.sae_layers) return;
    u.sae_layers.forEach((layerFeats, lIdx) => {
      if (!Array.isArray(layerFeats)) return;
      layerFeats.forEach(f => {
        if (!f || f.act <= 0.05) return;
        const key = `L${lIdx}_F${f.fid}`;
        if (!featMap.has(key)) {
          featMap.set(key, {
            key,
            layer: lIdx,
            fid: f.fid,
            label: cleanSaeLabel(f.label || `Feature ${f.fid}`),
            sumAct: 0.0,
            maxAct: 0.0,
            activeCount: 0,
          });
        }
        const entry = featMap.get(key);
        entry.sumAct += f.act;
        entry.maxAct = Math.max(entry.maxAct, f.act);
        entry.activeCount++;
      });
    });
  });

  const featList = Array.from(featMap.values()).map(f => ({
    ...f,
    meanAct: f.sumAct / matchingUnits.length,
  }));
  featList.sort((a, b) => b.meanAct - a.meanAct);

  const topFeats = featList.slice(0, 7);
  if (topFeats.length === 0) {
    pnlBody.innerHTML = `<div style="color:var(--text-secondary);font-style:italic;">No strong SAE feature activations recorded for this unit group on this frame.</div>`;
    return;
  }

  // Normalize relative to strongest activation in the selected group
  const maxMean = Math.max(...topFeats.map(f => f.meanAct), 0.001);

  const html = `
    <div style="font-size:10.5px;color:var(--text-secondary);margin-bottom:3px;display:flex;justify-content:space-between;">
      <span>Top Monosemantic Features (Blocks 0..6):</span>
      <span>${matchingUnits.length} alive</span>
    </div>
    <div class="sae-attr-grid">
      ${topFeats.map(f => {
        const barPct = Math.min(100, Math.max(3, Math.round((f.meanAct / maxMean) * 100)));
        const isSoloActive = ACTIVE_SAE_SELECTION && ACTIVE_SAE_SELECTION.layer === f.layer && ACTIVE_SAE_SELECTION.fid === f.fid;
        return `
          <div class="sae-feat-card" title="Layer ${f.layer}, Feature ${f.fid}: Mean ${f.meanAct.toFixed(2)}, Max ${f.maxAct.toFixed(2)} (${f.activeCount}/${matchingUnits.length} active)">
            <div style="display:flex;flex-direction:column;gap:1px;min-width:0;flex:1;">
              <div style="display:flex;align-items:center;gap:4px;">
                <span class="badge" style="font-size:9.5px;background:rgba(99,102,241,0.25);border-color:#818cf8;color:#e0e7ff;padding:0 3px;">B${f.layer}</span>
                <span style="font-weight:700;color:#f1f5f9;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">F${String(f.fid).padStart(3, '0')}: ${f.label}</span>
              </div>
              <div style="display:flex;align-items:center;gap:6px;font-size:9.5px;color:var(--text-secondary);margin-top:1px;">
                <span>Σ: <b>${f.sumAct.toFixed(1)}</b></span>
                <span>μ: <b>${f.meanAct.toFixed(2)}</b> <span style="color:#a5b4fc;font-size:9px;">(${barPct}%)</span></span>
                <span>max: <b>${f.maxAct.toFixed(2)}</b></span>
                <span>(${f.activeCount}/${matchingUnits.length})</span>
                <div style="flex:1;height:4px;background:rgba(51,65,85,0.5);border-radius:2px;overflow:hidden;margin-left:4px;">
                  <div style="height:100%;width:${barPct}%;background:linear-gradient(90deg,#818cf8,#c084fc);border-radius:2px;"></div>
                </div>
              </div>
            </div>
            <button style="padding:1px 6px;font-size:9.5px;flex-shrink:0;${isSoloActive ? 'background:#818cf8;color:#0b1120;font-weight:700;' : ''}" 
                    onclick="isolateSaeFeatureOnMap(${f.layer}, ${f.fid}, '${escapeHtml(f.label)}', event)" 
                    title="Paint and highlight this SAE feature across the battlefield">
              ${isSoloActive ? '✓ Active' : 'Solo Map'}
            </button>
          </div>
        `;
      }).join("")}
    </div>
  `;
  pnlBody.innerHTML = html;
}

let ACTIVE_SAE_SELECTION = null;

function isolateSaeFeatureOnMap(layer, fid, label, event) {
  if (event) event.stopPropagation();
  if (ACTIVE_SAE_SELECTION && ACTIVE_SAE_SELECTION.layer === layer && ACTIVE_SAE_SELECTION.fid === fid) {
    ACTIVE_SAE_SELECTION = null;
    ACTIVE_SAE_FID = -1;
    const saeSelect = document.getElementById("saeMapSelect");
    if (saeSelect) saeSelect.value = "-1";
    renderMap();
    renderSelectionBar();
    if (SELECTED_GROUP_KEY && SELECTED_GROUP_ROWS.size > 0) {
      const parts = SELECTED_GROUP_KEY.split("_");
      renderReverseSaeAttribution(parts[0], parts.slice(1).join("_"), Array.from(SELECTED_GROUP_ROWS));
    }
    return;
  }

  ACTIVE_SAE_SELECTION = { layer, fid, label };
  ACTIVE_SAE_FID = fid;

  const saeSelect = document.getElementById("saeMapSelect");
  if (saeSelect) {
    const valKey = String(fid);
    let opt = saeSelect.querySelector(`option[value="${valKey}"]`);
    if (!opt) {
      opt = document.createElement("option");
      opt.value = valKey;
      opt.textContent = `[L${layer} F${String(fid).padStart(3, '0')}] ${cleanSaeLabel(label)}`;
      saeSelect.appendChild(opt);
    }
    saeSelect.value = valKey;
  }

  const bgSel = document.getElementById("mapBgMode");
  if (bgSel && bgSel.value === "terrain") {
    bgSel.value = "diff";
  }
  renderMap();
  renderSelectionBar();
  if (SELECTED_GROUP_KEY && SELECTED_GROUP_ROWS.size > 0) {
    const parts = SELECTED_GROUP_KEY.split("_");
    renderReverseSaeAttribution(parts[0], parts.slice(1).join("_"), Array.from(SELECTED_GROUP_ROWS));
  }
}
