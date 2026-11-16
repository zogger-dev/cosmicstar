// ============================================================================
// EVENT LISTENERS & INTERACTION
// ============================================================================
document.getElementById("gameSelect").addEventListener("change", async e => {
  CURRENT_GID = e.target.value;
  COUNTERFACTUAL = { deleted_rows: [], moved_rows: {}, hp_scale: {} };
  await loadGameTimeline(CURRENT_GID);
  const gInfo = CATALOG.find(x => x.gid === CURRENT_GID);
  CURRENT_LOOP = gInfo ? Math.floor(gInfo.duration_loops * 0.45) : 2000;
  await loadFrame(CURRENT_GID, CURRENT_LOOP);
});

document.getElementById("l1CkptSelect").addEventListener("change", async e => {
  await fetch("/api/checkpoint", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ l1_ckpt: e.target.value })
  });
  await loadFrame(CURRENT_GID, CURRENT_LOOP);
});

document.getElementById("l2ModeSelect").addEventListener("change", async () => {
  await loadGameTimeline(CURRENT_GID);
});

document.getElementById("loopInput").addEventListener("change", e => {
  loadFrame(CURRENT_GID, parseInt(e.target.value, 10));
});
document.getElementById("winLenSelect").addEventListener("change", () => {
  loadFrame(CURRENT_GID, CURRENT_LOOP);
});

document.getElementById("btnStepBack5s").addEventListener("click", () => loadFrame(CURRENT_GID, CURRENT_LOOP - 112));
const bBack1s = document.getElementById("btnStepBack1s");
if (bBack1s) bBack1s.addEventListener("click", () => loadFrame(CURRENT_GID, CURRENT_LOOP - 22));
document.getElementById("btnStepBack1").addEventListener("click", () => loadFrame(CURRENT_GID, CURRENT_LOOP - 1));
document.getElementById("btnStepFwd1").addEventListener("click", () => loadFrame(CURRENT_GID, CURRENT_LOOP + 1));
const bFwd1s = document.getElementById("btnStepFwd1s");
if (bFwd1s) bFwd1s.addEventListener("click", () => loadFrame(CURRENT_GID, CURRENT_LOOP + 22));
document.getElementById("btnStepFwd5s").addEventListener("click", () => loadFrame(CURRENT_GID, CURRENT_LOOP + 112));

document.getElementById("btnPlayPause").addEventListener("click", e => {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    e.target.textContent = "▶ Play";
    e.target.classList.remove("active");
  } else {
    e.target.textContent = "⏸ Pause";
    e.target.classList.add("active");
    playTimer = setInterval(() => {
      loadFrame(CURRENT_GID, CURRENT_LOOP + 28);
    }, 350);
  }
});

["mapBgMode", "fieldCombineMode", "unitColorMode", "chkFieldContours", "chkMomentumQuivers", "chkAttnRays", "chkForceAdv", "chkForceKin", "chkS2MA", "chkSquadTokens", "chkAnchors"].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", () => {
    renderMap();
    renderObserverPanel();
  });
});

const btnToggleThreat = document.getElementById("btnToggleThreatMap");
if (btnToggleThreat) {
  btnToggleThreat.addEventListener("click", () => {
    const bgSel = document.getElementById("mapBgMode");
    const sqChk = document.getElementById("chkSquadTokens");
    if (!bgSel) return;
    if (bgSel.value.startsWith("threat_")) {
      bgSel.value = "diff";
      if (sqChk) sqChk.checked = false;
    } else {
      bgSel.value = "threat_net";
      if (sqChk) sqChk.checked = true;
    }
    renderAll();
  });
}

// Observer v2 Mode Switcher (Forecast Play | P1 Counterplay | P2 Counterplay)
document.querySelectorAll(".obs-mode-btn[data-obs-mode]").forEach(btn => {
  btn.addEventListener("click", () => {
    const m = btn.getAttribute("data-obs-mode") || "forecast";
    if (!CURRENT_GID) return;
    loadObserverRollout(CURRENT_GID, CURRENT_LOOP, m);
  });
});

// Observer v2 +5.0s L1ChunkFlowDecoder Micro Scrubber & Play/Pause
const microSliderEl = document.getElementById("microStepSlider");
if (microSliderEl) {
  microSliderEl.addEventListener("input", e => {
    stopMicroPlayback();
    MICRO_STEP = parseInt(e.target.value, 10) || 0;
    updateMicroStepUI();
    renderMap();
  });
}

const btnMicroPlayEl = document.getElementById("btnMicroPlay");
if (btnMicroPlayEl) {
  btnMicroPlayEl.addEventListener("click", () => {
    if (microPlayTimer) {
      stopMicroPlayback();
      return;
    }
    if (!OBSERVER_DATA || !OBSERVER_DATA.micro_chunk) return;
    const maxS = OBSERVER_DATA.micro_chunk.n_steps || 16;
    if (MICRO_STEP >= maxS) MICRO_STEP = 0;
    btnMicroPlayEl.textContent = "⏸ +5s Micro";
    microPlayTimer = setInterval(() => {
      if (!OBSERVER_DATA || !OBSERVER_DATA.micro_chunk) {
        stopMicroPlayback();
        return;
      }
      const nS = OBSERVER_DATA.micro_chunk.n_steps || 16;
      MICRO_STEP = MICRO_STEP >= nS ? 0 : (MICRO_STEP + 1);
      updateMicroStepUI();
      renderMap();
    }, 180);
  });
}

const alphaSliderEl = document.getElementById("fieldAlphaSlider");
if (alphaSliderEl) {
  alphaSliderEl.addEventListener("input", () => renderMap());
}

const saeMapSelEl = document.getElementById("saeMapSelect");
if (saeMapSelEl) {
  saeMapSelEl.addEventListener("change", e => {
    ACTIVE_SAE_FID = parseInt(e.target.value, 10);
    if (ACTIVE_SAE_FID >= 0) {
      const bgSel = document.getElementById("mapBgMode");
      if (bgSel && bgSel.value === "terrain") bgSel.value = "diff";
    }
    renderMap();
  });
}

document.querySelectorAll(".dim-pill[data-dim]").forEach(btn => {
  btn.addEventListener("click", e => {
    const rDim = getRDim();
    const dAttr = btn.getAttribute("data-dim");
    const isMulti = e.shiftKey || e.ctrlKey || e.metaKey || (document.getElementById("chkMultiDim") && document.getElementById("chkMultiDim").checked);

    if (dAttr === "off") {
      const bgSel = document.getElementById("mapBgMode");
      if (bgSel) bgSel.value = "terrain";
      ACTIVE_FIELD_DIMS = new Set();
      ACTIVE_SAE_FID = -1;
      const saeSel = document.getElementById("saeMapSelect");
      if (saeSel) saeSel.value = "-1";
      renderAll();
      return;
    }

    // Turn off SAE overlay when clicking a z_k layer pill
    ACTIVE_SAE_FID = -1;
    const saeSel = document.getElementById("saeMapSelect");
    if (saeSel) saeSel.value = "-1";

    if (dAttr === "all") {
      ACTIVE_FIELD_DIMS = new Set(Array.from({ length: rDim }, (_, i) => i));
    } else {
      const k = parseInt(dAttr, 10);
      if (isMulti) {
        if (ACTIVE_FIELD_DIMS.size >= rDim) {
          // Switching from All rD into multi-select starts with the clicked layer
          ACTIVE_FIELD_DIMS = new Set([k]);
        } else if (ACTIVE_FIELD_DIMS.has(k)) {
          if (ACTIVE_FIELD_DIMS.size > 1) ACTIVE_FIELD_DIMS.delete(k);
        } else {
          ACTIVE_FIELD_DIMS.add(k);
        }
      } else {
        // If clicking a pill k that is already the ONLY active solo pill, toggle back to "All"
        if (ACTIVE_FIELD_DIMS.size === 1 && ACTIVE_FIELD_DIMS.has(k)) {
          ACTIVE_FIELD_DIMS = new Set(Array.from({ length: rDim }, (_, i) => i));
        } else {
          ACTIVE_FIELD_DIMS = new Set([k]);
        }
      }
      // Sync Tab 3 active cell to (z_k, z_k) so the right-hand panel also reflects this layer
      ACTIVE_GRAM_CELL = [k, k];
      ACTIVE_SVD_MODE = null;
      ACTIVE_L2_BASIS = -1;
    }

    // If overlay was turned off ("terrain"), turn it on to P1-P2 Field when a layer pill is clicked
    const bgSel = document.getElementById("mapBgMode");
    if (bgSel && bgSel.value === "terrain") {
      bgSel.value = "diff";
    }

    renderAll();
  });
});

document.getElementById("btnResetView").addEventListener("click", () => {
  mapZoom = 1.0; mapPanX = 0.0; mapPanY = 0.0;
  renderMap();
});

// Map Canvas Click / Drag / Zoom & Unit Deselection
const mapCanvas = document.getElementById("mapCanvas");
let mapMouseDownPos = [0, 0];
let mapDragMoved = false;
let mapPendingTarget = null;

mapCanvas.addEventListener("wheel", e => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.15 : 0.87;
  mapZoom = Math.max(0.7, Math.min(6.0, mapZoom * factor));
  renderMap();
});

mapCanvas.addEventListener("mousedown", async e => {
  if (!FRAME) return;
  const rect = mapCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  dragStartMouse = [mx, my];
  mapMouseDownPos = [mx, my];
  mapDragMoved = false;
  mapPendingTarget = null;

  // Find closest unit or field anchor within 14px
  let bestDist = 14.0;
  let clickedUnit = null;
  let clickedTok = null;

  FRAME.units.forEach(u => {
    const [sx, sy] = worldToCanvas(u.x, u.y, rect.width, rect.height);
    const d = Math.hypot(mx - sx, my - sy);
    if (d < bestDist) {
      bestDist = d;
      clickedUnit = u;
      clickedTok = u.tok_idx;
    }
  });

  if (!clickedUnit && document.getElementById("chkAnchors").checked) {
    FRAME.tokens.filter(t => t.kind === "field").forEach(ft => {
      const [sx, sy] = worldToCanvas(ft.x, ft.y, rect.width, rect.height);
      const d = Math.hypot(mx - sx, my - sy);
      if (d < bestDist) {
        bestDist = d;
        clickedTok = ft.tok_idx;
      }
    });
  }

  if (clickedUnit) {
    if (document.getElementById("chkDragMove").checked) {
      SELECTED_UNIT_ROW = clickedUnit.row;
      SELECTED_TOK_IDX = clickedUnit.tok_idx;
      isDraggingUnit = true;
      await fetchTokenDetails(SELECTED_TOK_IDX);
      renderAll();
    } else {
      mapPendingTarget = { type: "unit", unit: clickedUnit };
    }
  } else if (clickedTok !== null) {
    mapPendingTarget = { type: "tok", tokIdx: clickedTok };
  } else {
    isDraggingMap = true;
    mapPendingTarget = { type: "empty" };
  }
});

mapCanvas.addEventListener("mousemove", e => {
  if (!FRAME) return;
  const rect = mapCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const [wx, wy] = canvasToWorld(mx, my, rect.width, rect.height);

  if (Math.hypot(mx - mapMouseDownPos[0], my - mapMouseDownPos[1]) > 4) {
    mapDragMoved = true;
  }

  if (isDraggingUnit && SELECTED_UNIT_ROW !== null) {
    const u = FRAME.units.find(x => x.row === SELECTED_UNIT_ROW);
    if (u) {
      u.x = Math.round(wx * 10) / 10;
      u.y = Math.round(wy * 10) / 10;
      renderMap();
    }
    return;
  }
  if (isDraggingMap) {
    mapPanX += mx - dragStartMouse[0];
    mapPanY += my - dragStartMouse[1];
    dragStartMouse = [mx, my];
    renderMap();
    return;
  }

  // Hover tooltip
  let nearest = null, bestD = 14.0;
  FRAME.units.forEach(u => {
    const [sx, sy] = worldToCanvas(u.x, u.y, rect.width, rect.height);
    const d = Math.hypot(mx - sx, my - sy);
    if (d < bestD) { bestD = d; nearest = u; }
  });
  const infoEl = document.getElementById("mapCursorInfo");
  const fVal = sampleFieldAtWorld(wx, wy);
  const fStr = fVal !== null && LAST_FIELD_EVAL
    ? ` | ${LAST_FIELD_EVAL.label}=${fVal >= 0 ? "+" : ""}${fVal.toFixed(3)}`
    : "";
  if (nearest) {
    const dims = getSortedActiveDims();
    let zStr = "";
    if (nearest.z) {
      if (dims.length === 1) {
        const zv = nearest.z[dims[0]] || 0;
        zStr = ` | z${dims[0]}=${zv >= 0 ? "+" : ""}${zv.toFixed(3)}`;
      } else {
        zStr = ` | ||z||=${(nearest.mass || 0).toFixed(2)}`;
      }
    }
    const mOwn = nearest.marginal_adv >= 0 ? `+${nearest.marginal_adv.toFixed(2)}` : nearest.marginal_adv.toFixed(2);
    const mP1Val = (nearest.marginal_adv_p1 ?? -nearest.marginal_adv);
    const mP1 = mP1Val >= 0 ? `+${mP1Val.toFixed(2)}` : mP1Val.toFixed(2);
    infoEl.textContent = `(${wx.toFixed(1)}, ${wy.toFixed(1)}) • P${nearest.owner} ${nearest.name} #${nearest.row} | dA_own=${mOwn}‰ (dA₁=${mP1}‰)${zStr}${fStr}`;
  } else {
    infoEl.textContent = `Map (${wx.toFixed(1)}, ${wy.toFixed(1)})${fStr}`;
  }
});

window.addEventListener("mouseup", async e => {
  if (isDraggingUnit && SELECTED_UNIT_ROW !== null && FRAME) {
    isDraggingUnit = false;
    const u = FRAME.units.find(x => x.row === SELECTED_UNIT_ROW);
    if (u) {
      COUNTERFACTUAL.moved_rows[u.row] = [u.x, u.y];
      await loadFrame(CURRENT_GID, CURRENT_LOOP);
    }
  }
  isDraggingMap = false;

  // Click interaction (only when mouse didn't drag)
  if (!mapDragMoved && mapPendingTarget) {
    if (mapPendingTarget.type === "unit") {
      const u = mapPendingTarget.unit;
      if (SELECTED_UNIT_ROW === u.row) {
        deselectUnit();
      } else {
        await selectUnitOrToken(u.row, u.tok_idx);
      }
    } else if (mapPendingTarget.type === "tok") {
      const tok = mapPendingTarget.tokIdx;
      if (SELECTED_TOK_IDX === tok) {
        deselectUnit();
      } else {
        await selectUnitOrToken(null, tok);
      }
    } else if (mapPendingTarget.type === "empty") {
      // Clicked on empty terrain without dragging -> deselect unit
      if (SELECTED_UNIT_ROW !== null || SELECTED_TOK_IDX !== null) {
        deselectUnit();
      }
    }
  }
  mapPendingTarget = null;
});

// Deselection Chip and Button Bindings
const bindClick = (id, fn) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("click", fn);
};

bindClick("btnResetAllSelections", resetAllSelections);
bindClick("selChipUnit", deselectUnit);
bindClick("selChipOverlay", clearOverlaySelection);
bindClick("selChipMode", clearGramModeSelection);
bindClick("btnDeselectUnitLeft", deselectUnit);
bindClick("btnDeselectUnitMap", deselectUnit);
bindClick("btnClearSaeLeft", () => {
  ACTIVE_SAE_FID = -1;
  const s = document.getElementById("saeMapSelect");
  if (s) s.value = "-1";
  renderAll();
});
bindClick("btnClearOverlayMap", clearOverlaySelection);
bindClick("btnGramClearMode", clearGramModeSelection);
bindClick("btnTokDeselect", deselectUnit);

// Escape key to reset all selections
window.addEventListener("keydown", e => {
  if (e.key === "Escape") {
    resetAllSelections();
  }
});

// Timeline Click-to-Seek
["timelineAdvCanvas", "timelineHpCanvas"].forEach(cid => {
  document.getElementById(cid).addEventListener("click", e => {
    if (!TIMELINE) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const padL = 42, padR = 12;
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left - padL) / (rect.width - padL - padR)));
    const idx = Math.min(TIMELINE.loops.length - 1, Math.max(0, Math.round(frac * (TIMELINE.loops.length - 1))));
    loadFrame(CURRENT_GID, TIMELINE.loops[idx]);
  });
});

["saeFeatFilter"].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", () => renderSaeExplorerPanel());
});

["contribOwnerFilter", "contribSortMode"].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", () => renderContributingUnitsPanel());
});

["chkShowNeutrals", "chkDimNeutrals"].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", () => renderMap());
});

window.addEventListener("resize", () => renderAll());
initApp();
