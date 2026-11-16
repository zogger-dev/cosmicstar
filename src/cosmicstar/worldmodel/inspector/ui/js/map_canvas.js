// ============================================================================
// PANEL 1: 2D MAP & BATTLEFIELD CANVAS
// ============================================================================
function worldToCanvas(wx, wy, w, h) {
  const [x0, y0, x1, y1] = FRAME.map.playable;
  const pw = x1 - x0;
  const ph = y1 - y0;
  const scale = Math.min(w / pw, h / ph) * 0.94 * mapZoom;
  const cx = (x0 + x1) * 0.5;
  const cy = (y0 + y1) * 0.5;
  const sx = w * 0.5 + (wx - cx) * scale + mapPanX;
  // Flip Y so North (+y in SC2) is Up on canvas
  const sy = h * 0.5 - (wy - cy) * scale + mapPanY;
  return [sx, sy, scale];
}

function canvasToWorld(sx, sy, w, h) {
  const [x0, y0, x1, y1] = FRAME.map.playable;
  const pw = x1 - x0;
  const ph = y1 - y0;
  const scale = Math.min(w / pw, h / ph) * 0.94 * mapZoom;
  const cx = (x0 + x1) * 0.5;
  const cy = (y0 + y1) * 0.5;
  const wx = cx + (sx - w * 0.5 - mapPanX) / scale;
  const wy = cy - (sy - h * 0.5 - mapPanY) / scale;
  return [wx, wy];
}

function computeUnitGramScores() {
  const scores = {};
  if (!FRAME) return scores;
  const oEl = document.getElementById("gramOwnerSelect");
  const ownerMode = oEl ? oEl.value : "p1";
  if (ACTIVE_L2_BASIS >= 0) {
    FRAME.units.forEach(u => {
      scores[u.row] = u.l2_w_proj[ACTIVE_L2_BASIS] || 0.0;
    });
    return scores;
  }
  if (ACTIVE_SVD_MODE !== null) {
    const ownKey = ownerMode === "p2" ? "p2" : (ownerMode === "neutral" ? "neutral" : "p1");
    const gOwn = FRAME.gram.owners[ownKey];
    const loadings = gOwn.eigenmode_unit_loadings[ACTIVE_SVD_MODE] || [];
    gOwn.unit_rows.forEach((r, idx) => {
      scores[r] = loadings[idx] || 0.0;
    });
    return scores;
  }
  if (isFullGramMode() && GRAM_256_DATA && GRAM_256_DATA.unit_scores) {
    Object.entries(GRAM_256_DATA.unit_scores).forEach(([r, val]) => {
      scores[parseInt(r, 10)] = val;
    });
    return scores;
  }
  const [a, b] = ACTIVE_GRAM_CELL;
  const rDim = getRDim();
  if (is8x8FieldMode()) {
    const ia = Math.min(rDim - 1, a);
    const ib = Math.min(rDim - 1, b);
    FRAME.units.forEach(u => {
      scores[u.row] = ((u.z[ia] || 0) * (u.z[ib] || 0)) / 100.0;
    });
    return scores;
  }
  FRAME.units.forEach(u => {
    const va = a < 8 ? u.q[a] : (u.z[a - 8] || 0);
    const vb = b < 8 ? u.q[b] : (u.z[b - 8] || 0);
    scores[u.row] = (va * vb) / 100.0;
  });
  return scores;
}

function getSortedActiveDims() {
  const rDim = getRDim();
  const arr = Array.from(ACTIVE_FIELD_DIMS).filter(d => d < rDim).sort((a, b) => a - b);
  return arr.length > 0 ? arr : [0];
}

function getActiveSAEFeature() {
  if (!FRAME || !FRAME.map || ACTIVE_SAE_FID < 0) return null;
  if (FRAME.map.sae_features) {
    const f = FRAME.map.sae_features.find(x => x.fid === ACTIVE_SAE_FID);
    if (f) return f;
  }
  if (ACTIVE_SAE_SELECTION && ACTIVE_SAE_SELECTION.fid === ACTIVE_SAE_FID) {
    return ACTIVE_SAE_SELECTION;
  }
  return null;
}

function cleanSaeLabel(lbl) {
  return String(lbl || "").replace(/^F\d+:\s*/, "");
}

function populateSaeMapSelect() {
  const sel = document.getElementById("saeMapSelect");
  if (!sel || !FRAME || !FRAME.map) return;
  const feats = FRAME.map.sae_features || [];
  const prevVal = ACTIVE_SAE_FID;
  sel.innerHTML = `<option value="-1">SAE Feature Overlay: Off (Use z0..z${getRDim() - 1})</option>`;
  feats.forEach(f => {
    const opt = document.createElement("option");
    opt.value = f.fid;
    opt.textContent = `SAE F${String(f.fid).padStart(3, "0")}: ${cleanSaeLabel(f.label)} [${f.n_units}u]`;
    if (f.fid === prevVal) opt.selected = true;
    sel.appendChild(opt);
  });
  if (ACTIVE_SAE_SELECTION && !feats.some(f => f.fid === ACTIVE_SAE_SELECTION.fid)) {
    const opt = document.createElement("option");
    opt.value = ACTIVE_SAE_SELECTION.fid;
    opt.textContent = `[L${ACTIVE_SAE_SELECTION.layer} F${String(ACTIVE_SAE_SELECTION.fid).padStart(3, "0")}] ${cleanSaeLabel(ACTIVE_SAE_SELECTION.label)}`;
    opt.selected = true;
    sel.appendChild(opt);
  }
  if (prevVal >= 0 && !feats.some(f => f.fid === prevVal) && (!ACTIVE_SAE_SELECTION || ACTIVE_SAE_SELECTION.fid !== prevVal)) {
    ACTIVE_SAE_FID = -1;
    sel.value = "-1";
  }
}

function computeFieldOverlayGrid() {
  if (!FRAME || !FRAME.map) return null;
  const mp = FRAME.map;
  const bgMode = document.getElementById("mapBgMode").value;
  if (bgMode === "terrain") {
    LAST_FIELD_EVAL = null;
    return null;
  }

  const G = mp.pot_res || 44;
  const grid = Array.from({ length: G }, () => new Float32Array(G));
  let isDiverging = true;
  let label = "";

  // Case 0: Multi-Scale Stochastic Observer v2 Intrinsic Learned Leverage ||∇_z A||
  if (bgMode.startsWith("threat_")) {
    const netGrid = OBSERVER_DATA && (OBSERVER_DATA.leverage_grid_net || OBSERVER_DATA.threat_grid_net);
    if (!netGrid) {
      LAST_FIELD_EVAL = null;
      return null;
    }
    const p1Grid = OBSERVER_DATA.leverage_grid_p1 || OBSERVER_DATA.threat_grid_p1;
    const p2Grid = OBSERVER_DATA.leverage_grid_p2 || OBSERVER_DATA.threat_grid_p2;
    const src = bgMode === "threat_p1_risk"
      ? p1Grid
      : (bgMode === "threat_p2_risk" ? p2Grid : netGrid);
    isDiverging = (bgMode === "threat_net");
    label = bgMode === "threat_p1_risk"
      ? "P1 Squad Leverage ||∇_z A|| · g_m"
      : (bgMode === "threat_p2_risk" ? "P2 Squad Leverage ||∇_z A|| · g_m" : "Intrinsic Leverage: P1 vs P2 ||∇_z A||");
    let lo = 1e9, hi = -1e9, maxAbs = 1e-6;
    for (let iy = 0; iy < G; iy++) {
      const row = src[iy] || [];
      for (let ix = 0; ix < G; ix++) {
        const v = row[ix] || 0;
        grid[iy][ix] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
      }
    }
    LAST_FIELD_EVAL = { grid, G, isDiverging, lo, hi, maxAbs, label, dims: [], bgMode };
    return LAST_FIELD_EVAL;
  }

  // Case A: Monosemantic Layer-6 Top-K SAE Feature Overlay is selected!
  const saeFeat = getActiveSAEFeature();
  if (saeFeat) {
    const g1 = saeFeat.grid_p1 || [];
    const g2 = saeFeat.grid_p2 || [];
    const ge = saeFeat.grid_env || [];
    const fTag = `SAE F${String(saeFeat.fid).padStart(3, "0")} (${cleanSaeLabel(saeFeat.label)})`;
    if (g1.length > 0 && g2.length > 0) {
      for (let iy = 0; iy < G; iy++) {
        for (let ix = 0; ix < G; ix++) {
          const v1 = (g1[iy] && g1[iy][ix]) || 0;
          const v2 = (g2[iy] && g2[iy][ix]) || 0;
          const ve = (ge[iy] && ge[iy][ix]) || 0;
          if (bgMode === "diff") {
            grid[iy][ix] = v1 - v2;
            isDiverging = true;
            label = `F₁ − F₂ [${fTag}]`;
          } else if (bgMode === "p1") {
            grid[iy][ix] = v1;
            isDiverging = false;
            label = `F₁ [${fTag}]`;
          } else if (bgMode === "p2") {
            grid[iy][ix] = v2;
            isDiverging = false;
            label = `F₂ [${fTag}]`;
          } else if (bgMode === "contest") {
            grid[iy][ix] = v1 * v2;
            isDiverging = false;
            label = `F₁ · F₂ [${fTag}]`;
          } else {
            grid[iy][ix] = ve + v1 + v2;
            isDiverging = false;
            label = `Total [${fTag}]`;
          }
        }
      }
    } else if (FRAME.units) {
      const [x0, y0, x1, y1] = mp.playable || [0, 0, 128, 128];
      const dx = (x1 - x0) / Math.max(1, G - 1);
      const dy = (y1 - y0) / Math.max(1, G - 1);
      const synthTag = `L${saeFeat.layer !== undefined ? saeFeat.layer : 6}·F${String(saeFeat.fid).padStart(3, "0")} (${cleanSaeLabel(saeFeat.label)})`;
      const sigma = 6.0;
      const sigmaSq2 = 2 * sigma * sigma;

      FRAME.units.forEach(u => {
        let act = 0;
        if (u.sae_layers && saeFeat.layer !== undefined && u.sae_layers[saeFeat.layer]) {
          const m = u.sae_layers[saeFeat.layer].find(f => f.fid === saeFeat.fid);
          if (m) act = m.act;
        }
        if (act <= 0.05) return;
        const sign = u.owner === 1 ? 1.0 : (u.owner === 2 ? -1.0 : 0.0);
        const centerIx = Math.round((u.x - x0) / dx);
        const centerIy = Math.round((u.y - y0) / dy);
        const radCells = Math.ceil((sigma * 2.5) / dx);

        for (let iy = Math.max(0, centerIy - radCells); iy <= Math.min(G - 1, centerIy + radCells); iy++) {
          const dY = (iy - centerIy) * dy;
          for (let ix = Math.max(0, centerIx - radCells); ix <= Math.min(G - 1, centerIx + radCells); ix++) {
            const dX = (ix - centerIx) * dx;
            const distSq = dX * dX + dY * dY;
            const w = Math.exp(-distSq / sigmaSq2) * act;
            if (bgMode === "diff") {
              grid[iy][ix] += sign * w;
            } else if (bgMode === "p1") {
              if (u.owner === 1) grid[iy][ix] += w;
            } else if (bgMode === "p2") {
              if (u.owner === 2) grid[iy][ix] += w;
            } else {
              grid[iy][ix] += Math.abs(sign) * w;
            }
          }
        }
      });
      isDiverging = (bgMode === "diff");
      label = `[${synthTag}]`;
    }
    let lo = 1e9, hi = -1e9, maxAbs = 1e-6;
    for (let iy = 0; iy < G; iy++) {
      for (let ix = 0; ix < G; ix++) {
        const v = grid[iy][ix];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
      }
    }
    LAST_FIELD_EVAL = { grid, G, isDiverging, lo, hi, maxAbs, label, dims: [], bgMode };
    return LAST_FIELD_EVAL;
  }

  // Case B: Non-Negative Continuous Field Modes z0..z_{r-1}
  const rDim = getRDim();
  const combineMode = document.getElementById("fieldCombineMode").value;
  const dims = getSortedActiveDims();
  const isSingle = dims.length === 1;
  const k0 = dims[0];
  const dimTag = dims.length === rDim ? `All ${rDim}D` : dims.map(d => `z${d}`).join("+");

  const p1M = mp.field_p1_modes || [];
  const p2M = mp.field_p2_modes || [];
  const envM = mp.field_env_modes || [];
  const pi1M = mp.field_pi1_modes || [];
  const pi2M = mp.field_pi2_modes || [];
  const tvM = mp.pot_z_modes || [];

  for (let iy = 0; iy < G; iy++) {
    for (let ix = 0; ix < G; ix++) {
      let val = 0.0;
      if (bgMode === "diff") {
        if (isSingle) {
          const v1 = p1M[k0] ? p1M[k0][iy][ix] : 0;
          const v2 = p2M[k0] ? p2M[k0][iy][ix] : 0;
          val = v1 - v2;
        } else if (combineMode === "signed") {
          for (const d of dims) {
            val += (p1M[d] ? p1M[d][iy][ix] : 0) - (p2M[d] ? p2M[d][iy][ix] : 0);
          }
        } else {
          let s1 = 0, s2 = 0;
          for (const d of dims) {
            const v1 = p1M[d] ? p1M[d][iy][ix] : 0;
            const v2 = p2M[d] ? p2M[d][iy][ix] : 0;
            s1 += v1 * v1;
            s2 += v2 * v2;
          }
          val = Math.sqrt(s1) - Math.sqrt(s2);
        }
        isDiverging = true;
        label = isSingle ? `F₁[z${k0}] − F₂[z${k0}]` : (combineMode === "signed" ? `Σ(F₁ − F₂) [${dimTag}]` : `||F₁|| − ||F₂|| [${dimTag}]`);
      } else if (bgMode === "p1") {
        if (isSingle || combineMode === "signed") {
          for (const d of dims) val += p1M[d] ? p1M[d][iy][ix] : 0;
          isDiverging = false;
          label = `F₁ [${dimTag}]`;
        } else {
          let s1 = 0;
          for (const d of dims) {
            const v1 = p1M[d] ? p1M[d][iy][ix] : 0;
            s1 += v1 * v1;
          }
          val = Math.sqrt(s1);
          isDiverging = false;
          label = `||F₁|| [${dimTag}]`;
        }
      } else if (bgMode === "p2") {
        if (isSingle || combineMode === "signed") {
          for (const d of dims) val += p2M[d] ? p2M[d][iy][ix] : 0;
          isDiverging = false;
          label = `F₂ [${dimTag}]`;
        } else {
          let s2 = 0;
          for (const d of dims) {
            const v2 = p2M[d] ? p2M[d][iy][ix] : 0;
            s2 += v2 * v2;
          }
          val = Math.sqrt(s2);
          isDiverging = false;
          label = `||F₂|| [${dimTag}]`;
        }
      } else if (bgMode === "contest") {
        for (const d of dims) {
          const v1 = p1M[d] ? p1M[d][iy][ix] : 0;
          const v2 = p2M[d] ? p2M[d][iy][ix] : 0;
          val += v1 * v2;
        }
        isDiverging = false;
        label = `F₁ · F₂ [${dimTag}]`;
      } else if (bgMode === "env") {
        if (isSingle || combineMode === "signed") {
          for (const d of dims) val += envM[d] ? envM[d][iy][ix] : 0;
          isDiverging = false;
          label = `F_env [${dimTag}]`;
        } else {
          let se = 0;
          for (const d of dims) {
            const ve = envM[d] ? envM[d][iy][ix] : 0;
            se += ve * ve;
          }
          val = Math.sqrt(se);
          isDiverging = false;
          label = `||F_env|| [${dimTag}]`;
        }
      } else if (bgMode === "kinetic") {
        if (isSingle && combineMode === "signed") {
          val = (pi1M[k0] ? pi1M[k0][iy][ix] : 0) - (pi2M[k0] ? pi2M[k0][iy][ix] : 0);
          isDiverging = true;
          label = `Π₁[z${k0}] − Π₂[z${k0}]`;
        } else {
          for (const d of dims) {
            const a1 = pi1M[d] ? pi1M[d][iy][ix] : 0;
            const a2 = pi2M[d] ? pi2M[d][iy][ix] : 0;
            val += a1 * a1 + a2 * a2;
          }
          isDiverging = false;
          label = `||Π₁||² + ||Π₂||² [${dimTag}]`;
        }
      } else if (bgMode === "terrain_v") {
        if (isSingle || combineMode === "signed") {
          for (const d of dims) val += tvM[d] ? tvM[d][iy][ix] : 0;
          isDiverging = true;
          label = `V_loc [${dimTag}]`;
        } else {
          let sv = 0;
          for (const d of dims) {
            const vv = tvM[d] ? tvM[d][iy][ix] : 0;
            sv += vv * vv;
          }
          val = Math.sqrt(sv);
          isDiverging = false;
          label = `||V_loc|| [${dimTag}]`;
        }
      }
      grid[iy][ix] = val;
    }
  }

  let lo = 1e9, hi = -1e9, maxAbs = 1e-6;
  for (let iy = 0; iy < G; iy++) {
    for (let ix = 0; ix < G; ix++) {
      const v = grid[iy][ix];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
      if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    }
  }

  LAST_FIELD_EVAL = { grid, G, isDiverging, lo, hi, maxAbs, label, dims, bgMode };
  return LAST_FIELD_EVAL;
}

function sampleFieldAtWorld(wx, wy) {
  if (!LAST_FIELD_EVAL || !FRAME || !FRAME.map) return null;
  const [x0, y0, x1, y1] = FRAME.map.playable;
  const G = LAST_FIELD_EVAL.G;
  const fx = ((wx - x0) / Math.max(1e-5, x1 - x0)) * (G - 1);
  const fy = ((wy - y0) / Math.max(1e-5, y1 - y0)) * (G - 1);
  if (fx < 0 || fx > G - 1 || fy < 0 || fy > G - 1) return null;
  const ix = Math.min(G - 1, Math.max(0, Math.round(fx)));
  const iy = Math.min(G - 1, Math.max(0, Math.round(fy)));
  return LAST_FIELD_EVAL.grid[iy][ix];
}

function updateFieldLayerBar() {
  if (!FRAME || !FRAME.gram || !FRAME.gram.owners) return;
  const rDim = getRDim();
  const dims = getSortedActiveDims();
  const isAll = dims.length === rDim;
  const saeFeat = getActiveSAEFeature();

  const allLbl = document.getElementById("dimPillAllLabel");
  if (allLbl) allLbl.textContent = `All ${rDim}D`;

  const bgMode = document.getElementById("mapBgMode") ? document.getElementById("mapBgMode").value : "diff";
  const isTerrain = bgMode === "terrain";

  const offBtn = document.getElementById("dimPillOff");
  if (offBtn) {
    offBtn.classList.toggle("active", isTerrain);
  }

  // Update active button classes and hide z8..z15 if rDim === 8
  const allBtn = document.getElementById("dimPillAll");
  if (allBtn) allBtn.classList.toggle("active", !isTerrain && !saeFeat && isAll);
  document.querySelectorAll(".dim-pill[data-dim]").forEach(btn => {
    const dAttr = btn.getAttribute("data-dim");
    if (dAttr === "all" || dAttr === "off") return;
    const dIdx = parseInt(dAttr, 10);
    btn.style.display = dIdx < rDim ? "" : "none";
    btn.classList.toggle("active", !isTerrain && !saeFeat && !isAll && ACTIVE_FIELD_DIMS.has(dIdx));
  });

  // Update mini energy bars on z0..z_{r-1} pills using F_self[k,k] (P1 + P2)
  const p1O = FRAME.gram.owners.p1;
  const p2O = FRAME.gram.owners.p2;
  if (p1O && p2O && p1O.F_self_8x8 && p2O.F_self_8x8) {
    let maxE = 1e-6;
    const diagE = [];
    for (let k = 0; k < rDim; k++) {
      const e = ((p1O.F_self_8x8[k] && p1O.F_self_8x8[k][k]) || 0) + ((p2O.F_self_8x8[k] && p2O.F_self_8x8[k][k]) || 0);
      diagE.push(e);
      if (e > maxE) maxE = e;
    }
    for (let k = 0; k < rDim; k++) {
      const fillEl = document.getElementById(`dimFill${k}`);
      if (fillEl) {
        fillEl.style.width = `${Math.round(Math.min(100, (diagE[k] / maxE) * 100))}%`;
      }
    }
  }

  // Update live summary strip for selected dimension(s) or active SAE feature
  const sumEl = document.getElementById("fieldLayerSummary");
  if (!sumEl || !p1O || !p2O) return;

  if (saeFeat) {
    const p1Units = {}, p2Units = {};
    FRAME.units.forEach(u => {
      const act = (saeFeat.unit_acts && saeFeat.unit_acts[u.row]) || 0;
      if (act > 0.05) {
        if (u.owner === 1) p1Units[u.name] = (p1Units[u.name] || 0) + 1;
        else if (u.owner === 2) p2Units[u.name] = (p2Units[u.name] || 0) + 1;
      }
    });
    const fmtU = obj => Object.entries(obj).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k}(${v})`).join(", ") || "—";
    sumEl.innerHTML =
      `<span><b style="color:#c084fc;">[Layer-6 Top-K SAE F${String(saeFeat.fid).padStart(3, "0")}: ${cleanSaeLabel(saeFeat.label)}]</b> ` +
      `Active Units: <b>${saeFeat.n_units}</b> | Total Activation Σf: <b style="color:#10b981;">${saeFeat.total_act}</b> | Peak: <b>${saeFeat.max_act}</b></span>` +
      `<span><b style="color:var(--p1);">P1 Units:</b> ${fmtU(p1Units)} &nbsp;•&nbsp; <b style="color:var(--p2);">P2 Units:</b> ${fmtU(p2Units)}</span>`;
    return;
  }

  let f1Self = 0, f2Self = 0, fCross = 0, fEnv = 0, tField = 0, gradSelf = 0;
  dims.forEach(k => {
    f1Self += (p1O.F_self_8x8 && p1O.F_self_8x8[k] && p1O.F_self_8x8[k][k]) || 0;
    f2Self += (p2O.F_self_8x8 && p2O.F_self_8x8[k] && p2O.F_self_8x8[k][k]) || 0;
    fCross += (p1O.F_cross_8x8 && p1O.F_cross_8x8[k] && p1O.F_cross_8x8[k][k]) || 0;
    gradSelf += (p1O.grad_F_self_8x8 && p1O.grad_F_self_8x8[k] && p1O.grad_F_self_8x8[k][k]) || 0;
    if (p1O.mode_vectors && p2O.mode_vectors) {
      fEnv += (p1O.mode_vectors.F_env[k] || 0) + (p2O.mode_vectors.F_env[k] || 0);
      tField += (p1O.mode_vectors.T_field[k] || 0) + (p2O.mode_vectors.T_field[k] || 0);
    }
  });

  // Compute top P1 and P2 units loading onto the selected non-negative dimension(s)
  const p1Counts = {}, p2Counts = {};
  FRAME.units.forEach(u => {
    if (!u.z) return;
    let score = 0;
    dims.forEach(k => { score += u.z[k] || 0; });
    score /= dims.length;
    if (Math.abs(score) > 0.08) {
      if (u.owner === 1) p1Counts[u.name] = (p1Counts[u.name] || 0) + 1;
      else if (u.owner === 2) p2Counts[u.name] = (p2Counts[u.name] || 0) + 1;
    }
  });
  const fmtTop = obj => Object.entries(obj).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k}(${v})`).join(", ") || "—";

  const dimLabel = isAll ? `All ${rDim}D (z0..z${rDim - 1})` : dims.map(d => `z${d}`).join("+");
  const gradCol = gradSelf >= 0 ? "#10b981" : "#f43f5e";
  sumEl.innerHTML =
    `<span><b style="color:#a5b4fc;">[${dimLabel}]</b> ` +
    `P1 ∫F₁²: <b style="color:var(--p1);">${f1Self.toFixed(4)}</b> | ` +
    `P2 ∫F₂²: <b style="color:var(--p2);">${f2Self.toFixed(4)}</b> | ` +
    `∫F₁·F₂: <b style="color:#c084fc;">${fCross >= 0 ? "+" : ""}${fCross.toFixed(4)}</b> | ` +
    `∫Π²: <b style="color:#fbbf24;">${tField.toFixed(4)}</b> | ` +
    `∂(-A₁)/∂F₁: <b style="color:${gradCol};">${gradSelf >= 0 ? "+" : ""}${gradSelf.toFixed(2)}‰</b></span>` +
    `<span><b style="color:var(--p1);">P1 Units:</b> ${fmtTop(p1Counts)} &nbsp;•&nbsp; <b style="color:var(--p2);">P2 Units:</b> ${fmtTop(p2Counts)}</span>`;
}

function renderMap() {
  const canvas = document.getElementById("mapCanvas");
  const { ctx, w, h } = setupHiDPICanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  if (!FRAME) return;

  updateFieldLayerBar();

  const mp = FRAME.map;
  const [x0, y0, x1, y1] = mp.playable;
  const bgMode = document.getElementById("mapBgMode").value;
  const colorMode = document.getElementById("unitColorMode").value;
  const layerEl = document.getElementById("layerSelect");
  const layerIdx = layerEl ? parseInt(layerEl.value, 10) : 5;

  // 1. ALWAYS Render Base Map Terrain (Height + Pathable + Creep + Fog of War)
  {
    const G = mp.grid_res;
    const dx = (x1 - x0) / G;
    const dy = (y1 - y0) / G;
    for (let iy = 0; iy < G; iy++) {
      for (let ix = 0; ix < G; ix++) {
        const wx = x0 + ix * dx;
        const wy = y0 + (iy + 1) * dy;
        const [sx, sy, sc] = worldToCanvas(wx, wy, w, h);
        const cw = Math.ceil(dx * sc) + 1;
        const ch = Math.ceil(dy * sc) + 1;

        const hVal = mp.height[iy][ix];
        const path = mp.pathable[iy][ix];
        const creep = mp.creep[iy][ix];
        const v1 = mp.p1_vis[iy][ix];
        const v2 = mp.p2_vis[iy][ix];

        let r = path ? 18 + (hVal - 100) * 0.42 : 9;
        let g = path ? 27 + (hVal - 100) * 0.50 : 13;
        let b = path ? 44 + (hVal - 100) * 0.60 : 20;
        if (creep) { r += 34; g -= 5; b += 38; }
        if (v1 > 0) { r += 14; }
        if (v2 > 0) { b += 14; g += 6; }
        ctx.fillStyle = `rgb(${Math.max(6, Math.min(255, r|0))},${Math.max(8, Math.min(255, g|0))},${Math.max(12, Math.min(255, b|0))})`;
        ctx.fillRect(sx, sy, cw, ch);
      }
    }
  }

  // 1b. Render Translucent 2D Potential Field Overlay (Bilinearly Smoothed + Iso-Contours)
  const fev = computeFieldOverlayGrid();
  if (fev) {
    const { grid, G, isDiverging, lo, hi, maxAbs, label } = fev;
    const alphaSlider = document.getElementById("fieldAlphaSlider");
    const overlayAlpha = (alphaSlider ? parseInt(alphaSlider.value, 10) : 72) / 100.0;

    if (!_fieldOffscreenCanvas || _fieldOffscreenCanvas.width !== G || _fieldOffscreenCanvas.height !== G) {
      _fieldOffscreenCanvas = document.createElement("canvas");
      _fieldOffscreenCanvas.width = G;
      _fieldOffscreenCanvas.height = G;
    }
    const octx = _fieldOffscreenCanvas.getContext("2d");
    const imgData = octx.createImageData(G, G);
    const data = imgData.data;

    // Note: canvas Y runs top-to-bottom (y1 down to y0), while grid iy=0 is y0 and iy=G-1 is y1
    for (let iy = 0; iy < G; iy++) {
      const rowFlip = G - 1 - iy;
      for (let ix = 0; ix < G; ix++) {
        const v = grid[iy][ix];
        const pIdx = (rowFlip * G + ix) * 4;
        let r = 0, g = 0, b = 0, a = 0;

        if (isDiverging) {
          const mag = Math.min(1.0, Math.abs(v) / maxAbs);
          if (mag < 0.03) {
            a = 0;
          } else {
            const norm = Math.pow(mag, 0.72);
            if (v >= 0) {
              // Red / Crimson (+ve potential / P1 Red dominance)
              r = Math.round(170 + 82 * norm);
              g = Math.round(28 + 65 * norm);
              b = Math.round(68 + 65 * norm);
            } else {
              // Sky Blue / Cyan (-ve potential / P2 Blue dominance)
              r = Math.round(18 + 38 * norm);
              g = Math.round(95 + 135 * norm);
              b = Math.round(165 + 88 * norm);
            }
            a = Math.round(255 * overlayAlpha * Math.min(1.0, 0.18 + 0.88 * norm));
          }
        } else {
          const t = Math.max(0.0, Math.min(1.0, (v - lo) / Math.max(1e-6, hi - lo)));
          if (t < 0.04) {
            a = 0;
          } else {
            const tp = Math.pow(t, 0.8);
            // Magma / Viridis-like warm density colormap
            r = Math.round(30 + 222 * Math.pow(tp, 1.1));
            g = Math.round(25 + 195 * Math.sin(tp * Math.PI * 0.88));
            b = Math.round(110 + 135 * (1.0 - tp));
            a = Math.round(255 * overlayAlpha * Math.min(1.0, 0.15 + 0.88 * tp));
          }
        }
        data[pIdx] = r;
        data[pIdx + 1] = g;
        data[pIdx + 2] = b;
        data[pIdx + 3] = a;
      }
    }
    octx.putImageData(imgData, 0, 0);

    const [sx0, sy1] = worldToCanvas(x0, y1, w, h);
    const [sx1, sy0] = worldToCanvas(x1, y0, w, h);
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(_fieldOffscreenCanvas, sx0, sy1, sx1 - sx0, sy0 - sy1);

    // Optional Iso-Potential Contour Lines (marching-squares cell edges at +/- 0.38 and +/- 0.72)
    const chkContours = document.getElementById("chkFieldContours");
    if (chkContours && chkContours.checked && maxAbs > 1e-5) {
      const dx = (x1 - x0) / (G - 1);
      const dy = (y1 - y0) / (G - 1);
      const levels = isDiverging
        ? [
            { thr: 0.35 * maxAbs, col: "rgba(253, 164, 175, 0.55)", lw: 1.0 },
            { thr: 0.68 * maxAbs, col: "rgba(255, 228, 230, 0.85)", lw: 1.4 },
            { thr: -0.35 * maxAbs, col: "rgba(125, 211, 252, 0.55)", lw: 1.0 },
            { thr: -0.68 * maxAbs, col: "rgba(224, 242, 254, 0.85)", lw: 1.4 }
          ]
        : [
            { thr: lo + 0.40 * (hi - lo), col: "rgba(253, 224, 71, 0.55)", lw: 1.0 },
            { thr: lo + 0.72 * (hi - lo), col: "rgba(254, 249, 195, 0.85)", lw: 1.4 }
          ];

      levels.forEach(({ thr, col, lw }) => {
        ctx.strokeStyle = col;
        ctx.lineWidth = lw;
        ctx.beginPath();
        for (let iy = 0; iy < G - 1; iy++) {
          for (let ix = 0; ix < G - 1; ix++) {
            const v00 = grid[iy][ix] - thr;
            const v10 = grid[iy][ix + 1] - thr;
            const v01 = grid[iy + 1][ix] - thr;
            const v11 = grid[iy + 1][ix + 1] - thr;
            const s00 = v00 >= 0 ? 1 : 0;
            const s10 = v10 >= 0 ? 1 : 0;
            const s01 = v01 >= 0 ? 1 : 0;
            const s11 = v11 >= 0 ? 1 : 0;
            const mask = s00 | (s10 << 1) | (s11 << 2) | (s01 << 3);
            if (mask === 0 || mask === 15) continue;

            const wx0 = x0 + ix * dx;
            const wy0 = y0 + iy * dy;
            const pts = [];
            if (s00 !== s10) {
              const t = v00 / (v00 - v10);
              pts.push(worldToCanvas(wx0 + t * dx, wy0, w, h));
            }
            if (s10 !== s11) {
              const t = v10 / (v10 - v11);
              pts.push(worldToCanvas(wx0 + dx, wy0 + t * dy, w, h));
            }
            if (s01 !== s11) {
              const t = v01 / (v01 - v11);
              pts.push(worldToCanvas(wx0 + t * dx, wy0 + dy, w, h));
            }
            if (s00 !== s01) {
              const t = v00 / (v00 - v01);
              pts.push(worldToCanvas(wx0, wy0 + t * dy, w, h));
            }
            if (pts.length >= 2) {
              ctx.moveTo(pts[0][0], pts[0][1]);
              ctx.lineTo(pts[1][0], pts[1][1]);
            }
          }
        }
        ctx.stroke();
      });
    }
    ctx.restore();

    // 2D Momentum Flux Quivers (directional arrows showing spatial momentum flow)
    const chkQuiv = document.getElementById("chkMomentumQuivers");
    const showQuivers = chkQuiv && chkQuiv.checked && (bgMode === "kinetic" || bgMode === "diff" || bgMode === "contest");
    if (showQuivers && mp.flux_x && mp.flux_y) {
      const fx = mp.flux_x;
      const fy = mp.flux_y;
      const p1x = mp.flux_p1_x || fx;
      const p1y = mp.flux_p1_y || fy;
      const p2x = mp.flux_p2_x || fx;
      const p2y = mp.flux_p2_y || fy;
      const G = fx.length;

      let maxFlux = 1e-6;
      for (let iy = 0; iy < G; iy++) {
        for (let ix = 0; ix < G; ix++) {
          const m = Math.hypot(fx[iy][ix], fy[iy][ix]);
          if (m > maxFlux) maxFlux = m;
        }
      }

      if (maxFlux > 1e-4) {
        ctx.save();
        const step = G > 30 ? 2 : 1;
        const dx = (x1 - x0) / (G - 1);
        const dy = (y1 - y0) / (G - 1);
        const maxArrowLen = 22.0;

        for (let iy = 0; iy < G; iy += step) {
          for (let ix = 0; ix < G; ix += step) {
            const jx = fx[iy][ix];
            const jy = fy[iy][ix];
            const mag = Math.hypot(jx, jy);
            if (mag < 0.06 * maxFlux) continue;

            const wx = x0 + ix * dx;
            const wy = y0 + iy * dy;
            const [sx, sy] = worldToCanvas(wx, wy, w, h);

            const angle = Math.atan2(-jy, jx);
            const len = Math.max(6, (mag / maxFlux) * maxArrowLen);

            const m1 = Math.hypot(p1x[iy][ix], p1y[iy][ix]);
            const m2 = Math.hypot(p2x[iy][ix], p2y[iy][ix]);
            let arrowCol = "#fde047";
            if (m1 > 1.25 * m2) arrowCol = "#f87171";
            else if (m2 > 1.25 * m1) arrowCol = "#38bdf8";

            ctx.save();
            ctx.translate(sx, sy);
            ctx.rotate(angle);

            ctx.strokeStyle = arrowCol;
            ctx.fillStyle = arrowCol;
            ctx.lineWidth = 1.7;
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(len, 0);
            ctx.stroke();

            const headLen = Math.min(5.5, len * 0.45);
            ctx.beginPath();
            ctx.moveTo(len, 0);
            ctx.lineTo(len - headLen, -headLen * 0.5);
            ctx.lineTo(len - headLen, headLen * 0.5);
            ctx.closePath();
            ctx.fill();

            ctx.restore();
          }
        }
        ctx.restore();
      }
    }

    const legEl = document.getElementById("mapModeLegend");
    if (legEl) {
      legEl.textContent = isDiverging
        ? `${label} • Max |F|=${maxAbs.toFixed(3)} (Red: P1 + | Blue: P2 −)`
        : `${label} • Range [${lo.toFixed(3)}, ${hi.toFixed(3)}]`;
    }
  } else {
    const legEl = document.getElementById("mapModeLegend");
    if (legEl) legEl.textContent = "P1: Red | P2: Blue | NFZ: Magenta";
  }

  // 2. .s2ma NoFlyZone Air Blockers & Healing Shrines
  if (document.getElementById("chkS2MA").checked && mp.s2ma && mp.s2ma.found_s2ma) {
    mp.s2ma.healing_shrines.forEach(sh => {
      const [sx, sy, sc] = worldToCanvas(sh.x, sh.y, w, h);
      ctx.strokeStyle = "#10b981";
      ctx.fillStyle = "rgba(16, 185, 129, 0.22)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx, sy, Math.max(6, 2.5 * sc), 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
    mp.s2ma.no_fly_zones.forEach(nfz => {
      const [sx, sy, sc] = worldToCanvas(nfz.x, nfz.y, w, h);
      ctx.save();
      ctx.strokeStyle = "#e879f9";
      ctx.fillStyle = "rgba(217, 70, 239, 0.16)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(sx, sy, Math.max(5, nfz.r_hard * sc), 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    });
  }

  // 3. 8x8 Continuous Potential Field Anchor Tokens
  if (document.getElementById("chkAnchors").checked) {
    FRAME.tokens.filter(t => t.kind === "field").forEach(ft => {
      const [sx, sy] = worldToCanvas(ft.x, ft.y, w, h);
      const isSel = ft.tok_idx === SELECTED_TOK_IDX;
      ctx.fillStyle = isSel ? "#facc15" : "rgba(148, 163, 184, 0.45)";
      ctx.beginPath();
      ctx.arc(sx, sy, isSel ? 5 : 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // 3b. Dynamic Perceiver Squad Tokens (M=64 Smooth Queries, Intrinsic Leverage ||∇_z A||, Shimmer)
  const chkSquads = document.getElementById("chkSquadTokens");
  if (((chkSquads && chkSquads.checked) || bgMode.startsWith("threat_")) && OBSERVER_DATA && OBSERVER_DATA.squads) {
    ctx.save();
    let maxLev = 1e-3;
    OBSERVER_DATA.squads.forEach(sq => {
      const lv = sq.leverage !== undefined ? sq.leverage : (sq.risk || 0);
      if (lv > maxLev) maxLev = lv;
    });
    OBSERVER_DATA.squads.forEach((sq, sqIdx) => {
      const [sx, sy, sc] = worldToCanvas(sq.x, sq.y, w, h);
      const rx = Math.max(5, Math.min(48, sq.spread_x * sc));
      const ry = Math.max(5, Math.min(48, sq.spread_y * sc));
      const isP1 = sq.owner === 1;
      const hiddenInFog = sq.vis_by_opp < 0.35;
      const hasShimmer = (sq.shimmer_by_opp || 0) >= 0.15;
      const lev = sq.leverage !== undefined ? sq.leverage : (sq.risk || 0);
      const isHighLev = sqIdx < 6 || lev >= Math.max(25, 0.38 * maxLev);

      if (hasShimmer) {
        ctx.save();
        ctx.strokeStyle = "rgba(250, 204, 21, 0.85)";
        ctx.lineWidth = 1.6;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.ellipse(sx, sy, rx + 3, ry + 3, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      ctx.strokeStyle = isP1
        ? (hiddenInFog ? "rgba(248, 113, 113, 0.58)" : "rgba(252, 165, 165, 0.94)")
        : (hiddenInFog ? "rgba(96, 165, 250, 0.58)" : "rgba(147, 197, 253, 0.94)");
      ctx.lineWidth = isHighLev ? 2.1 : 1.2;
      ctx.setLineDash(hiddenInFog ? [3, 3] : []);
      ctx.beginPath();
      ctx.ellipse(sx, sy, rx, ry, 0, 0, Math.PI * 2);
      ctx.stroke();

      // Squad centroid diamond
      ctx.setLineDash([]);
      ctx.fillStyle = isP1 ? "#ef4444" : "#3b82f6";
      ctx.strokeStyle = hasShimmer ? "#fde047" : "#f8fafc";
      ctx.lineWidth = hasShimmer ? 1.5 : 1.0;
      ctx.beginPath();
      ctx.moveTo(sx, sy - 4);
      ctx.lineTo(sx + 4, sy);
      ctx.lineTo(sx, sy + 4);
      ctx.lineTo(sx - 4, sy);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      if (isHighLev && lev > 1.0) {
        const shortU = (sqIdx < 4 && sq.units) ? ` ${String(sq.units).split(",")[0].slice(0, 12)}` : "";
        const tag = `⚡${Math.round(lev)}${shortU}`;
        ctx.font = "bold 9px ui-monospace, monospace";
        const tw = ctx.measureText(tag).width + 6;
        ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
        ctx.fillRect(sx - tw * 0.5, sy - ry - 13, tw, 12);
        ctx.strokeStyle = isP1 ? "#f87171" : "#60a5fa";
        ctx.lineWidth = 1;
        ctx.strokeRect(sx - tw * 0.5, sy - ry - 13, tw, 12);
        ctx.fillStyle = "#fde047";
        ctx.textAlign = "center";
        ctx.fillText(tag, sx, sy - ry - 4);
      }
    });
    ctx.restore();
  }

  // 4. Spatial 2D-RoPE Attention Rays from SELECTED_TOK_IDX
  if (document.getElementById("chkAttnRays").checked && TOKEN_DETAILS && TOKEN_DETAILS.outgoing) {
    const srcTok = FRAME.tokens.find(t => t.tok_idx === SELECTED_TOK_IDX);
    if (srcTok) {
      const [xA, yA] = worldToCanvas(srcTok.x, srcTok.y, w, h);
      TOKEN_DETAILS.outgoing.forEach(edge => {
        if (edge.tok_idx === SELECTED_TOK_IDX) return;
        const dstTok = FRAME.tokens.find(t => t.tok_idx === edge.tok_idx);
        if (!dstTok) return;
        const [xB, yB] = worldToCanvas(dstTok.x, dstTok.y, w, h);
        const alpha = Math.min(0.95, Math.max(0.12, edge.weight * 4.5));
        ctx.strokeStyle = dstTok.kind === "field"
          ? `rgba(250, 204, 21, ${alpha})`
          : `rgba(167, 139, 250, ${alpha})`;
        ctx.lineWidth = Math.min(5.0, 0.8 + edge.weight * 14.0);
        ctx.beginPath();
        ctx.moveTo(xA, yA);
        ctx.lineTo(xB, yB);
        ctx.stroke();
      });
    }
  }

  // 5. Render Living Units
  const gramScores = colorMode === "gram_mode" ? computeUnitGramScores() : {};
  let maxGramAbs = 1e-6;
  Object.values(gramScores).forEach(v => { if (Math.abs(v) > maxGramAbs) maxGramAbs = Math.abs(v); });

  let maxMargAbs = 1e-5;
  let maxMass = 1e-5;
  FRAME.units.forEach(u => {
    if ((u.owner === 1 || u.owner === 2) && Math.abs(u.marginal_adv) > maxMargAbs) maxMargAbs = Math.abs(u.marginal_adv);
    if ((u.owner === 1 || u.owner === 2) && (u.mass || 0) > maxMass) maxMass = u.mass;
  });
  if (maxMargAbs <= 1e-5) {
    FRAME.units.forEach(u => { if (Math.abs(u.marginal_adv) > maxMargAbs) maxMargAbs = Math.abs(u.marginal_adv); });
  }
  if (maxMass <= 1e-5) {
    FRAME.units.forEach(u => { if ((u.mass || 0) > maxMass) maxMass = u.mass; });
  }

  const showForceAdv = document.getElementById("chkForceAdv").checked;
  const showForceKin = document.getElementById("chkForceKin").checked;
  const showNeutrals = document.getElementById("chkShowNeutrals") ? document.getElementById("chkShowNeutrals").checked : true;
  const dimNeutrals = document.getElementById("chkDimNeutrals") ? document.getElementById("chkDimNeutrals").checked : true;
  const saeFeat = getActiveSAEFeature();

  FRAME.units.forEach(u => {
    const isNeutral = u.owner !== 1 && u.owner !== 2;
    const isSel = u.row === SELECTED_UNIT_ROW || (SELECTED_TOK_IDX !== null && u.tok_idx === SELECTED_TOK_IDX);
    const isGroupSel = SELECTED_GROUP_ROWS && SELECTED_GROUP_ROWS.has(u.row);

    // SAE contributing unit activation across any layer
    let saeAct = 0.0;
    if (saeFeat) {
      if (saeFeat.unit_acts && saeFeat.unit_acts[u.row] !== undefined) {
        saeAct = saeFeat.unit_acts[u.row];
      } else if (u.sae_layers && saeFeat.layer !== undefined && u.sae_layers[saeFeat.layer]) {
        const lf = u.sae_layers[saeFeat.layer].find(f => f.fid === saeFeat.fid);
        if (lf) saeAct = lf.act;
      }
    }
    const isSaeContributor = saeFeat && saeAct > 0.05;

    // Filter out neutral minerals if user unchecked "Minerals"
    if (isNeutral && !showNeutrals && !isSel && !isSaeContributor && !isGroupSel) {
      return;
    }

    const [sx, sy, sc] = worldToCanvas(u.x, u.y, w, h);
    const baseRad = u.radius * sc * (u.structure ? 1.15 : (isNeutral ? 1.05 : 1.35));
    const rad = Math.max(2.6, Math.min(13.0, baseRad));

    ctx.save();
    // Dim non-contributing units when an SAE feature or unit group is isolated
    if (SELECTED_GROUP_ROWS && SELECTED_GROUP_ROWS.size > 0 && !isGroupSel && !isSel) {
      ctx.globalAlpha = 0.28;
    } else if (saeFeat && !isSaeContributor && !isSel && !isGroupSel) {
      ctx.globalAlpha = 0.30;
    } else if (isNeutral && dimNeutrals && !isSel && !isSaeContributor && !isGroupSel) {
      // Subtly dim background neutral mineral fields and rocks so player armies pop out
      ctx.globalAlpha = 0.38;
    }
    if (MICRO_STEP > 0 && !isNeutral && !u.structure && !isSel) {
      ctx.globalAlpha *= 0.24;
    }

    // Prominent golden halo for units in the currently selected Unit/Structure group
    if (isGroupSel) {
      ctx.strokeStyle = "#facc15";
      ctx.fillStyle = "rgba(250, 204, 21, 0.28)";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      if (u.structure) {
        ctx.rect(sx - rad - 3, sy - rad - 3, (rad + 3) * 2, (rad + 3) * 2);
      } else {
        ctx.arc(sx, sy, rad + 4, 0, Math.PI * 2);
      }
      ctx.fill();
      ctx.stroke();
    }

    // Prominent glowing ring for units contributing to the active SAE feature
    if (isSaeContributor) {
      const actNorm = Math.min(1.0, saeAct / Math.max(0.1, saeFeat.max_act || 1.0));
      ctx.strokeStyle = "#c084fc";
      ctx.fillStyle = `rgba(192, 132, 252, ${0.15 + 0.30 * actNorm})`;
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.arc(sx, sy, rad + 3 + 6 * actNorm, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // Marginal advantage or mass halo
    if (colorMode === "owner" || colorMode === "marginal") {
      const mFrac = Math.min(1.0, Math.abs(u.marginal_adv) / maxMargAbs);
      if (mFrac > 0.10) {
        let haloCol = "rgba(148, 163, 184, 0.2)";
        if (u.owner === 1) {
          haloCol = u.marginal_adv >= 0 ? `rgba(239, 68, 68, ${0.15 + 0.35 * mFrac})` : `rgba(59, 130, 246, ${0.15 + 0.35 * mFrac})`;
        } else if (u.owner === 2) {
          haloCol = u.marginal_adv >= 0 ? `rgba(59, 130, 246, ${0.15 + 0.35 * mFrac})` : `rgba(239, 68, 68, ${0.15 + 0.35 * mFrac})`;
        }
        ctx.fillStyle = haloCol;
        ctx.beginPath();
        ctx.arc(sx, sy, rad + 2 + 6 * mFrac, 0, Math.PI * 2);
        ctx.fill();
      }
    } else if (colorMode === "mass") {
      const mFrac = Math.min(1.0, (u.mass || 0) / maxMass);
      ctx.fillStyle = u.owner === 1
        ? `rgba(239, 68, 68, ${0.18 + 0.38 * mFrac})`
        : (u.owner === 2 ? `rgba(59, 130, 246, ${0.18 + 0.38 * mFrac})` : `rgba(251, 191, 36, ${0.15 + 0.3 * mFrac})`);
      ctx.beginPath();
      ctx.arc(sx, sy, rad + 1 + 8 * mFrac, 0, Math.PI * 2);
      ctx.fill();
    }

    // Unit fill color based on selected mode
    let fillCol = "#fbbf24";
    const isMutedNeutral = isNeutral && dimNeutrals && !isSel && !isSaeContributor && !isGroupSel;

    if (isMutedNeutral && colorMode === "owner") {
      fillCol = "rgba(217, 119, 6, 0.35)";
    } else if (colorMode === "owner") {
      fillCol = u.owner === 1 ? "#ef4444" : (u.owner === 2 ? "#3b82f6" : "#fbbf24");
    } else if (colorMode === "field_dim") {
      let zScore = 0;
      if (saeFeat && saeFeat.unit_acts) {
        zScore = (saeFeat.unit_acts[u.row] || 0) / Math.max(0.25, saeFeat.max_act || 1.0);
      } else if (u.z) {
        const dims = getSortedActiveDims();
        dims.forEach(d => { zScore += u.z[d] || 0; });
        zScore /= dims.length;
        zScore *= 1.8;
      }
      const t = Math.max(-1, Math.min(1, zScore));
      if (Math.abs(t) < 0.05) {
        fillCol = "rgba(100, 116, 139, 0.40)";
      } else if (u.owner === 1) {
        fillCol = `rgb(240, ${Math.round(40 + 60*(1-Math.abs(t)))}, ${Math.round(40 + 60*(1-Math.abs(t)))})`;
      } else if (u.owner === 2) {
        fillCol = `rgb(${Math.round(40 + 60*(1-Math.abs(t)))}, ${Math.round(100 + 80*(1-Math.abs(t)))}, 250)`;
      } else {
        fillCol = `rgb(250, ${Math.round(170 + 70*Math.abs(t))}, 40)`;
      }
    } else if (colorMode === "mass") {
      const t = Math.min(1.0, (u.mass || 0) / maxMass);
      fillCol = u.owner === 1
        ? `rgb(${Math.round(200 + 50*t)}, ${Math.round(50 + 50*t)}, 50)`
        : (u.owner === 2 ? `rgb(${Math.round(50 + 50*t)}, ${Math.round(100 + 80*t)}, ${Math.round(200 + 50*t)})` : "rgba(217, 119, 6, 0.4)");
    } else if (colorMode === "marginal") {
      const t = Math.max(-1, Math.min(1, u.marginal_adv / maxMargAbs));
      if (u.owner === 1) {
        fillCol = t >= 0 ? `rgb(239, 68, 68)` : `rgb(59, 130, 246)`;
      } else if (u.owner === 2) {
        fillCol = t >= 0 ? `rgb(59, 130, 246)` : `rgb(239, 68, 68)`;
      } else {
        fillCol = "rgb(251, 191, 36)";
      }
    } else if (colorMode === "gram_mode") {
      const gv = gramScores[u.row] || 0.0;
      const t = Math.max(-1, Math.min(1, gv / maxGramAbs));
      fillCol = Math.abs(t) < 0.05
        ? "rgba(100, 116, 139, 0.35)"
        : (t >= 0 ? `rgba(16, 185, 129, ${0.35 + 0.65*t})` : `rgba(244, 63, 94, ${0.35 - 0.65*t})`);
    } else if (colorMode === "pca_rgb") {
      const rgb = (u.layer_pca_rgb && u.layer_pca_rgb[layerIdx + 1]) || [160, 160, 160];
      fillCol = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    } else if (colorMode === "cos_sim" && TOKEN_DETAILS && TOKEN_DETAILS.cosine_similarity) {
      const cs = TOKEN_DETAILS.cosine_similarity[u.tok_idx] || 0.0;
      const t = Math.max(0, Math.min(1, (cs + 0.2) / 1.2));
      fillCol = `rgb(${Math.round(40 + 210*t)}, ${Math.round(60 + 140*t)}, ${Math.round(240 - 120*t)})`;
    }

    ctx.fillStyle = fillCol;
    ctx.strokeStyle = isSel
      ? "#ffffff"
      : (isGroupSel
        ? "#facc15"
        : (isSaeContributor
          ? "#c084fc"
          : (isMutedNeutral
            ? "rgba(180, 83, 9, 0.35)"
            : (u.flying ? "#e879f9" : "rgba(15, 23, 42, 0.85)"))));
    ctx.lineWidth = isSel ? 2.8 : (isGroupSel ? 2.2 : (isSaeContributor ? 2.0 : (isMutedNeutral ? 0.7 : (u.flying ? 1.5 : 1.0))));

    ctx.beginPath();
    if (u.structure) {
      ctx.rect(sx - rad, sy - rad, rad * 2, rad * 2);
    } else {
      ctx.arc(sx, sy, rad, 0, Math.PI * 2);
    }
    ctx.fill();
    ctx.stroke();

    // Force vectors (-grad_{x,y} A1)
    if (showForceAdv && u.owner !== 0 && !u.structure) {
      const [fx, fy] = u.force_adv;
      // For P1, +force_adv increases P1 advantage; for P2, -force_adv increases P2 advantage
      const sgn = u.owner === 1 ? 1.0 : -1.0;
      const vx = fx * sgn * 320.0;
      const vy = -fy * sgn * 320.0; // flip Y for canvas
      const mag = Math.hypot(vx, vy);
      if (mag > 2.5) {
        const clampMag = Math.min(26.0, mag);
        const ex = sx + (vx / mag) * clampMag;
        const ey = sy + (vy / mag) * clampMag;
        ctx.strokeStyle = u.owner === 1 ? "#fca5a5" : "#93c5fd";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
      }
    }

    if (showForceKin && u.owner !== 0 && !u.structure) {
      const [kx, ky] = u.force_kin;
      const vx = kx * 180.0;
      const vy = -ky * 180.0;
      const mag = Math.hypot(vx, vy);
      if (mag > 2.0) {
        const clampMag = Math.min(22.0, mag);
        ctx.strokeStyle = "#fde047";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(sx + (vx / mag) * clampMag, sy + (vy / mag) * clampMag);
        ctx.stroke();
      }
    }
    ctx.restore();
  });

  // 6. L1ChunkFlowDecoder +5.0s (16 x 0.3125s) Tactical Micro Trajectories, Orders, Casualties & Spawns
  const mc = OBSERVER_DATA && OBSERVER_DATA.micro_chunk;
  if (mc && mc.units && MICRO_STEP > 0) {
    ctx.save();
    const kMax = MICRO_STEP > 0 ? Math.min(mc.n_steps - 1, MICRO_STEP - 1) : (mc.n_steps - 1);

    // 6a. Movement polyline trails over +5.0s
    mc.units.forEach(mu => {
      if (mu.structure) return;
      const dxTot = Math.hypot((mu.x[kMax] || mu.x0) - mu.x0, (mu.y[kMax] || mu.y0) - mu.y0);
      if (dxTot < 1.1) return;
      ctx.strokeStyle = mu.owner === 1
        ? (MICRO_STEP > 0 ? "rgba(252, 165, 165, 0.72)" : "rgba(248, 113, 113, 0.36)")
        : (MICRO_STEP > 0 ? "rgba(147, 197, 253, 0.72)" : "rgba(96, 165, 250, 0.36)");
      ctx.lineWidth = MICRO_STEP > 0 ? 1.35 : 0.95;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      const [sx0, sy0] = worldToCanvas(mu.x0, mu.y0, w, h);
      ctx.moveTo(sx0, sy0);
      for (let j = 0; j <= kMax; j++) {
        const [px, py] = worldToCanvas(mu.x[j], mu.y[j], w, h);
        ctx.lineTo(px, py);
      }
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // 6b. Animated entity positions, tactical order arrows, casualties & predicted spawns at step k
    if (MICRO_STEP > 0) {
      const k = kMax;
      mc.units.forEach(mu => {
        const [ux, uy, sc] = worldToCanvas(mu.x[k], mu.y[k], w, h);
        const rad = Math.max(2.8, Math.min(11.5, (mu.radius || 0.5) * sc * (mu.structure ? 1.1 : 1.35)));
        const aliveP = mu.alive_prob ? mu.alive_prob[k] : 1.0;
        const isDead = aliveP < 0.35;

        if (isDead) {
          ctx.strokeStyle = mu.owner === 1 ? "rgba(248, 113, 113, 0.75)" : "rgba(96, 165, 250, 0.75)";
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(ux - 3.5, uy - 3.5);
          ctx.lineTo(ux + 3.5, uy + 3.5);
          ctx.moveTo(ux + 3.5, uy - 3.5);
          ctx.lineTo(ux - 3.5, uy + 3.5);
          ctx.stroke();
          return;
        }

        // Tactical unit order target ray
        const ordP = mu.ord_prob ? mu.ord_prob[k] : 0.0;
        if (!mu.structure && ordP > 0.55 && mu.ord_tx && mu.ord_ty) {
          const [tx, ty] = worldToCanvas(mu.ord_tx[k], mu.ord_ty[k], w, h);
          const dist = Math.hypot(tx - ux, ty - uy);
          if (dist > 4.0) {
            const clampD = Math.min(34.0, dist);
            const ex = ux + ((tx - ux) / dist) * clampD;
            const ey = uy + ((ty - uy) / dist) * clampD;
            ctx.strokeStyle = mu.owner === 1 ? "rgba(254, 202, 202, 0.75)" : "rgba(191, 219, 254, 0.75)";
            ctx.lineWidth = 1.15;
            ctx.beginPath();
            ctx.moveTo(ux, uy);
            ctx.lineTo(ex, ey);
            ctx.stroke();
          }
        }

        ctx.globalAlpha = Math.max(0.45, Math.min(1.0, aliveP));
        ctx.fillStyle = mu.owner === 1 ? "#ef4444" : "#3b82f6";
        ctx.strokeStyle = mu.flying ? "#f0abfc" : "#f8fafc";
        ctx.lineWidth = mu.flying ? 1.6 : 1.1;
        ctx.beginPath();
        if (mu.structure) {
          ctx.rect(ux - rad, uy - rad, rad * 2, rad * 2);
        } else {
          ctx.arc(ux, uy, rad, 0, Math.PI * 2);
        }
        ctx.fill();
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      });

      // Predicted newly spawned / warped-in units
      (mc.spawns || []).forEach(sp => {
        if (sp.birth_k > k) return;
        const [ox, oy] = worldToCanvas(sp.origin_x, sp.origin_y, w, h);
        const [sx, sy, sc] = worldToCanvas(sp.x[k], sp.y[k], w, h);
        const rad = Math.max(3.5, Math.min(10.0, (sp.radius || 0.5) * sc * 1.4));
        ctx.save();
        ctx.strokeStyle = sp.owner === 1 ? "rgba(252, 165, 165, 0.7)" : "rgba(147, 197, 253, 0.7)";
        ctx.setLineDash([2, 2]);
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(sx, sy);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = sp.owner === 1 ? "rgba(239, 68, 68, 0.88)" : "rgba(59, 130, 246, 0.88)";
        ctx.strokeStyle = "#fde047";
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.arc(sx, sy, rad + 1.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.font = "bold 8.8px ui-monospace, monospace";
        ctx.fillStyle = "#fde047";
        ctx.textAlign = "center";
        ctx.fillText(`+${sp.name}`, sx, sy - rad - 4);
        ctx.restore();
      });
    }
    ctx.restore();
  }

  // 7. Highlighted Unscouted Threat Target Reticle (from Theory-of-Mind Pill Click)
  if (HIGHLIGHT_THREAT_XY) {
    const [hx, hy] = worldToCanvas(HIGHLIGHT_THREAT_XY.x, HIGHLIGHT_THREAT_XY.y, w, h);
    ctx.save();
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.arc(hx, hy, 18, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(hx, hy, 26, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(hx - 23, hy); ctx.lineTo(hx - 12, hy);
    ctx.moveTo(hx + 12, hy); ctx.lineTo(hx + 23, hy);
    ctx.moveTo(hx, hy - 23); ctx.lineTo(hx, hy - 12);
    ctx.moveTo(hx, hy + 12); ctx.lineTo(hx, hy + 23);
    ctx.stroke();

    const lbl = `P${HIGHLIGHT_THREAT_XY.viewer} Blind: ${HIGHLIGHT_THREAT_XY.units}`;
    ctx.font = "bold 10px ui-monospace, monospace";
    const tw = ctx.measureText(lbl).width + 10;
    ctx.fillStyle = "rgba(15, 23, 42, 0.92)";
    ctx.fillRect(hx - tw * 0.5, hy - 42, tw, 15);
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 1.2;
    ctx.strokeRect(hx - tw * 0.5, hy - 42, tw, 15);
    ctx.fillStyle = "#fde047";
    ctx.textAlign = "center";
    ctx.fillText(lbl, hx, hy - 31);
    ctx.restore();
  }

  const s2maBadge = mp.s2ma && mp.s2ma.found_s2ma
    ? ` | .s2ma NFZ:${mp.s2ma.no_fly_zones.length} Shrines:${mp.s2ma.healing_shrines.length}`
    : "";
  const microBadge = MICRO_STEP > 0 && mc
    ? ` | Micro +${((mc.step_secs && mc.step_secs[MICRO_STEP - 1]) || (MICRO_STEP * 0.3125)).toFixed(2)}s (${OBSERVER_MODE})`
    : "";
  document.getElementById("mapOverlayTop").textContent =
    `${mp.map_name} • t=${fmtTime(FRAME.time_s)} (Loop ${FRAME.loop_t}) • Live Units: ${FRAME.units.length}${microBadge}${s2maBadge}`;
}
