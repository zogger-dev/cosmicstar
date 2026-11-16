// ============================================================================
// RENDER ORCHESTRATION
// ============================================================================
function renderAll() {
  if (!FRAME) return;
  renderKPIs();
  renderSelectionBar();
  renderSaeExplorerPanel();
  renderContributingUnitsPanel();
  renderMap();
  renderTimeline();
  renderHUDAndForces();
  renderObserverPanel();
  if (SELECTED_GROUP_KEY && SELECTED_GROUP_ROWS.size > 0) {
    const parts = SELECTED_GROUP_KEY.split("_");
    const pKey = parts[0];
    const uName = parts.slice(1).join("_");
    const aliveRows = FRAME.units.filter(u => SELECTED_GROUP_ROWS.has(u.row)).map(u => u.row);
    if (aliveRows.length > 0) {
      renderReverseSaeAttribution(pKey, uName, aliveRows);
    } else {
      SELECTED_GROUP_KEY = null;
      SELECTED_GROUP_ROWS.clear();
      renderReverseSaeAttribution(null, null, []);
    }
  }
}

function summarizeSaeFeatureOnFrame(f) {
  if (!FRAME || !FRAME.units || !f || !f.unit_acts) {
    return { liveShort: cleanSaeLabel(f ? f.label : ""), p1Str: "—", p2Str: "—", neuStr: "—", list: [] };
  }
  const byTypeAct = {}, byTypeCnt = {};
  const p1Act = {}, p1Cnt = {};
  const p2Act = {}, p2Cnt = {};
  const neuAct = {}, neuCnt = {};
  const list = [];
  let totalWt = 0.0;
  FRAME.units.forEach(u => {
    const act = f.unit_acts[u.row] || 0;
    if (act > 1e-4) {
      const wt = (f.unit_weights && f.unit_weights[u.row] !== undefined)
        ? f.unit_weights[u.row]
        : (u.mass || 1.0) * act;
      totalWt += wt;
      byTypeAct[u.name] = (byTypeAct[u.name] || 0) + act;
      byTypeCnt[u.name] = (byTypeCnt[u.name] || 0) + 1;
      if (u.owner === 1) {
        p1Act[u.name] = (p1Act[u.name] || 0) + act;
        p1Cnt[u.name] = (p1Cnt[u.name] || 0) + 1;
      } else if (u.owner === 2) {
        p2Act[u.name] = (p2Act[u.name] || 0) + act;
        p2Cnt[u.name] = (p2Cnt[u.name] || 0) + 1;
      } else {
        neuAct[u.name] = (neuAct[u.name] || 0) + act;
        neuCnt[u.name] = (neuCnt[u.name] || 0) + 1;
      }
      list.push({ u, act, wt });
    }
  });
  // Sort unit types by sum of activation so high-activation structures dominate
  const fmtMap = (actObj, cntObj, lim = 3) =>
    Object.entries(actObj)
      .sort((a, b) => b[1] - a[1])
      .slice(0, lim)
      .map(([k, sumAct]) => {
        const cnt = cntObj[k] || 1;
        return cnt > 1 ? `${k}×${cnt}(${sumAct.toFixed(1)})` : `${k}(${sumAct.toFixed(1)})`;
      })
      .join(", ");
  const liveShort = fmtMap(byTypeAct, byTypeCnt, 2) || cleanSaeLabel(f.label);
  return {
    liveShort,
    p1Str: fmtMap(p1Act, p1Cnt, 3) || "—",
    p2Str: fmtMap(p2Act, p2Cnt, 3) || "—",
    neuStr: fmtMap(neuAct, neuCnt, 2) || "—",
    totalWt: Math.max(1e-6, totalWt),
    list
  };
}

function renderSelectionBar() {
  if (!FRAME) return;
  const rDim = getRDim();
  const dims = getSortedActiveDims();
  const isAllDims = dims.length === rDim;
  const saeFeat = getActiveSAEFeature();
  const bgMode = document.getElementById("mapBgMode").value;

  // 1. Unit / Token chip
  const chipUnit = document.getElementById("selChipUnit");
  const btnULeft = document.getElementById("btnDeselectUnitLeft");
  const btnUMap = document.getElementById("btnDeselectUnitMap");
  if (SELECTED_UNIT_ROW !== null) {
    const u = FRAME.units.find(x => x.row === SELECTED_UNIT_ROW);
    chipUnit.textContent = u ? `Unit: P${u.owner} ${u.name} #${u.row} ✕` : `Unit: #${SELECTED_UNIT_ROW} ✕`;
    chipUnit.classList.add("active");
    if (btnULeft) btnULeft.style.display = "";
    if (btnUMap) btnUMap.style.display = "";
  } else if (SELECTED_TOK_IDX !== null && SELECTED_TOK_IDX >= 0) {
    const t = FRAME.tokens.find(x => x.tok_idx === SELECTED_TOK_IDX);
    chipUnit.textContent = t ? `Token: ${t.label} ✕` : `Token #${SELECTED_TOK_IDX} ✕`;
    chipUnit.classList.add("active");
    if (btnULeft) btnULeft.style.display = "";
    if (btnUMap) btnUMap.style.display = "";
  } else {
    chipUnit.textContent = "Unit: None";
    chipUnit.classList.remove("active");
    if (btnULeft) btnULeft.style.display = "none";
    if (btnUMap) btnUMap.style.display = "none";
  }

  // 2. Overlay chip (SAE or Field Layer or Off)
  const chipOvl = document.getElementById("selChipOverlay");
  const btnSaeLeft = document.getElementById("btnClearSaeLeft");
  const btnOvlMap = document.getElementById("btnClearOverlayMap");
  if (bgMode === "terrain") {
    chipOvl.textContent = "Field: Off ✕";
    chipOvl.classList.add("active");
    if (btnSaeLeft) btnSaeLeft.style.display = "none";
    if (btnOvlMap) btnOvlMap.style.display = "";
  } else if (saeFeat) {
    const fCode = `F${String(saeFeat.fid).padStart(3, "0")}`;
    chipOvl.textContent = `SAE: ${fCode} [${saeFeat.n_units}u] ✕`;
    chipOvl.classList.add("active");
    if (btnSaeLeft) btnSaeLeft.style.display = "";
    if (btnOvlMap) btnOvlMap.style.display = "";
  } else if (!isAllDims) {
    chipOvl.textContent = `Layer: ${dims.map(d => "z" + d).join("+")} ✕`;
    chipOvl.classList.add("active");
    if (btnSaeLeft) btnSaeLeft.style.display = "none";
    if (btnOvlMap) btnOvlMap.style.display = "";
  } else {
    chipOvl.textContent = `Field: All ${rDim}D`;
    chipOvl.classList.remove("active");
    if (btnSaeLeft) btnSaeLeft.style.display = "none";
    if (btnOvlMap) btnOvlMap.style.display = "none";
  }

  // 3. Gram / SVD Mode chip
  const chipMode = document.getElementById("selChipMode");
  const btnGramClr = document.getElementById("btnGramClearMode");
  if (ACTIVE_SVD_MODE !== null) {
    chipMode.style.display = "inline-flex";
    chipMode.textContent = `SVD: σ${ACTIVE_SVD_MODE} ✕`;
    chipMode.classList.add("active");
    if (btnGramClr) btnGramClr.style.display = "";
  } else if (ACTIVE_L2_BASIS >= 0) {
    chipMode.style.display = "inline-flex";
    chipMode.textContent = `L2: W_c[${ACTIVE_L2_BASIS}] ✕`;
    chipMode.classList.add("active");
    if (btnGramClr) btnGramClr.style.display = "";
  } else if (ACTIVE_GRAM_CELL !== null) {
    const lbls = is8x8FieldMode() ? (FRAME.gram.labels_8 || []) : FRAME.gram.labels_16;
    const [ia, ib] = ACTIVE_GRAM_CELL;
    chipMode.style.display = "inline-flex";
    chipMode.textContent = `Cell: (${lbls[ia] || ia},${lbls[ib] || ib}) ✕`;
    chipMode.classList.add("active");
    if (btnGramClr) btnGramClr.style.display = "";
  } else {
    chipMode.style.display = "none";
    chipMode.classList.remove("active");
    if (btnGramClr) btnGramClr.style.display = "none";
  }
}

function renderSaeExplorerPanel() {
  const tbody = document.getElementById("saeFeatureTableBody");
  const badge = document.getElementById("saeFrameBadge");
  if (!tbody || !FRAME || !FRAME.map) return;
  const feats = FRAME.map.sae_features || [];
  const r2Arr = FRAME.map.sae_layer_r2 || [];
  const l6R2 = r2Arr.length > 0 ? ` • R²=${(r2Arr[r2Arr.length - 1] * 100).toFixed(0)}%` : "";

  const featFilter = document.getElementById("saeFeatFilter") ? document.getElementById("saeFeatFilter").value : "players";
  let displayFeats = feats;
  if (featFilter === "players") {
    displayFeats = feats.filter(f => (f.player_act !== undefined ? f.player_act >= 0.5 : (f.is_player !== false)));
    if (displayFeats.length === 0) displayFeats = feats;
  } else if (featFilter === "resources") {
    displayFeats = feats.filter(f => (f.player_act !== undefined ? f.player_act < 0.5 : (f.is_player === false)));
  }

  if (badge) badge.textContent = `${displayFeats.length} / ${feats.length} active${l6R2}`;

  tbody.innerHTML = "";
  if (displayFeats.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="color:var(--text-secondary);text-align:center;padding:12px;">No matching SAE features on this frame.</td></tr>`;
    return;
  }

  let maxTotW = 1e-6;
  displayFeats.forEach(f => {
    const w = (f.player_act !== undefined && f.player_act > 0) ? f.player_act : (f.total_weight !== undefined ? f.total_weight : f.total_act);
    if (w > maxTotW) maxTotW = w;
  });

  displayFeats.forEach(f => {
    const isSel = f.fid === ACTIVE_SAE_FID;
    const tr = document.createElement("tr");
    if (isSel) tr.classList.add("selected-row");
    const fCode = `F${String(f.fid).padStart(3, "0")}`;
    const info = summarizeSaeFeatureOnFrame(f);
    const corpusLbl = cleanSaeLabel(f.label);
    const isPly = f.is_player !== false && (f.player_act === undefined || f.player_act >= 0.5);
    const wVal = (f.player_act !== undefined && f.player_act > 0) ? f.player_act : (f.total_weight !== undefined ? f.total_weight : f.total_act);
    const wPct = Math.round(Math.min(100, (wVal / maxTotW) * 100));
    const tagBadge = isPly
      ? `<span style="font-size:9px; background:rgba(56,189,248,0.18); border:1px solid #38bdf8; border-radius:3px; padding:0 3px; color:#bae6fd;">Player</span>`
      : `<span style="font-size:9px; background:rgba(251,191,36,0.14); border:1px solid #d97706; border-radius:3px; padding:0 3px; color:#fde68a;">Resource</span>`;

    tr.innerHTML = `
      <td style="font-weight:700; color:${isSel ? '#c084fc' : '#a5b4fc'}; white-space:nowrap;">
        ${fCode}${isSel ? ' <span style="color:#f87171;">✕</span>' : ''}
      </td>
      <td>
        <div style="font-weight:600; color:#f1f5f9; display:flex; align-items:center; gap:4px;">
          <span>${info.liveShort}</span>
          ${tagBadge}
        </div>
        <div style="font-size:9.5px; color:var(--text-muted);">Corpus: ${corpusLbl}</div>
      </td>
      <td style="white-space:nowrap;">${f.n_units}u</td>
      <td style="white-space:nowrap;">
        <span>${wVal.toFixed(1)}</span>
        <div class="bar-mini-wrap"><div class="bar-mini-fill" style="width:${wPct}%;"></div></div>
      </td>
    `;
    tr.addEventListener("click", () => {
      if (ACTIVE_SAE_FID === f.fid) {
        // Toggle off (deselect)
        ACTIVE_SAE_FID = -1;
        const sel = document.getElementById("saeMapSelect");
        if (sel) sel.value = "-1";
      } else {
        ACTIVE_SAE_FID = f.fid;
        const sel = document.getElementById("saeMapSelect");
        if (sel) sel.value = String(f.fid);
        const bgSel = document.getElementById("mapBgMode");
        if (bgSel && bgSel.value === "terrain") bgSel.value = "diff";
      }
      renderAll();
    });
    tbody.appendChild(tr);
  });
}

function renderContributingUnitsPanel() {
  const tbody = document.getElementById("contribUnitsTableBody");
  const titleEl = document.getElementById("contribUnitsTitle");
  const badgeEl = document.getElementById("contribCountBadge");
  const sumEl = document.getElementById("contribTypeSummary");
  const colActEl = document.getElementById("contribColAct");
  const colWtEl = document.getElementById("contribColWt");
  if (!tbody || !FRAME) return;

  const ownFilterVal = document.getElementById("contribOwnerFilter").value;
  const sortMode = document.getElementById("contribSortMode").value;
  const isPlayerFilter = ownFilterVal === "players";
  const ownFilter = (ownFilterVal === "all" || ownFilterVal === "players") ? null : parseInt(ownFilterVal, 10);

  const saeFeat = getActiveSAEFeature();
  const rDim = getRDim();
  const dims = getSortedActiveDims();
  const isAllDims = dims.length === rDim;
  const bgMode = document.getElementById("mapBgMode").value;

  let rows = [];
  let maxAct = 1e-6;
  let totalWt = 1e-6;

  if (saeFeat) {
    // Mode 1: Units contributing to selected Layer-6 SAE Feature
    const fCode = `F${String(saeFeat.fid).padStart(3, "0")}`;
    const info = summarizeSaeFeatureOnFrame(saeFeat);
    if (titleEl) titleEl.textContent = `Units in SAE ${fCode}`;
    if (colActEl) colActEl.textContent = "SAE Act f_u";
    if (colWtEl) colWtEl.textContent = "Field m·f (%)";

    totalWt = info.totalWt;
    info.list.forEach(({ u, act, wt }) => {
      if (isPlayerFilter && u.owner !== 1 && u.owner !== 2) return;
      if (ownFilter !== null && u.owner !== ownFilter) return;
      if (act > maxAct) maxAct = act;
      rows.push({ u, act, wt, pct: (wt / totalWt) * 100.0, tag: "" });
    });

    if (badgeEl) badgeEl.textContent = `${rows.length} / ${saeFeat.n_units}u`;
    if (sumEl) {
      sumEl.innerHTML =
        `<div><b style="color:#c084fc;">SAE ${fCode} Live Contributors</b> <span style="color:var(--text-muted)">(Corpus prototype: ${cleanSaeLabel(saeFeat.label)})</span></div>` +
        `<div style="margin-top:2px;"><b style="color:var(--p1);">P1:</b> ${info.p1Str} &nbsp;•&nbsp; <b style="color:var(--p2);">P2:</b> ${info.p2Str}` +
        (info.neuStr !== "—" ? ` &nbsp;•&nbsp; <b style="color:#fbbf24;">Neu:</b> ${info.neuStr}` : "") +
        `</div>`;
    }
  } else if (!isAllDims && bgMode !== "terrain") {
    // Mode 2: Units contributing to isolated Field Layer(s) z_k
    const dimTag = dims.map(d => `z${d}`).join("+");
    if (titleEl) titleEl.textContent = `Units in Layer ${dimTag}`;
    if (colActEl) colActEl.textContent = `Load ${dimTag}`;
    if (colWtEl) colWtEl.textContent = "Field m·z (%)";

    const p1C = {}, p2C = {};
    FRAME.units.forEach(u => {
      if (!u.z) return;
      let zMean = 0.0;
      dims.forEach(k => { zMean += u.z[k] || 0; });
      zMean /= dims.length;
      if (zMean > 0.02) {
        const wt = (u.mass || 1.0) * zMean;
        totalWt += wt;
        if (u.owner === 1) p1C[u.name] = (p1C[u.name] || 0) + 1;
        else if (u.owner === 2) p2C[u.name] = (p2C[u.name] || 0) + 1;
        if (isPlayerFilter && u.owner !== 1 && u.owner !== 2) return;
        if (ownFilter !== null && u.owner !== ownFilter) return;
        if (zMean > maxAct) maxAct = zMean;
        rows.push({ u, act: zMean, wt, pct: 0, tag: "" });
      }
    });
    rows.forEach(r => { r.pct = (r.wt / totalWt) * 100.0; });
    const fmtM = obj => Object.entries(obj).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k}(${v})`).join(", ") || "—";
    if (badgeEl) badgeEl.textContent = `${rows.length}u`;
    if (sumEl) {
      sumEl.innerHTML =
        `<div><b style="color:#a5b4fc;">Field Layer [${dimTag}] Contributors</b> (z_k &gt; 0.02)</div>` +
        `<div style="margin-top:2px;"><b style="color:var(--p1);">P1:</b> ${fmtM(p1C)} &nbsp;•&nbsp; <b style="color:var(--p2);">P2:</b> ${fmtM(p2C)}</div>`;
    }
  } else {
    // Mode 3: All live units on map (with their top Layer-6 SAE feature badge!)
    if (titleEl) titleEl.textContent = isPlayerFilter ? `Player Units on Map (P1+P2)` : `Live Units on Map`;
    if (colActEl) colActEl.textContent = "Top L6 SAE";
    if (colWtEl) colWtEl.textContent = "Mass m_u";

    FRAME.units.forEach(u => {
      if (isPlayerFilter && u.owner !== 1 && u.owner !== 2) return;
      if (ownFilter !== null && u.owner !== ownFilter) return;
      const topL6 = (u.sae_layers && u.sae_layers[6] && u.sae_layers[6][0]) || null;
      const act = topL6 ? topL6.act : 0.0;
      const wt = u.mass || 0.0;
      totalWt += wt;
      if (act > maxAct) maxAct = act;
      rows.push({ u, act, wt, pct: 0, topL6 });
    });
    rows.forEach(r => { r.pct = (r.wt / totalWt) * 100.0; });
    if (badgeEl) badgeEl.textContent = `${rows.length}u`;
    if (sumEl) {
      sumEl.innerHTML = isPlayerFilter
        ? `Showing <b>${rows.length}</b> player army &amp; economy units. Switch filter to "All Units" to inspect neutral resources.`
        : `Showing all <b>${rows.length}</b> live units on map. Click any SAE feature above or field layer <b>z0..z${rDim - 1}</b> to filter, or click a unit to inspect.`;
    }
  }

  // Sort rows
  rows.sort((a, b) => {
    if (sortMode === "act") return b.act - a.act;
    if (sortMode === "marginal") return Math.abs(b.u.marginal_adv) - Math.abs(a.u.marginal_adv);
    if (sortMode === "marginal_p1") return Math.abs(b.u.marginal_adv_p1 ?? b.u.marginal_adv) - Math.abs(a.u.marginal_adv_p1 ?? a.u.marginal_adv);
    if (sortMode === "mass") return (b.u.mass || 0) - (a.u.mass || 0);
    return b.wt - a.wt;
  });

  tbody.innerHTML = "";
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="color:var(--text-secondary);text-align:center;padding:12px;">No matching units on this frame.</td></tr>`;
    return;
  }

  rows.slice(0, 80).forEach(r => {
    const { u, act, wt, pct, topL6 } = r;
    const isSel = u.row === SELECTED_UNIT_ROW;
    const tr = document.createElement("tr");
    if (isSel) tr.classList.add("selected-row");
    const ownCol = u.owner === 1 ? "var(--p1)" : (u.owner === 2 ? "var(--p2)" : "#fbbf24");
    const ownTxt = u.owner === 1 ? "P1" : (u.owner === 2 ? "P2" : "Neu");
    const barPct = Math.round(Math.min(100, (act / maxAct) * 100));

    let actCellHtml = "";
    if (!saeFeat && (isAllDims || bgMode === "terrain") && topL6) {
      const fCode = `F${String(topL6.fid).padStart(3, "0")}`;
      actCellHtml = `<span onclick="event.stopPropagation(); window.selectSaeFeatureOnMap(${topL6.fid});" title="Click to isolate SAE ${fCode} on map" style="cursor:pointer; color:#c084fc; font-weight:600; text-decoration:underline dotted;">${fCode} (${act.toFixed(1)})</span>`;
    } else {
      actCellHtml = `<span>${act.toFixed(2)}</span><div class="bar-mini-wrap"><div class="bar-mini-fill" style="width:${barPct}%;"></div></div>`;
    }

    const wtCellHtml = (!saeFeat && (isAllDims || bgMode === "terrain"))
      ? `${wt.toFixed(2)}`
      : `${wt.toFixed(2)} <span style="color:var(--text-muted);font-size:10px;">(${pct.toFixed(0)}%)</span>`;

    const mAdv = (sortMode === "marginal_p1") ? (u.marginal_adv_p1 ?? u.marginal_adv) : u.marginal_adv;
    let mCol = "#94a3b8";
    if (sortMode === "marginal_p1") {
      mCol = mAdv >= 0 ? "var(--p1)" : "var(--p2)";
    } else {
      if (u.owner === 1) {
        mCol = mAdv >= 0 ? "var(--p1)" : "var(--p2)";
      } else if (u.owner === 2) {
        mCol = mAdv >= 0 ? "var(--p2)" : "var(--p1)";
      } else {
        mCol = "#fbbf24";
      }
    }
    const mSign = mAdv >= 0 ? "+" : "";
    const mTitle = (sortMode === "marginal_p1")
      ? `Global dA₁ (P1 perspective): ${mSign}${mAdv.toFixed(2)}‰`
      : (u.owner === 1
          ? `P1 Unit: ${mSign}${mAdv.toFixed(2)}‰ to P1 Advantage`
          : (u.owner === 2
              ? `P2 Unit: ${mSign}${mAdv.toFixed(2)}‰ to P2 Advantage (Global dA₁: ${(-mAdv >= 0 ? "+" : "") + (-mAdv).toFixed(2)}‰)`
              : `Neutral: ${mSign}${mAdv.toFixed(2)}‰`));

    tr.innerHTML = `
      <td style="white-space:nowrap; font-weight:${isSel ? '700' : '400'};">
        ${u.name} <span style="color:var(--text-muted)">#${u.row}</span>${isSel ? ' <span style="color:#f87171;">✕</span>' : ''}
      </td>
      <td style="color:${ownCol}; font-weight:600;">${ownTxt}</td>
      <td style="white-space:nowrap;">${actCellHtml}</td>
      <td style="white-space:nowrap;">${wtCellHtml}</td>
      <td style="color:${mCol}; white-space:nowrap; font-weight:600;" title="${mTitle}">
        ${mSign}${mAdv.toFixed(2)}
      </td>
      <td style="color:var(--text-secondary); max-width:75px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
        ${u.order_abil.split("/").pop()}
      </td>
    `;
    tr.addEventListener("click", async () => {
      if (SELECTED_UNIT_ROW === u.row) {
        deselectUnit();
      } else {
        await selectUnitOrToken(u.row, u.tok_idx);
      }
    });
    tbody.appendChild(tr);
  });
}

function renderKPIs() {
  const s = FRAME.scalars;
  const v1 = s.V1 !== undefined ? s.V1 : s.H1;
  const v2 = s.V2 !== undefined ? s.V2 : s.H2;
  const t1 = s.T1 !== undefined ? s.T1 : 0.0;
  const t2 = s.T2 !== undefined ? s.T2 : 0.0;
  document.getElementById("kpiH1").innerHTML =
    `${s.H1 >= 0 ? "+" : ""}${s.H1.toFixed(1)}‰ <span style="font-size:10px;font-weight:400;color:var(--text-secondary)">(V:${v1.toFixed(1)} T:${t1.toFixed(1)})</span>`;
  document.getElementById("kpiH2").innerHTML =
    `${s.H2 >= 0 ? "+" : ""}${s.H2.toFixed(1)}‰ <span style="font-size:10px;font-weight:400;color:var(--text-secondary)">(V:${v2.toFixed(1)} T:${t2.toFixed(1)})</span>`;

  const elA1 = document.getElementById("kpiA1");
  const pLead = s.A1 >= 0 ? "P1" : "P2";
  const aMag = Math.abs(s.A1).toFixed(1);
  elA1.textContent = `${s.A1 >= 0 ? "+" : "-"}${aMag}‰ (${pLead})`;
  elA1.style.color = s.A1 >= 0 ? "var(--p1)" : "var(--p2)";

  const tug2 = 2.0 * s.tug;
  document.getElementById("kpiTug").textContent = `${tug2 >= 0 ? "+" : ""}${tug2.toFixed(1)}‰`;
  document.getElementById("kpiDiss").textContent = `${s.diss1.toFixed(2)} / ${s.diss2.toFixed(2)}`;
  document.getElementById("kpiGoalCos").textContent =
    `${s.u1_active ? s.goal_cos1.toFixed(2) : "idle"} / ${s.u2_active ? s.goal_cos2.toFixed(2) : "idle"}`;
}
