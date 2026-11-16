// ============================================================================
// INITIALIZATION & API CALLS
// ============================================================================
async function initApp() {
  const res = await fetch("/api/init");
  const data = await res.json();
  CATALOG = data.games;

  const gSel = document.getElementById("gameSelect");
  gSel.innerHTML = "";
  const grpUnseen = document.createElement("optgroup");
  grpUnseen.label = "Unseen Validation Games (5.0.14)";
  const grpTrain = document.createElement("optgroup");
  grpTrain.label = "Training Games (5.0.14)";

  CATALOG.forEach(g => {
    const opt = document.createElement("option");
    opt.value = g.gid;
    const p1n = unescapeHtml(g.p1_name);
    const p2n = unescapeHtml(g.p2_name);
    const winTag = g.winner === 1 ? `P1 ${p1n}` : (g.winner === 2 ? `P2 ${p2n}` : "Draw");
    opt.textContent = `[${g.map}] ${p1n} (${g.p1_race[0]}) vs ${p2n} (${g.p2_race[0]}) — ${fmtTime(g.duration_s)} (Win: ${winTag})`;
    if (g.subset === "unseen_games") grpUnseen.appendChild(opt);
    else grpTrain.appendChild(opt);
  });
  gSel.appendChild(grpUnseen);
  gSel.appendChild(grpTrain);

  const l1Sel = document.getElementById("l1CkptSelect");
  l1Sel.innerHTML = "";
  data.checkpoints.l1.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p;
    const parts = p.split("/");
    opt.textContent = `L1: ${parts[parts.length - 2].replace("energy_", "")}/${parts[parts.length - 1]}`;
    if (p === data.active_l1_ckpt) opt.selected = true;
    l1Sel.appendChild(opt);
  });

  if (CATALOG.length > 0) {
    CURRENT_GID = CATALOG[0].gid;
    CURRENT_LOOP = Math.min(2500, Math.floor(CATALOG[0].duration_loops * 0.45));
    document.getElementById("loopInput").value = CURRENT_LOOP;
    await loadGameTimeline(CURRENT_GID);
    await loadFrame(CURRENT_GID, CURRENT_LOOP);
  }
}

async function loadGameTimeline(gid) {
  const mode = document.getElementById("l2ModeSelect").value;
  const res = await fetch(`/api/game/${gid}/timeline?mode=${mode}`);
  TIMELINE = await res.json();
  renderTimeline();
}

function hasCounterfactuals() {
  return (
    COUNTERFACTUAL.deleted_rows.length > 0 ||
    Object.keys(COUNTERFACTUAL.moved_rows).length > 0 ||
    Object.keys(COUNTERFACTUAL.hp_scale).length > 0
  );
}

let isLoadingFrame = false;
let pendingFrameReq = null;
let OBSERVER_DATA = null;
let OBSERVER_MODE = "forecast";
let MICRO_STEP = 0;
let microPlayTimer = null;
let HIGHLIGHT_THREAT_XY = null;
let isLoadingObserver = false;
let pendingObserverReq = null;

function stopMicroPlayback() {
  if (microPlayTimer) {
    clearInterval(microPlayTimer);
    microPlayTimer = null;
  }
  const btn = document.getElementById("btnMicroPlay");
  if (btn) btn.textContent = "▶ +5s Micro";
}

function updateMicroStepUI() {
  const sl = document.getElementById("microStepSlider");
  const lbl = document.getElementById("microStepLabel");
  if (sl) sl.value = String(MICRO_STEP);
  if (!lbl) return;
  if (MICRO_STEP <= 0 || !OBSERVER_DATA || !OBSERVER_DATA.micro_chunk) {
    const nSp = (OBSERVER_DATA && OBSERVER_DATA.micro_chunk && OBSERVER_DATA.micro_chunk.spawns)
      ? OBSERVER_DATA.micro_chunk.spawns.length
      : 0;
    lbl.textContent = nSp > 0 ? `Live τ₀ (${nSp} sp)` : "Live τ₀";
    lbl.style.color = "#94a3b8";
  } else {
    const mc = OBSERVER_DATA.micro_chunk;
    const k = Math.min(mc.n_steps - 1, MICRO_STEP - 1);
    const dt = (mc.step_secs && mc.step_secs[k]) ? mc.step_secs[k] : ((k + 1) * 0.3125);
    const activeSp = (mc.spawns || []).filter(s => s.birth_k <= k).length;
    lbl.textContent = activeSp > 0 ? `+${dt.toFixed(2)}s • +${activeSp}u` : `+${dt.toFixed(2)}s (f${k + 1})`;
    lbl.style.color = "#fde047";
  }
}

async function loadObserverRollout(gid, loop_t, modeOverride = null) {
  if (modeOverride) {
    OBSERVER_MODE = modeOverride;
  }
  document.querySelectorAll(".obs-mode-btn[data-obs-mode]").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-obs-mode") === OBSERVER_MODE);
  });
  if (isLoadingObserver) {
    pendingObserverReq = [gid, loop_t, OBSERVER_MODE];
    return;
  }
  isLoadingObserver = true;
  const badge = document.getElementById("obsStatusBadge");
  if (badge) {
    badge.textContent = `rolling +120s (${OBSERVER_MODE === "forecast" ? "fc" : (OBSERVER_MODE === "counterplay_p1" ? "cp1" : "cp2")})…`;
    badge.style.color = "#fbbf24";
  }
  try {
    const res = await fetch(`/api/game/${gid}/observer/${loop_t}?branches=32&horizon=24&mode=${encodeURIComponent(OBSERVER_MODE)}`);
    const data = await res.json();
    if (data && data.available) {
      OBSERVER_DATA = data;
      if (badge) {
        const mTag = data.mode === "counterplay_p1" ? "CP1" : (data.mode === "counterplay_p2" ? "CP2" : "FC");
        badge.textContent = `${mTag} ${data.n_branches}b • ${Math.round(data.elapsed_ms)}ms`;
        badge.style.color = "#a5b4fc";
      }
      updateMicroStepUI();
      renderObserverPanel();
      const bgSel = document.getElementById("mapBgMode");
      const sqChk = document.getElementById("chkSquadTokens");
      if ((bgSel && bgSel.value.startsWith("threat_")) || (sqChk && sqChk.checked) || MICRO_STEP > 0 || HIGHLIGHT_THREAT_XY) {
        renderMap();
      }
    }
  } catch (err) {
    console.warn("Observer rollout fetch failed:", err);
  } finally {
    isLoadingObserver = false;
    if (pendingObserverReq) {
      const [nGid, nLoop, nMode] = pendingObserverReq;
      pendingObserverReq = null;
      loadObserverRollout(nGid, nLoop, nMode);
    }
  }
}

async function loadFrame(gid, loop_t) {
  if (isLoadingFrame) {
    pendingFrameReq = [gid, loop_t];
    return;
  }
  isLoadingFrame = true;
  const badge = document.getElementById("latencyBadge");
  badge.style.borderColor = "#f59e0b";
  badge.style.color = "#fbbf24";

  try {
    const winLen = parseInt(document.getElementById("winLenSelect").value, 10);
    let res;
    if (hasCounterfactuals()) {
      res = await fetch(`/api/game/${gid}/frame/${loop_t}?window=${winLen}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(COUNTERFACTUAL)
      });
    } else {
      res = await fetch(`/api/game/${gid}/frame/${loop_t}?window=${winLen}`);
    }
    FRAME = await res.json();
    CURRENT_LOOP = FRAME.loop_t;
    document.getElementById("loopInput").value = CURRENT_LOOP;
    badge.textContent = `${Math.round(FRAME.elapsed_ms)} ms`;
    badge.style.borderColor = "var(--border-solid)";
    badge.style.color = "#10b981";

    // Keep selected unit row if still alive; if deselected or dead, leave deselected!
    const aliveRows = new Set(FRAME.units.map(u => u.row));
    if (SELECTED_UNIT_ROW !== null && !aliveRows.has(SELECTED_UNIT_ROW)) {
      SELECTED_UNIT_ROW = null;
      SELECTED_TOK_IDX = null;
    } else if (SELECTED_UNIT_ROW !== null) {
      const u = FRAME.units.find(x => x.row === SELECTED_UNIT_ROW);
      if (u) SELECTED_TOK_IDX = u.tok_idx;
    }

    HIGHLIGHT_THREAT_XY = null;
    MICRO_STEP = 0;
    stopMicroPlayback();
    updateMicroStepUI();

    populateTokenSelect();
    populateSaeMapSelect();
    await fetchTokenDetails(SELECTED_TOK_IDX);
    if (isFullGramMode()) {
      await fetchGram256();
    }
    renderAll();
    loadObserverRollout(gid, CURRENT_LOOP);
  } finally {
    isLoadingFrame = false;
    if (pendingFrameReq) {
      const [nextGid, nextLoop] = pendingFrameReq;
      pendingFrameReq = null;
      loadFrame(nextGid, nextLoop);
    }
  }
}

async function fetchTokenDetails(tokIdx) {
  if (tokIdx === null || tokIdx === undefined || tokIdx < 0) {
    TOKEN_DETAILS = null;
    return;
  }
  const lEl = document.getElementById("layerSelect");
  const hEl = document.getElementById("headSelect");
  const layer = lEl ? parseInt(lEl.value, 10) : 6;
  const head = hEl ? parseInt(hEl.value, 10) : 0;
  const res = await fetch(`/api/token/${tokIdx}?layer=${layer}&head=${head}`);
  TOKEN_DETAILS = await res.json();
}

function isFullGramMode() {
  const el = document.getElementById("gramMatType");
  if (!el) return false;
  const mt = el.value;
  return (
    mt === "G_cc_active" ||
    mt === "G_cc_256" ||
    mt === "G_cc_cov" ||
    mt === "G_cc_clustered" ||
    mt === "K_uu" ||
    mt === "K_spatial" ||
    mt === "G_qc_8x256"
  );
}

function is8x8FieldMode() {
  const el = document.getElementById("gramMatType");
  if (!el) return false;
  const mt = el.value;
  return (
    mt === "F_self_8x8" ||
    mt === "grad_F_self" ||
    mt === "contrib_F_self" ||
    mt === "F_cross_8x8" ||
    mt === "grad_F_cross" ||
    mt === "contrib_F_cross"
  );
}

async function fetchGram256() {
  const oEl = document.getElementById("gramOwnerSelect");
  const mEl = document.getElementById("gramMatType");
  if (!oEl || !mEl) return;
  const owner = oEl.value;
  const mt = mEl.value;
  const [ci, cj] = ACTIVE_GRAM256_CELL;
  const res = await fetch(`/api/gram256?owner=${owner}&mat_type=${mt}&i=${ci}&j=${cj}`);
  GRAM_256_DATA = await res.json();
}

function populateTokenSelect() {
  const sel = document.getElementById("tokenSelect");
  if (!sel) return;
  sel.innerHTML = "";
  const noneOpt = document.createElement("option");
  noneOpt.value = "-1";
  noneOpt.textContent = "— None (Deselected) —";
  if (SELECTED_TOK_IDX === null || SELECTED_TOK_IDX < 0) noneOpt.selected = true;
  sel.appendChild(noneOpt);
  if (!FRAME) return;
  FRAME.tokens.forEach(t => {
    const opt = document.createElement("option");
    opt.value = t.tok_idx;
    const ownTag = t.owner === 1 ? "P1 " : (t.owner === 2 ? "P2 " : "");
    opt.textContent = `${ownTag}${t.label}`;
    if (t.tok_idx === SELECTED_TOK_IDX) opt.selected = true;
    sel.appendChild(opt);
  });
}
