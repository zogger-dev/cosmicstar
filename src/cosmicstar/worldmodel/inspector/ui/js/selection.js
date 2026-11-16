// ============================================================================
// DESELECTION & SELECTION HELPERS
// ============================================================================
async function selectUnitOrToken(unitRow, tokIdx) {
  SELECTED_UNIT_ROW = unitRow;
  SELECTED_TOK_IDX = tokIdx;
  const tSel = document.getElementById("tokenSelect");
  if (tSel) tSel.value = tokIdx !== null ? String(tokIdx) : "-1";
  await fetchTokenDetails(SELECTED_TOK_IDX);
  renderAll();
}

function deselectUnit() {
  SELECTED_UNIT_ROW = null;
  SELECTED_TOK_IDX = null;
  TOKEN_DETAILS = null;
  const tSel = document.getElementById("tokenSelect");
  if (tSel) tSel.value = "-1";
  const cMode = document.getElementById("unitColorMode");
  if (cMode && cMode.value === "cos_sim") cMode.value = "owner";
  renderAll();
}

function clearOverlaySelection() {
  const rDim = getRDim();
  ACTIVE_SAE_FID = -1;
  const saeSel = document.getElementById("saeMapSelect");
  if (saeSel) saeSel.value = "-1";
  ACTIVE_FIELD_DIMS = new Set(Array.from({ length: rDim }, (_, i) => i));
  const bgSel = document.getElementById("mapBgMode");
  if (bgSel && bgSel.value === "terrain") bgSel.value = "diff";
  HIGHLIGHT_THREAT_XY = null;
  MICRO_STEP = 0;
  stopMicroPlayback();
  updateMicroStepUI();
  renderAll();
}

function clearGramModeSelection() {
  ACTIVE_GRAM_CELL = null;
  ACTIVE_SVD_MODE = null;
  ACTIVE_L2_BASIS = -1;
  const l2Sel = document.getElementById("l2BasisSelect");
  if (l2Sel) l2Sel.value = "-1";
  const cMode = document.getElementById("unitColorMode");
  if (cMode && cMode.value === "gram_mode") cMode.value = "owner";
  const badge = document.getElementById("gramCellBadge");
  if (badge) badge.textContent = "Click cell (a,b) or bar σ_k to project (click again to deselect)";
  renderAll();
}

function resetAllSelections() {
  SELECTED_UNIT_ROW = null;
  SELECTED_TOK_IDX = null;
  TOKEN_DETAILS = null;
  const tSel = document.getElementById("tokenSelect");
  if (tSel) tSel.value = "-1";
  ACTIVE_SAE_FID = -1;
  const saeSel = document.getElementById("saeMapSelect");
  if (saeSel) saeSel.value = "-1";
  const rDim = getRDim();
  ACTIVE_FIELD_DIMS = new Set(Array.from({ length: rDim }, (_, i) => i));
  ACTIVE_GRAM_CELL = null;
  ACTIVE_SVD_MODE = null;
  ACTIVE_L2_BASIS = -1;
  const l2Sel = document.getElementById("l2BasisSelect");
  if (l2Sel) l2Sel.value = "-1";
  const cMode = document.getElementById("unitColorMode");
  if (cMode && (cMode.value === "gram_mode" || cMode.value === "cos_sim")) cMode.value = "owner";
  const bgSel = document.getElementById("mapBgMode");
  if (bgSel && bgSel.value === "terrain") bgSel.value = "diff";
  HIGHLIGHT_THREAT_XY = null;
  MICRO_STEP = 0;
  stopMicroPlayback();
  updateMicroStepUI();
  renderAll();
}
