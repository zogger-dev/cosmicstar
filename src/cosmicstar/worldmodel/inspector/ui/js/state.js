window.onerror = function(msg, url, line, col, error) {
  const errDiv = document.createElement("div");
  errDiv.id = "globalJsError";
  errDiv.style = "position:fixed;top:0;left:0;right:0;background:#dc2626;color:white;z-index:999999;padding:12px;font-size:15px;font-family:monospace;font-weight:bold;box-shadow:0 4px 12px rgba(0,0,0,0.5);";
  errDiv.textContent = `JS ERROR: ${msg} (${url}:${line}:${col})`;
  document.body.prepend(errDiv);
  console.error("Global JS error:", msg, url, line, col, error);
};
window.onunhandledrejection = function(e) {
  const errDiv = document.createElement("div");
  errDiv.id = "globalJsPromiseError";
  errDiv.style = "position:fixed;top:40px;left:0;right:0;background:#b91c1c;color:white;z-index:999999;padding:12px;font-size:15px;font-family:monospace;font-weight:bold;box-shadow:0 4px 12px rgba(0,0,0,0.5);";
  errDiv.textContent = `PROMISE REJECTION: ${e.reason && e.reason.stack ? e.reason.stack : e.reason}`;
  document.body.prepend(errDiv);
  console.error("Unhandled promise rejection:", e.reason);
};
// ============================================================================
// GLOBAL STATE
// ============================================================================
let CATALOG = [];
let CURRENT_GID = "";
let CURRENT_LOOP = 2000;
let TIMELINE = null;
let FRAME = null;
let TOKEN_DETAILS = null;

let SELECTED_UNIT_ROW = null;
let SELECTED_TOK_IDX = null; // null when deselected
let SELECTED_GROUP_KEY = null; // e.g. "p1_SCV"
let SELECTED_GROUP_ROWS = new Set(); // Set of unit row indices in the selected group
let USER_DESELECTED_UNIT = true; // default to no unit selected so map isn't cluttered until clicked
let ACTIVE_GRAM_CELL = null; // null when deselected, or [i, j]
let ACTIVE_GRAM256_CELL = [0, 0]; // default (0, 0) in 256x256 / NxN / 8x256
let GRAM_256_DATA = null;
let ACTIVE_SVD_MODE = null; // 0..7 if eigenmode selected
let ACTIVE_L2_BASIS = -1;   // 0..15 if W_c column selected
let ACTIVE_FIELD_DIMS = new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]); // Selected potential field dimensions
let ACTIVE_SAE_FID = -1;    // Selected Layer-6 SAE feature ID (-1 = Off, use z0..z15)
let LAST_FIELD_EVAL = null; // Cached { grid, G, isDiverging, lo, hi, maxAbs, label }
let _fieldOffscreenCanvas = null;

function getRDim() {
  if (FRAME && FRAME.gram && FRAME.gram.labels_8 && FRAME.gram.labels_8.length > 0) {
    return FRAME.gram.labels_8.length;
  }
  return 16;
}

let COUNTERFACTUAL = {
  deleted_rows: [],
  moved_rows: {},
  hp_scale: {}
};

// Map Pan/Zoom & Drag state
let mapZoom = 1.0;
let mapPanX = 0.0;
let mapPanY = 0.0;
let isDraggingMap = false;
let isDraggingUnit = false;
let dragStartMouse = [0, 0];
let playTimer = null;

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s < 10 ? "0" : ""}${s}`;
}

function unescapeHtml(str) {
  if (!str) return "";
  const txt = document.createElement("textarea");
  txt.innerHTML = str;
  let val = txt.value;
  // Strip SC2 <sp/> tags and replace with a space
  val = val.replace(/<\s*\/?\s*sp\s*\/?>/gi, " ");
  // Strip SC2 color tags <c val="..."> and </c>
  val = val.replace(/<\/?c(\s+[^>]*)?>/gi, "");
  // Collapse consecutive whitespace
  val = val.replace(/\s+/g, " ").trim();
  return val;
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function setupHiDPICanvas(canvas, fixedW = null, fixedH = null) {
  const dpr = window.devicePixelRatio || 1;
  if (!canvas.dataset.baseH) {
    const attrH = canvas.getAttribute("height");
    canvas.dataset.baseH = attrH ? attrH : "rect";
    if (attrH) {
      canvas.style.height = `${attrH}px`;
    }
  }
  const rect = canvas.getBoundingClientRect();
  const w = fixedW || Math.max(10, Math.round(rect.width));
  const h = fixedH || (
    canvas.dataset.baseH === "rect"
      ? Math.max(10, Math.round(rect.height))
      : parseInt(canvas.dataset.baseH, 10)
  );
  const targetW = Math.round(w * dpr);
  const targetH = Math.round(h * dpr);
  if (canvas.width !== targetW) canvas.width = targetW;
  if (canvas.height !== targetH) canvas.height = targetH;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}
