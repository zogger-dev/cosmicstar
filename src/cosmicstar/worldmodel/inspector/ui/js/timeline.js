// ============================================================================
// PANEL 2: MULTI-SCALE TIMELINE SCRUBBER
// ============================================================================
function renderTimeline() {
  if (!TIMELINE || !TIMELINE.t_sec) return;
  const N = TIMELINE.t_sec.length;
  const tMax = TIMELINE.t_sec[N - 1] || 1;

  // Top Canvas: Advantage (-A1_L1, -A1_L2, -A1_tot)
  const { ctx: c1, w: w1, h: h1 } = setupHiDPICanvas(document.getElementById("timelineAdvCanvas"));
  c1.clearRect(0, 0, w1, h1);
  const padL = 42, padR = 12, padT = 8, padB = 16;
  const pw = w1 - padL - padR;
  const ph = h1 - padT - padB;

  let maxA = 0.5;
  for (let i = 0; i < N; i++) {
    maxA = Math.max(
      maxA,
      Math.abs(TIMELINE.A1_l1[i]),
      Math.abs(TIMELINE.A1_l2[i]),
      Math.abs(TIMELINE.A1_tot[i])
    );
  }
  maxA *= 1.12;

  const xOf = sec => padL + (sec / tMax) * pw;
  const yOf = val => padT + ph * (0.5 - 0.5 * (val / maxA));

  // Zero line
  c1.strokeStyle = "rgba(148, 163, 184, 0.3)";
  c1.beginPath();
  c1.moveTo(padL, yOf(0));
  c1.lineTo(padL + pw, yOf(0));
  c1.stroke();

  const drawCurve = (arr, color, width = 1.5) => {
    c1.strokeStyle = color;
    c1.lineWidth = width;
    c1.beginPath();
    for (let i = 0; i < N; i++) {
      const x = xOf(TIMELINE.t_sec[i]);
      const y = yOf(arr[i]);
      if (i === 0) c1.moveTo(x, y);
      else c1.lineTo(x, y);
    }
    c1.stroke();
  };

  // Draw A1_l1 with dual broadcast colors: positive in P1 (Red), negative in P2 (Blue)
  const drawDualCurve = (arr, width = 1.8) => {
    c1.lineWidth = width;
    for (let i = 0; i < N - 1; i++) {
      const xA = xOf(TIMELINE.t_sec[i]);
      const yA = yOf(arr[i]);
      const xB = xOf(TIMELINE.t_sec[i + 1]);
      const yB = yOf(arr[i + 1]);
      const avgVal = (arr[i] + arr[i + 1]) * 0.5;
      c1.strokeStyle = avgVal >= 0 ? "#ef4444" : "#3b82f6";
      c1.beginPath();
      c1.moveTo(xA, yA);
      c1.lineTo(xB, yB);
      c1.stroke();
    }
  };
  drawDualCurve(TIMELINE.A1_l1, 1.8);
  drawCurve(TIMELINE.A1_l2, "#a78bfa", 1.3);
  drawCurve(TIMELINE.A1_tot, "#10b981", 2.0);

  // Playhead line
  const curSec = CURRENT_LOOP / 22.4;
  const px = xOf(curSec);
  c1.strokeStyle = "#facc15";
  c1.lineWidth = 1.5;
  c1.beginPath();
  c1.moveTo(px, padT);
  c1.lineTo(px, padT + ph);
  c1.stroke();

  c1.fillStyle = "#ef4444";
  c1.font = "10px monospace";
  c1.fillText(`+${maxA.toFixed(1)}‰ P1`, 2, padT + 8);
  c1.fillStyle = "#3b82f6";
  c1.fillText(`-${maxA.toFixed(1)}‰ P2`, 2, padT + ph);

  // Bottom Canvas: Grounded Vital Pool (P1 vs P2 HP+Shield eq)
  const { ctx: c2, w: w2, h: h2 } = setupHiDPICanvas(document.getElementById("timelineHpCanvas"));
  c2.clearRect(0, 0, w2, h2);
  const ph2 = h2 - padT - padB;
  let maxHP = 10.0;
  for (let i = 0; i < N; i++) {
    maxHP = Math.max(maxHP, TIMELINE.p1_hp_eq[i], TIMELINE.p2_hp_eq[i]);
  }
  maxHP *= 1.1;
  const yHP = v => padT + ph2 * (1.0 - v / maxHP);

  const drawHP = (arr, col) => {
    c2.strokeStyle = col;
    c2.lineWidth = 1.5;
    c2.beginPath();
    for (let i = 0; i < N; i++) {
      const x = xOf(TIMELINE.t_sec[i]);
      const y = yHP(arr[i]);
      if (i === 0) c2.moveTo(x, y);
      else c2.lineTo(x, y);
    }
    c2.stroke();
  };
  drawHP(TIMELINE.p1_hp_eq, "#ef4444");
  drawHP(TIMELINE.p2_hp_eq, "#3b82f6");

  c2.strokeStyle = "#facc15";
  c2.beginPath();
  c2.moveTo(px, padT);
  c2.lineTo(px, padT + ph2);
  c2.stroke();

  c2.fillStyle = "#94a3b8";
  c2.font = "10px monospace";
  c2.fillText("HP+Shd", 2, padT + 10);
  for (let m = 0; m <= Math.floor(tMax / 120); m++) {
    const s = m * 120;
    const x = xOf(s);
    c2.fillText(fmtTime(s), x - 12, h2 - 3);
  }
}
