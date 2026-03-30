// ─── Interpolation: SVG path d-string builders ───
// Ported from protovis basis/cardinal/monotone/hermite algorithms

const Interpolation = {
  _basisPoint(t0, p0, t1, p1, t2, p2, t3, p3) {
    return t0 * p0 + t1 * p1 + t2 * p2 + t3 * p3;
  },

  _pathBasis(p0, p1, p2, p3) {
    const bp = Interpolation._basisPoint;
    const x1 = bp(0, p0.x, 2/3, p1.x, 1/3, p2.x, 0, p3.x);
    const y1 = bp(0, p0.y, 2/3, p1.y, 1/3, p2.y, 0, p3.y);
    const x2 = bp(0, p0.x, 1/3, p1.x, 2/3, p2.x, 0, p3.x);
    const y2 = bp(0, p0.y, 1/3, p1.y, 2/3, p2.y, 0, p3.y);
    const x  = bp(0, p0.x, 1/6, p1.x, 2/3, p2.x, 1/6, p3.x);
    const y  = bp(0, p0.y, 1/6, p1.y, 2/3, p2.y, 1/6, p3.y);
    return `C${x1},${y1},${x2},${y2},${x},${y}`;
  },

  basis(points) {
    if (points.length <= 2) return Interpolation.linear(points);
    let d = '';
    let b0 = points[0], b1 = points[0], b2 = points[0], b3 = points[1];
    d += Interpolation._pathBasis(b0, b1, b2, b3);
    for (let i = 2; i < points.length; i++) {
      b0 = b1; b1 = b2; b2 = b3; b3 = points[i];
      d += Interpolation._pathBasis(b0, b1, b2, b3);
    }
    b0 = b1; b1 = b2; b2 = b3;
    d += Interpolation._pathBasis(b0, b1, b2, b3);
    b0 = b1; b1 = b2;
    d += Interpolation._pathBasis(b0, b1, b2, b3);
    return d;
  },

  _cardinalTangents(points, tension) {
    const alpha = (1 - tension) / 2;
    const tangents = [];
    let f = points[0], g = points[1], h = points[2];
    for (let i = 3; i < points.length; i++) {
      tangents.push({ x: alpha * (h.x - f.x), y: alpha * (h.y - f.y) });
      f = g; g = h; h = points[i];
    }
    tangents.push({ x: alpha * (h.x - f.x), y: alpha * (h.y - f.y) });
    return tangents;
  },

  _hermite(points, tangents) {
    if (tangents.length < 1 ||
        (points.length !== tangents.length && points.length !== tangents.length + 2))
      return '';
    const quad = points.length !== tangents.length;
    let d = '';
    let g = points[0], h = points[1], i = tangents[0], j = i, k = 1;

    if (quad) {
      d += `Q${h.x - i.x * 2/3},${h.y - i.y * 2/3},${h.x},${h.y}`;
      g = points[1]; k = 2;
    }
    if (tangents.length > 1) {
      j = tangents[1]; h = points[k]; k++;
      d += `C${g.x + i.x},${g.y + i.y},${h.x - j.x},${h.y - j.y},${h.x},${h.y}`;
      for (let idx = 2; idx < tangents.length; idx++, k++) {
        h = points[k]; j = tangents[idx];
        d += `S${h.x - j.x},${h.y - j.y},${h.x},${h.y}`;
      }
    }
    if (quad) {
      const last = points[k];
      d += `Q${h.x + j.x * 2/3},${h.y + j.y * 2/3},${last.x},${last.y}`;
    }
    return d;
  },

  cardinal(points, tension) {
    if (points.length <= 2) return Interpolation.linear(points);
    return Interpolation._hermite(points, Interpolation._cardinalTangents(points, tension));
  },

  _monotoneTangents(points) {
    const n = points.length;
    const d = [], m = [], dx = [];
    for (let i = 0; i < n - 1; i++) d.push((points[i + 1].y - points[i].y) / (points[i + 1].x - points[i].x));
    m.push(d[0]);
    for (let i = 1; i < n - 1; i++) m.push((d[i - 1] + d[i]) / 2);
    m.push(d[n - 2]);
    dx.push(points[1].x - points[0].x);
    for (let i = 1; i < n - 1; i++) dx.push((points[i + 1].x - points[i - 1].x) / 2);
    dx.push(points[n - 1].x - points[n - 2].x);
    for (let i = 0; i < n - 1; i++) { if (Math.abs(d[i]) < 1e-7) { m[i] = 0; m[i + 1] = 0; } }
    for (let i = 0; i < n - 1; i++) {
      if (Math.abs(m[i]) >= 1e-5 && Math.abs(m[i + 1]) >= 1e-5) {
        const alpha = m[i] / d[i], beta = m[i + 1] / d[i], sigma = alpha * alpha + beta * beta;
        if (sigma > 9) { const k = 3 / Math.sqrt(sigma); m[i] = k * alpha * d[i]; m[i + 1] = k * beta * d[i]; }
      }
    }
    const tangents = [];
    for (let i = 0; i < n; i++) { const denom = 1 + m[i] * m[i]; tangents.push({ x: dx[i] / 3 / denom, y: m[i] * dx[i] / 3 / denom }); }
    return tangents;
  },

  monotone(points) {
    if (points.length <= 2) return Interpolation.linear(points);
    return Interpolation._hermite(points, Interpolation._monotoneTangents(points));
  },

  linear(points) { let d = ''; for (let i = 1; i < points.length; i++) d += `L${points[i].x},${points[i].y}`; return d; },
  stepBefore(points) { let d = ''; for (let i = 1; i < points.length; i++) d += `V${points[i].y}H${points[i].x}`; return d; },
  stepAfter(points) { let d = ''; for (let i = 1; i < points.length; i++) d += `H${points[i].x}V${points[i].y}`; return d; },

  bezier(points) {
    let d = '';
    for (let i = 1; i < points.length; i++) {
      const prev = points[i - 1], cur = points[i];
      d += `C${prev.h2x ?? prev.x},${prev.h2y ?? prev.y},${cur.h1x ?? cur.x},${cur.h1y ?? cur.y},${cur.x},${cur.y}`;
    }
    return d;
  },

  ensureBezierHandles(points) {
    for (let i = 0; i < points.length; i++) {
      if (points[i].h1x !== undefined) continue;
      const prev = points[Math.max(0, i - 1)], next = points[Math.min(points.length - 1, i + 1)];
      const dx = (next.x - prev.x) * 0.25, dy = (next.y - prev.y) * 0.25;
      points[i].h1x = points[i].x - dx; points[i].h1y = points[i].y - dy;
      points[i].h2x = points[i].x + dx; points[i].h2y = points[i].y + dy;
    }
  },

  buildPathD(points, interpolation, tension) {
    if (!points || points.length === 0) return '';
    let d = `M${points[0].x},${points[0].y}`;
    if (points.length === 1) return d;
    switch (interpolation) {
      case 'basis': d += Interpolation.basis(points); break;
      case 'cardinal': d += Interpolation.cardinal(points, tension); break;
      case 'monotone': d += Interpolation.monotone(points); break;
      case 'step-before': d += Interpolation.stepBefore(points); break;
      case 'step-after': d += Interpolation.stepAfter(points); break;
      case 'bezier': Interpolation.ensureBezierHandles(points); d += Interpolation.bezier(points); break;
      case 'linear': default: d += Interpolation.linear(points); break;
    }
    return d;
  },

  drawOnCanvas(ctx, points, interpolation, tension) {
    if (!points || points.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    switch (interpolation) {
      case 'basis':
        if (points.length < 3) { ctx.lineTo(points[1].x, points[1].y); break; }
        Interpolation._drawBasisCanvas(ctx, points); break;
      case 'cardinal':
        if (points.length < 3) { ctx.lineTo(points[1].x, points[1].y); break; }
        Interpolation._drawHermiteCanvas(ctx, points, Interpolation._cardinalTangents(points, tension)); break;
      case 'monotone':
        if (points.length < 3) { ctx.lineTo(points[1].x, points[1].y); break; }
        Interpolation._drawHermiteCanvas(ctx, points, Interpolation._monotoneTangents(points)); break;
      case 'step-before':
        for (let i = 1; i < points.length; i++) { ctx.lineTo(points[i - 1].x, points[i].y); ctx.lineTo(points[i].x, points[i].y); } break;
      case 'step-after':
        for (let i = 1; i < points.length; i++) { ctx.lineTo(points[i].x, points[i - 1].y); ctx.lineTo(points[i].x, points[i].y); } break;
      case 'bezier':
        Interpolation.ensureBezierHandles(points);
        for (let i = 1; i < points.length; i++) {
          const prev = points[i - 1], cur = points[i];
          ctx.bezierCurveTo(prev.h2x ?? prev.x, prev.h2y ?? prev.y, cur.h1x ?? cur.x, cur.h1y ?? cur.y, cur.x, cur.y);
        } break;
      case 'linear': default:
        for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y); break;
    }
  },

  _drawBasisCanvas(ctx, points) {
    const bp = Interpolation._basisPoint;
    const drawSeg = (p0, p1, p2, p3) => {
      ctx.bezierCurveTo(
        bp(0,p0.x,2/3,p1.x,1/3,p2.x,0,p3.x), bp(0,p0.y,2/3,p1.y,1/3,p2.y,0,p3.y),
        bp(0,p0.x,1/3,p1.x,2/3,p2.x,0,p3.x), bp(0,p0.y,1/3,p1.y,2/3,p2.y,0,p3.y),
        bp(0,p0.x,1/6,p1.x,2/3,p2.x,1/6,p3.x), bp(0,p0.y,1/6,p1.y,2/3,p2.y,1/6,p3.y));
    };
    let b0 = points[0], b1 = points[0], b2 = points[0], b3 = points[1];
    drawSeg(b0, b1, b2, b3);
    for (let i = 2; i < points.length; i++) { b0 = b1; b1 = b2; b2 = b3; b3 = points[i]; drawSeg(b0, b1, b2, b3); }
    b0 = b1; b1 = b2; b2 = b3; drawSeg(b0, b1, b2, b3);
    b0 = b1; b1 = b2; drawSeg(b0, b1, b2, b3);
  },

  _drawHermiteCanvas(ctx, points, tangents) {
    if (tangents.length < 1 || (points.length !== tangents.length && points.length !== tangents.length + 2)) return;
    const quad = points.length !== tangents.length;
    let g = points[0], h = points[1], i = tangents[0], j = i, k = 1;
    let prevCp2x, prevCp2y;
    if (quad) { ctx.quadraticCurveTo(h.x - i.x * 2/3, h.y - i.y * 2/3, h.x, h.y); g = points[1]; k = 2; }
    if (tangents.length > 1) {
      j = tangents[1]; h = points[k]; k++;
      prevCp2x = h.x - j.x; prevCp2y = h.y - j.y;
      ctx.bezierCurveTo(g.x + i.x, g.y + i.y, prevCp2x, prevCp2y, h.x, h.y);
      for (let idx = 2; idx < tangents.length; idx++, k++) {
        const prevH = h; h = points[k]; j = tangents[idx];
        const rcp1x = 2 * prevH.x - prevCp2x, rcp1y = 2 * prevH.y - prevCp2y;
        prevCp2x = h.x - j.x; prevCp2y = h.y - j.y;
        ctx.bezierCurveTo(rcp1x, rcp1y, prevCp2x, prevCp2y, h.x, h.y);
      }
    }
    if (quad) { const last = points[k]; ctx.quadraticCurveTo(h.x + j.x * 2/3, h.y + j.y * 2/3, last.x, last.y); }
  }
};

export { Interpolation };
