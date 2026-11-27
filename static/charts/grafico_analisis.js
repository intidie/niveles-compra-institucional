/*
 * Grafico de la pestana Analisis sobre TradingView Lightweight Charts (v5, build standalone servida en local).
 *
 * Un unico grafico (canvas) con un "series primitive" que dibuja, con las MISMAS coordenadas de precio/tiempo del
 * propio grafico: bandas de zonas N1-N4 con etiqueta (sin solaparse), Volume Profile lateral anclado al borde derecho
 * (misma escala vertical y mismo zoom/arrastre que las velas, sin sincronizar dos graficos), anclas de valoracion,
 * trimestres 13F (banda vertical + rombo) y compras Form 4 (triangulo). El valor justo medio es una priceLine nativa
 * (etiqueta en el eje) mas su rotulo. Todos los colores llegan de Python (``tema``): este archivo no tiene literales.
 *
 * API (window.NivelesGrafico):
 *   actualizar(id, {tema, datos?, overlays?})  datos = velas/perfil/marcadores (se reenvian solo al cambiar de ticker);
 *                                              overlays = zonas, anclas, precio, valor justo, titulo (cambian a menudo).
 *   ajustar(id)  restablece zoom y escala;  estado(id)  diagnostico para pruebas;  zoom(id, f)  zoom programatico.
 */
(function () {
  'use strict';
  var REG = {};
  var ERR = (window.__ngErrores = window.__ngErrores || []);
  var VP_FRAC = 0.20;       // ancho maximo del perfil: 20 % del area de trazado
  var PAD_DER = 0.24;       // hueco a la derecha (fraccion de barras) para que las ultimas velas no queden bajo el perfil
  var MESES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic'];
  var FUENTE ="Roboto, 'Segoe UI', -apple-system, Helvetica, Arial, sans-serif";

  function err(ctx, e) {
    var m = '[NivelesGrafico] ' + ctx + ': ' + ((e && (e.stack || e.message)) || e);
    ERR.push(m);
    console.error(m);
  }
  function hexA(hex, a) {
    var m = /^#([0-9a-f]{6})$/i.exec(hex || '');
    if (!m) return hex;
    var n = parseInt(m[1], 16);
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }
  function fecha(t) { return new Date(t * 1000).toISOString().slice(0, 10); }
  function usd(v) { return '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); }
  function num(v) { return Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 }); }

  // ------------------------------------------------------------------------------------------- colocacion de etiquetas
  function choca(r, puestos) {
    for (var i = 0; i < puestos.length; i++) {
      var q = puestos[i];
      if (r.x < q.x + q.w + 2 && r.x + r.w + 2 > q.x && r.y < q.y + q.h + 1 && r.y + r.h + 1 > q.y) return true;
    }
    return false;
  }
  // Coloca rectangulos {x,y,w,h} sin solaparse: se procesan por y preferida y, si chocan, bajan (o suben cerca del borde).
  function colocar(items, H) {
    items.sort(function (a, b) { return a.prio - b.prio || a.y - b.y; });
    var puestos = [];
    items.forEach(function (it) {
      var paso = it.h + 2, k = 0, dir = 1;
      it.y0 = it.y;
      while (choca(it, puestos) && k < 60) {
        it.y += dir * paso;
        k++;
        if (it.y + it.h > H - 2) { it.y = it.y0; dir = -1; it.y += dir * paso; }
        else if (it.y < 2 && dir < 0) break;
      }
      puestos.push(it);
    });
    return puestos;
  }

  // ------------------------------------------------------------------------------------------------- dibujo (primitive)
  function rotulo(ctx, it, T) {
    ctx.font = it.fuente;
    ctx.fillStyle = it.fondo;
    ctx.fillRect(it.x, it.y, it.w, it.h);
    ctx.strokeStyle = it.borde;
    ctx.lineWidth = it.sel ? 2 : 1;
    ctx.strokeRect(it.x + 0.5, it.y + 0.5, it.w - 1, it.h - 1);
    ctx.fillStyle = it.color;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.fillText(it.txt, it.x + 5, it.y + it.h / 2 + 0.5);
  }
  function medir(ctx, txt, fuente) { ctx.font = fuente; return Math.ceil(ctx.measureText(txt).width) + 10; }

  function pintarFondo(st, ov, s) {
    var d = st.datos, o = st.overlays, T = st.tema;
    if (!d || d.vacio || !T) return;
    var ctx = s.context, W = s.mediaSize.width, H = s.mediaSize.height;
    var ts = ov.chart.timeScale(), sr = ov.series;
    // trimestres 13F: banda vertical translucida
    (d.q13 || []).forEach(function (q) {
      var x0 = ts.timeToCoordinate(q.t0), x1 = ts.timeToCoordinate(q.t);
      if (x0 === null && x1 === null) return;
      if (x0 === null) x0 = 0;
      if (x1 === null) x1 = W;
      ctx.fillStyle = hexA(T.color_13f, 0.10);
      ctx.fillRect(Math.min(x0, x1), 0, Math.abs(x1 - x0), H);
    });
    // Volume Profile: barras horizontales ancladas al borde derecho, alineadas a los bins de precio
    var vp = d.vp;
    if (vp && vp.p && vp.p.length) {
      var vmax = 0, i;
      for (i = 0; i < vp.v.length; i++) if (vp.v[i] > vmax) vmax = vp.v[i];
      var half = vp.paso * 0.44, wmax = W * VP_FRAC;
      for (i = 0; i < vp.p.length; i++) {
        var ya = sr.priceToCoordinate(vp.p[i] + half), yb = sr.priceToCoordinate(vp.p[i] - half);
        if (ya === null || yb === null || yb < 0 || ya > H) continue;
        var w = vmax > 0 ? wmax * vp.v[i] / vmax : 0;
        ctx.fillStyle = vp.v[i] >= 0.7 * vmax ? hexA(T.color_hvn, 0.85) : hexA(T.color_barra_vp, 0.9);
        ctx.fillRect(W - w, ya, w, Math.max(1, yb - ya - 0.5));
      }
    }
    if (!o) return;
    // anclas de valoracion (DCF / P/E): lineas punteadas
    (o.anclas || []).forEach(function (a) {
      var y = sr.priceToCoordinate(a.precio);
      if (y === null || y < 0 || y > H) return;
      ctx.strokeStyle = hexA(T.color_valoracion, 0.6);
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(W, y + 0.5); ctx.stroke();
      ctx.setLineDash([]);
    });
    // bandas de zonas (encima del perfil para teñirlo, debajo de las velas)
    (o.zonas || []).forEach(function (z) {
      var y0 = sr.priceToCoordinate(z.hi), y1 = sr.priceToCoordinate(z.lo);
      if (y0 === null || y1 === null || y1 < 0 || y0 > H) return;
      var c = T.niveles[z.nivel] || T.muted, h = Math.max(2, y1 - y0);
      ctx.fillStyle = hexA(c, z.sel ? 0.42 : 0.18);
      ctx.fillRect(0, y0, W, h);
      ctx.strokeStyle = z.sel ? T.blanco : hexA(c, 0.7);
      ctx.lineWidth = z.sel ? 2 : 1;
      ctx.strokeRect(0.5, y0 + 0.5, W - 1, Math.max(1, h - 1));
    });
  }

  function pintarFrente(st, ov, s) {
    var d = st.datos, o = st.overlays, T = st.tema;
    st.hits = []; st.rotulos = [];
    if (!d || d.vacio || !T) return;
    var ctx = s.context, W = s.mediaSize.width, H = s.mediaSize.height;
    var ts = ov.chart.timeScale(), sr = ov.series;
    var lineaPx = 0.5;
    // compras Form 4: triangulo; trimestres 13F: rombo (sobre el cierre del trimestre)
    function marca(x, y, forma, color, tip) {
      var r = 6;
      ctx.beginPath();
      if (forma === 'rombo') { ctx.moveTo(x, y - r - 1); ctx.lineTo(x + r + 1, y); ctx.lineTo(x, y + r + 1); ctx.lineTo(x - r - 1, y); }
      else { ctx.moveTo(x, y - r); ctx.lineTo(x + r, y + r - 1); ctx.lineTo(x - r, y + r - 1); }
      ctx.closePath();
      ctx.fillStyle = color; ctx.fill();
      ctx.strokeStyle = T.blanco; ctx.lineWidth = 1; ctx.stroke();
      st.hits.push({ x: x, y: y, r: 9, txt: tip });
    }
    (d.q13 || []).forEach(function (q) {
      var x = ts.timeToCoordinate(q.t), y = sr.priceToCoordinate(q.p);
      if (x === null || y === null || x < -10 || x > W + 10 || y < -10 || y > H + 10) return;
      marca(x, y, 'rombo', T.color_13f, 'Cierre trimestre 13F (acumula) | ' + q.f + ' | ' + usd(q.p));
    });
    (d.form4 || []).forEach(function (m) {
      var x = ts.timeToCoordinate(m.t), y = sr.priceToCoordinate(m.p);
      if (x === null || y === null || x < -10 || x > W + 10 || y < -10 || y > H + 10) return;
      marca(x, y, 'triangulo', T.color_form4, 'Insider Form 4 (P) | ' + m.f + ' | ' + usd(m.p) + (m.n > 1 ? ' | ' + m.n + ' compras' : ''));
    });
    if (!o) return;
    var hayVP = d.vp && d.vp.p && d.vp.p.length;
    var rx = W - (hayVP ? W * VP_FRAC : 0) - 8;
    var items = [];
    function nuevo(txt, fuente, color, borde, extra) {
      ctx.font = fuente;
      var it = { txt: txt, fuente: fuente, color: color, borde: borde, fondo: T.bg_etiqueta, w: medir(ctx, txt, fuente), h: fuente.indexOf('12px') >= 0 ? 18 : 16, prio: 5 };
      for (var k in extra) it[k] = extra[k];
      return it;
    }
    // zonas: etiqueta a la izquierda, junto al borde superior de la banda
    (o.zonas || []).forEach(function (z) {
      var y0 = sr.priceToCoordinate(z.hi), y1 = sr.priceToCoordinate(z.lo);
      if (y0 === null || y1 === null || y1 < -4 || y0 > H + 4) return;
      var c = T.niveles[z.nivel] || T.muted;
      var it = nuevo(z.etiqueta, (z.sel ? 'bold 12px ' : '11px ') + FUENTE, T.blanco, c, { sel: z.sel, prio: z.sel ? 0 : 3 });
      it.x = 8; it.y = Math.max(2, Math.min(H - it.h - 2, y0 + 1)); it.yBanda = (y0 + y1) / 2; it.cBanda = c; it.lado = 'izq';
      items.push(it);
    });
    // valor justo medio (linea = priceLine nativa; aqui solo el rotulo, a la derecha y sobre la linea)
    var vj = o.vj;
    if (vj && vj.etiqueta) {
      var f = 'bold 12px ' + FUENTE, yv;
      if (vj.fuera_de_rango) yv = vj.fuera_de_rango === 'arriba' ? 14 : H - 30;
      else { yv = sr.priceToCoordinate(vj.precio); yv = yv === null ? null : yv - 21; }
      if (yv !== null) {
        var iv = nuevo(vj.etiqueta, f, T.color_fair_value, T.color_fair_value, { prio: 1 });
        iv.x = rx - iv.w; iv.y = Math.max(2, Math.min(H - iv.h - 2, yv)); iv.lado = 'der';
        items.push(iv);
      }
    }
    // anclas de valoracion: tipo, a la derecha y bajo la linea
    (o.anclas || []).forEach(function (a) {
      var y = sr.priceToCoordinate(a.precio);
      if (y === null || y < 0 || y > H) return;
      var ia = nuevo(a.tipo, '10px ' + FUENTE, T.color_valoracion, hexA(T.color_valoracion, 0.5), { prio: 4 });
      ia.x = rx - ia.w; ia.y = y + 2; ia.lado = 'der';
      items.push(ia);
    });
    colocar(items, H).forEach(function (it) {
      if (it.lado === 'izq' && Math.abs(it.y + it.h / 2 - it.yBanda) > it.h / 2 + 2) {  // desplazada: conector a su banda
        ctx.strokeStyle = it.cBanda; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(4.5, it.y + it.h / 2); ctx.lineTo(4.5, it.yBanda); ctx.lineTo(8, it.yBanda); ctx.stroke();
      }
      rotulo(ctx, it, T);
      st.rotulos.push({ txt: it.txt, x: Math.round(it.x), y: Math.round(it.y), w: it.w, h: it.h });
    });
  }

  function crearOverlay(st) {
    var ov = {
      chart: null, series: null, req: null,
      attached: function (p) { ov.chart = p.chart; ov.series = p.series; ov.req = p.requestUpdate; },
      detached: function () { ov.req = null; },
      updateAllViews: function () {},
      refrescar: function () { if (ov.req) ov.req(); },
      paneViews: function () { return [ov.vFondo, ov.vFrente]; }
    };
    function vista(z, fn) {
      return {
        zOrder: function () { return z; },
        renderer: function () {
          return { draw: function (t) { t.useMediaCoordinateSpace(function (s) { try { fn(st, ov, s); } catch (e) { err('dibujo', e); } }); } };
        }
      };
    }
    ov.vFondo = vista('bottom', pintarFondo);
    ov.vFrente = vista('top', pintarFrente);
    return ov;
  }

  // -------------------------------------------------------------------------------------------------- montaje y datos
  function estilos(T) {
    return {
      layout: { background: { type: 'solid', color: T.bg_panel }, textColor: T.txt, fontFamily: FUENTE, attributionLogo: true },
      grid: { vertLines: { color: T.grilla }, horzLines: { color: T.grilla } },
      crosshair: {
        mode: 0,
        vertLine: { color: T.muted, labelBackgroundColor: T.bg_popup, style: 3 },
        horzLine: { color: T.muted, labelBackgroundColor: T.bg_popup, style: 3 }
      },
      rightPriceScale: { borderColor: T.borde, scaleMargins: { top: 0.06, bottom: 0.06 }, minimumWidth: 64 },
      timeScale: {
        borderColor: T.borde, timeVisible: false, secondsVisible: false, rightOffset: 0, minBarSpacing: 0.02,
        tickMarkFormatter: function (t, tipo) {   // 0 anio, 1 mes, 2 dia (tiempo UNIX de negocio, UTC)
          var d = new Date(t * 1000), m = MESES[d.getUTCMonth()];
          return tipo === 0 ? String(d.getUTCFullYear()) : (tipo === 1 ? m : d.getUTCDate() + ' ' + m);
        }
      },
      localization: { dateFormat: 'yyyy-MM-dd', priceFormatter: function (p) { return '$' + p.toFixed(2); } }
    };
  }

  function montar(st) {
    var el = document.getElementById(st.id);
    if (!el) return false;
    if (st.el === el && st.chart) return true;
    if (st.chart) { try { st.ro.disconnect(); st.chart.remove(); } catch (e) { err('desmontar', e); } st.chart = null; }
    var LC = window.LightweightCharts;
    if (!LC) { el.textContent = 'No se pudo cargar el grafico (lightweight-charts).'; err('montar', 'LightweightCharts no esta cargado'); return true; }
    var T = st.tema;
    if (!T) return false;
    st.el = el;
    el.innerHTML = '';
    el.style.display = 'flex'; el.style.flexDirection = 'column'; el.style.position = 'relative';
    el.style.background = T.bg_panel; el.style.border = '1px solid ' + T.borde; el.style.borderRadius = '4px';
    st.cab = document.createElement('div');
    st.cab.style.cssText = 'padding:6px 10px 2px 10px;color:' + T.txt + ';font:13px ' + FUENTE + ';display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center';
    st.cab.setAttribute('data-ng', 'cabecera');
    el.appendChild(st.cab);
    st.caja = document.createElement('div');
    st.caja.style.cssText = 'position:relative;flex:1 1 auto;min-height:0';
    el.appendChild(st.caja);
    st.chart = LC.createChart(st.caja, Object.assign({ autoSize: true }, estilos(T)));
    st.velas = st.chart.addSeries(LC.CandlestickSeries, {
      upColor: T.vela_alza, downColor: T.vela_baja, wickUpColor: T.vela_alza, wickDownColor: T.vela_baja,
      borderVisible: false, priceLineVisible: false, lastValueVisible: true,
      autoscaleInfoProvider: function (orig) { return extender(st, orig()); }
    });
    st.sma = st.chart.addSeries(LC.LineSeries, {
      color: T.muted, lineWidth: 1, lineStyle: LC.LineStyle.Dotted, priceLineVisible: false, lastValueVisible: false,
      crosshairMarkerVisible: false
    });
    st.overlay = crearOverlay(st);
    st.velas.attachPrimitive(st.overlay);
    st.lineas = [];
    st.tip = document.createElement('div');
    st.tip.setAttribute('data-ng', 'tooltip');
    st.tip.style.cssText = 'position:absolute;display:none;pointer-events:none;z-index:5;max-width:380px;padding:6px 9px;border-radius:4px;' +
      'font:12px/1.45 ' + FUENTE + ';color:' + T.txt + ';background:' + T.bg_popup + ';border:1px solid ' + T.borde;
    st.caja.appendChild(st.tip);
    st.btn = document.createElement('button');
    st.btn.textContent = 'Ajustar vista';
    st.btn.title = 'Restablece el zoom y la escala de precios';
    st.btn.setAttribute('data-ng', 'ajustar');
    st.btn.style.cssText = 'position:absolute;top:6px;right:80px;z-index:4;cursor:pointer;padding:2px 8px;border-radius:3px;font:11px ' + FUENTE +
      ';color:' + T.txt + ';background:' + T.bg_popup + ';border:1px solid ' + T.borde;
    st.btn.onclick = function () { ajustar(st.id); };
    st.caja.appendChild(st.btn);
    st.chart.subscribeCrosshairMove(function (p) { try { tooltip(st, p); } catch (e) { err('tooltip', e); } });
    st.ro = new ResizeObserver(function () {   // pestana oculta = tamano 0; al volver se reajusta una sola vez
      var w = st.caja.clientWidth, h = st.caja.clientHeight;
      if (w > 0 && h > 0 && st.ajustePend) { st.ajustePend = false; requestAnimationFrame(function () { rangoInicial(st); }); }
    });
    st.ro.observe(st.caja);
    return true;
  }

  function extender(st, r) {   // la escala incluye zonas / valor justo / precio cercanos a lo visible (no todo el historial)
    var o = st.overlays;
    if (!r || !o || !o.extras || !o.extras.length) return r;
    var lo = r.priceRange.minValue, hi = r.priceRange.maxValue;
    var piso = lo * 0.75, techo = hi * 1.25, mn = lo, mx = hi;
    o.extras.forEach(function (p) { if (p >= piso && p <= techo) { if (p < mn) mn = p; if (p > mx) mx = p; } });
    var pad = (mx - mn) * 0.02;
    if (mn < lo) mn -= pad;
    if (mx > hi) mx += pad;
    return { priceRange: { minValue: mn, maxValue: mx }, margins: r.margins };
  }

  function rangoInicial(st) {
    var d = st.datos;
    if (!st.chart || !d || d.vacio || !d.t || !d.t.length) return;
    var n = d.t.length, hayVP = d.vp && d.vp.p && d.vp.p.length;
    st.chart.priceScale('right').applyOptions({ autoScale: true });
    st.chart.timeScale().setVisibleLogicalRange({ from: -0.5, to: n - 1 + Math.max(3, n * (hayVP ? PAD_DER : 0.03)) });
  }

  function cabecera(st) {
    var d = st.datos, o = st.overlays, T = st.tema, c = st.cab;
    if (!c) return;
    c.textContent = '';
    function chip(simbolo, color, texto) {
      var s = document.createElement('span');
      s.style.cssText = 'white-space:nowrap;font-size:12px;color:' + T.txt;
      var i = document.createElement('span');
      i.textContent = simbolo + ' ';
      i.style.color = color;
      s.appendChild(i); s.appendChild(document.createTextNode(texto));
      c.appendChild(s);
    }
    var tit = document.createElement('strong');
    tit.style.cssText = 'font-size:14px;color:' + T.txt;
    tit.textContent = (o && o.titulo) || 'Mapa de niveles institucionales';
    c.appendChild(tit);
    if (!d || d.vacio) return;
    chip('■', T.vela_alza, 'Precio');
    if (d.tiene_sma) chip('┈', T.muted, 'SMA 200D');
    if (d.form4 && d.form4.length) chip('▲', T.color_form4, 'Compra insider Form 4 (P) x' + d.form4.length);
    if (d.q13 && d.q13.length) chip('◆', T.color_13f, 'Acumulación 13F x' + d.q13.length);
    if (o && o.anclas && o.anclas.length) chip('┈', T.color_valoracion, 'Valoración (DCF / P/E)');
    if (o && o.vj) chip('┄', T.color_fair_value, 'Valor justo medio');
    if (d.vp && d.vp.p && d.vp.p.length) chip('▮', T.color_hvn, 'Volume Profile (HVN naranja)');
  }

  function mensajeVacio(st, txt) {
    var T = st.tema;
    if (!st.vacio) {
      st.vacio = document.createElement('div');
      st.vacio.setAttribute('data-ng', 'vacio');
      st.vacio.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:3;pointer-events:none;' +
        'font:14px ' + FUENTE + ';color:' + T.muted;
      st.caja.appendChild(st.vacio);
    }
    st.vacio.textContent = txt || '';
    st.vacio.style.display = txt ? 'flex' : 'none';
  }

  function aplicar(st) {
    if (!montar(st) || !st.chart) return;
    var T = st.tema, d = st.datos, o = st.overlays, LC = window.LightweightCharts;
    var m = st.mensajes || {};
    if (m.datos) {
      m.datos = false;
      if (!d || d.vacio) {
        st.velas.setData([]); st.sma.setData([]);
        mensajeVacio(st, (d && d.vacio) || 'Sin datos de mercado');
        st.n = 0;
      } else {
        mensajeVacio(st, '');
        var i, n = d.t.length, arr = new Array(n);
        for (i = 0; i < n; i++) arr[i] = { time: d.t[i], open: d.o[i], high: d.h[i], low: d.l[i], close: d.c[i] };
        st.velas.setData(arr);
        var sm = [];
        if (d.sma) for (i = 0; i < n; i++) if (d.sma[i] !== null && d.sma[i] !== undefined) sm.push({ time: d.t[i], value: d.sma[i] });
        st.sma.setData(sm);
        st.n = n;
        st.ajustePend = false;
        if (st.caja.clientWidth > 0 && st.caja.clientHeight > 0) requestAnimationFrame(function () { rangoInicial(st); });
        else st.ajustePend = true;
      }
    }
    if (m.overlays || m.datos) {
      m.overlays = false;
      (st.lineas || []).forEach(function (l) { try { st.velas.removePriceLine(l); } catch (e) { /* ya quitada */ } });
      st.lineas = [];
      if (o && d && !d.vacio) {
        if (o.precio) st.lineas.push(st.velas.createPriceLine({ price: o.precio, color: hexA(T.color_precio, 0.8), lineWidth: 1, lineStyle: LC.LineStyle.Solid, axisLabelVisible: false, title: '' }));
        if (o.vj && !o.vj.fuera_de_rango) st.lineas.push(st.velas.createPriceLine({
          price: o.vj.precio, color: T.color_fair_value, lineWidth: 2, lineStyle: LC.LineStyle.Dashed,
          axisLabelVisible: true, axisLabelColor: T.color_fair_value, axisLabelTextColor: T.bg_pagina, title: ''
        }));
      }
      // re-evalua el autoscale (cambia el proveedor de rango) y repinta el primitive
      st.velas.applyOptions({ autoscaleInfoProvider: function (orig) { return extender(st, orig()); } });
    }
    cabecera(st);
    st.overlay.refrescar();
  }

  function pedir(st) {
    if (st.timer) return;
    st.timer = setTimeout(function () {
      st.timer = null;
      try {
        if (!montar(st)) {   // el contenedor aun no existe en el DOM (primer render): reintenta
          st.intentos = (st.intentos || 0) + 1;
          if (st.intentos < 200) pedir(st); else err('montar', 'contenedor ' + st.id + ' no encontrado');
          return;
        }
        aplicar(st);
      } catch (e) { err('aplicar', e); }
    }, 0);
  }

  function tooltip(st, p) {
    var tip = st.tip;
    if (!p || !p.point || p.point.x < 0 || p.point.y < 0 || !st.datos || st.datos.vacio) { tip.style.display = 'none'; return; }
    var T = st.tema, d = st.datos, o = st.overlays || {}, x = p.point.x, y = p.point.y, W = st.caja.clientWidth, H = st.caja.clientHeight;
    var lin = [];
    var b = p.seriesData && p.seriesData.get(st.velas);
    if (p.time) lin.push(['b', fecha(p.time)]);
    if (b) {
      var ch = b.open ? (b.close / b.open - 1) * 100 : 0;
      lin.push(['t', 'O ' + usd(b.open) + '  H ' + usd(b.high) + '  L ' + usd(b.low) + '  C ' + usd(b.close) + '  (' + (ch >= 0 ? '+' : '') + ch.toFixed(1) + '%)']);
    }
    var sv = p.seriesData && p.seriesData.get(st.sma);
    if (sv && sv.value !== undefined) lin.push(['m', 'SMA 200D ' + usd(sv.value)]);
    (st.hits || []).forEach(function (h) { if (Math.abs(h.x - x) <= h.r && Math.abs(h.y - y) <= h.r) lin.push(['t', h.txt]); });
    var precio = st.velas.coordinateToPrice(y);
    if (precio !== null) {
      (o.zonas || []).forEach(function (z) { if (precio >= z.lo && precio <= z.hi) lin.push(['t', z.etiqueta]); });
      var vp = d.vp;
      if (vp && vp.p && vp.p.length && x >= W * (1 - VP_FRAC) - 70) {
        for (var i = 0; i < vp.p.length; i++) {
          if (Math.abs(vp.p[i] - precio) <= vp.paso / 2) { lin.push(['m', 'Volume Profile ' + usd(vp.p[i]) + ': vol. ' + num(vp.v[i])]); break; }
        }
      }
    }
    tip.textContent = '';
    lin.forEach(function (l) {
      var r = document.createElement('div');
      r.textContent = l[1];
      if (l[0] === 'b') r.style.fontWeight = '700';
      if (l[0] === 'm') r.style.color = T.muted;
      tip.appendChild(r);
    });
    tip.style.display = 'block';
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    var left = x + 16 + tw > W - 70 ? x - 16 - tw : x + 16;
    var top = Math.min(Math.max(4, y - th / 2), H - th - 4);
    tip.style.left = Math.max(4, left) + 'px';
    tip.style.top = top + 'px';
  }

  // --------------------------------------------------------------------------------------------------------------- API
  function reg(id) { return REG[id] || (REG[id] = { id: id, mensajes: {} }); }
  function actualizar(id, msg) {
    try {
      var st = reg(id);
      if (msg.tema) { st.tema = msg.tema; }
      if (msg.datos !== undefined && msg.datos !== null) { st.datos = msg.datos; st.mensajes.datos = true; }
      if (msg.overlays !== undefined && msg.overlays !== null) { st.overlays = msg.overlays; st.mensajes.overlays = true; }
      pedir(st);
    } catch (e) { err('actualizar', e); }
  }
  function ajustar(id) {
    var st = REG[id];
    if (st && st.chart) { st.chart.priceScale('right').applyOptions({ autoScale: true }); rangoInicial(st); }
  }
  function zoom(id, f) {   // factor > 1 acerca, anclado al borde derecho de los datos
    var st = REG[id];
    if (!st || !st.chart) return null;
    var ts = st.chart.timeScale(), r = ts.getVisibleLogicalRange();
    if (!r) return null;
    var fin = r.to, ancho = (r.to - r.from) / f;
    ts.setVisibleLogicalRange({ from: fin - ancho, to: fin });
    return ts.getVisibleLogicalRange();
  }
  function estado(id) {
    var st = REG[id];
    if (!st || !st.chart) return { montado: false, errores: ERR.length };
    var o = st.overlays || {}, d = st.datos || {}, H = st.caja.clientHeight;
    var rg = st.chart.timeScale().getVisibleLogicalRange();
    return {
      montado: true, barras: st.n || 0, rango: rg ? [rg.from, rg.to] : null, ancho: st.caja.clientWidth, alto: H,
      precioArriba: st.velas.coordinateToPrice(0), precioAbajo: st.velas.coordinateToPrice(H),
      zonas: (o.zonas || []).length, anclas: (o.anclas || []).length, form4: (d.form4 || []).length, q13: (d.q13 || []).length,
      vp: d.vp && d.vp.p ? d.vp.p.length : 0, vj: o.vj ? o.vj.etiqueta : null, lineas: (st.lineas || []).length,
      rotulos: st.rotulos || [], marcas: (st.hits || []).length, errores: ERR.length, titulo: o.titulo || null
    };
  }
  window.NivelesGrafico = { actualizar: actualizar, ajustar: ajustar, zoom: zoom, estado: estado, _reg: REG };
  window.addEventListener('error', function (e) { if (e && e.message && /NivelesGrafico|lightweight/i.test(String(e.filename || '') + e.message)) err('window', e.message); });
})();
