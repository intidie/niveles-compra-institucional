# Metodologia de validacion empirica de rangos de compra institucional

Todas las cifras de este documento provienen de la ejecucion real de `src/validation/` (datos de mercado
yfinance ajustados por splits y dividendos, descargados el 2026-11-26; 13F-HR de SEC EDGAR via edgartools).
Reproducir con (PowerShell, desde la raiz del proyecto):

```
.\.venv\Scripts\python.exe -m src.validation.backtest_ranges
.\.venv\Scripts\python.exe -m src.validation.case_studies
.\.venv\Scripts\python.exe -m src.validation.scoring
```

## 1. Marco teorico y referencias

| Ancla / metodo | Fundamento | Referencia |
|---|---|---|
| Volume Profile (POC, VAL, VAH, HVN) | Aceptacion de precio por volumen; Value Area = 70 % del volumen | Steidlmayer & Koy, *Markets and Market Logic* (1986); Dalton, Jones & Dalton, *Mind Over Markets* |
| Anchored VWAP | Costo medio ponderado por volumen desde un evento estructural | Shannon, *Maximum Trading Gains With Anchored VWAP* (2023) |
| Order Blocks semanales no mitigados | Conceptos Smart Money, libreria `smartmoneyconcepts` | Implementacion de terceros; sin respaldo academico formal |
| ATR estructural | Unidad de ruido (ancho de bins y clusters) | Wilder, *New Concepts in Technical Trading Systems* (1978) |
| Bandas de regresion logaritmica | Canal de tendencia exponencial de largo plazo | Ajuste ln(P) = a + b t |
| Multiplos propios, DCF y margen de seguridad | Valor intrinseco y margen de seguridad | Graham & Dodd, *Security Analysis* (1934); Graham, *The Intelligent Investor* (1949); Damodaran, *Investment Valuation* |
| 13F-HR | Posiciones trimestrales de gestores con >= USD 100 M, presentadas hasta 45 dias tras el cierre; sin fecha de ejecucion | SEC, Form 13F (Section 13(f) Exchange Act), EDGAR |
| Form 4, codigo P | Compras de insiders en mercado abierto (Section 16(a)) | SEC EDGAR |
| STOCK Act | Divulgacion de operaciones de miembros del Congreso | STOCK Act (2012) |

## 2. Backtest walk-forward de rangos (`backtest_ranges.py`)

### 2.1 Construccion del rango sin look-ahead
* Cada 21 ruedas (~mensual) durante 10 anios (120 fechas por ticker) se recalcula el rango usando unicamente
  `df.iloc[:i+1]` (datos hasta la fecha t). Se reutiliza `src.levels.volume_liquidity` (Volume Profile 5 anios, AVWAP
  minimo 2 anios y minimo de la ultima correccion >= 20 %, Order Blocks semanales) y `bandas_log_regresion`, mas
  swing lows semanales macro (minimo de +-10 semanas, confirmado).
* Anclas puntuales se ensanchan +-0.25 ATR semanal; se agrupan en clusters de ancho <= 3 ATR semanal, bajo el precio
  (0.5 %-30 % debajo). Rango primario = cluster con mas familias de ancla (volumen, AVWAP, OB, swing, log-reg), empate por cercania.
* Se **excluyen** P/E, EV/FCF y DCF de EDGAR del backtest: `valuation_floors._facts` devuelve el historico completo de hoy
  y no es point-in-time. `avwap_multiciclo` se invoca sin ticker para no consultar el 10-K actual.

### 2.2 Simulacion de la orden
Orden limite GTC en `r_sup`, vigencia 63 ruedas. Llenado si `Low <= r_sup`; entrada = min(Open del dia, r_sup).
* **Absorcion:** en las 5 ruedas desde el toque, vela dentro del rango con volumen > 1.35 x SMA20 previa, cierre en el tercio superior de la vela y Close >= r_inf.
* **Bounce:** +5 % antes de cerrar bajo `r_inf - 3 %`, en 20 ruedas.
* **Retorno forward** 20/60/120/252 ruedas desde el Close, con censura por derecha (se descartan ordenes sin horizonte completo). **MAE** = minimo Low / entrada - 1 a 60 y 252 ruedas.
* **Linea base:** compra al Close de cada fecha de rebalanceo con los mismos horizontes.

### 2.3 Resultados (AAPL, MSFT, META, OXY, GOOGL, UNH; 686 ordenes validas, 236 llenadas)

Tasas (ordenes con ventana de vida completa):

| Ticker | Ordenes | Fills | Fill rate | Fill profundo (<= r_inf) | Dias a fill | Absorcion | Bounce |
|---|---|---|---|---|---|---|---|
| AAPL | 116 | 27 | 23.3 % | 22.2 % | 19.0 | 25.9 % | 51.9 % |
| GOOGL | 109 | 27 | 24.8 % | 7.4 % | 17.7 | 29.6 % | 77.8 % |
| META | 114 | 48 | 42.1 % | 6.2 % | 16.0 | 29.2 % | 70.8 % |
| MSFT | 113 | 20 | 17.7 % | 0.0 % | 12.2 | 25.0 % | 75.0 % |
| OXY | 117 | 68 | 58.1 % | 2.9 % | 11.9 | 23.5 % | 58.8 % |
| UNH | 117 | 46 | 39.3 % | 2.2 % | 12.7 | 26.1 % | 73.9 % |
| **POOL** | 686 | 236 | **34.4 %** | 5.9 % | 14.4 | **26.3 %** | **66.9 %** |

Retornos forward (pool; media / mediana de fills vs linea base):

| Horizonte | n fills | Media fill | Mediana fill | Media base | Mediana base |
|---|---|---|---|---|---|
| 20 ruedas | 236 | 2.5 % | 2.2 % | 1.9 % | 2.3 % |
| 60 ruedas | 236 | 9.0 % | 6.2 % | 5.7 % | 5.1 % |
| 120 ruedas | 229 | 16.3 % | 12.4 % | 11.1 % | 11.1 % |
| 252 ruedas | 214 | 30.3 % | 22.6 % | 24.2 % | 22.5 % |

MAE (pool): a 60 ruedas media -11.8 % / mediana -8.4 % (base: mediana -6.6 %); a 252 ruedas media -21.4 % / mediana -16.3 % (base: mediana -11.9 %).

Lectura: comprar en el rango mejora la media de retorno frente a la base (+3.3 pp a 60 ruedas, +6.1 pp a 252), pero la
**mediana a 252 ruedas es casi identica a la base (22.6 % vs 22.5 %)** y la **MAE es sistematicamente peor** (el rango
"cae" hasta el nivel: se compra tras una caida y el precio sigue cediendo antes de rebotar). OXY muestra mediana negativa a 60/120/252 ruedas.
La ventaja se concentra en las medias y en META/GOOGL/AAPL/MSFT (tendencia alcista del periodo); no hay evidencia
estadistica independiente de alfa por si sola.

## 3. Casos 13F-HR (`case_studies.py`)

### 3.1 Protocolo
1. Se descargan los 13F-HR de cada gestor (EDGAR) y se suma `SharesPrnAmount` del ticker (match por Ticker/CUSIP). La variacion trimestral de acciones (previo ajustado por splits, p. ej. GOOGL 20:1) determina si hubo compra neta. Para los casos institucionales el flag de acumulacion usa solo gestores **activos** (Vanguard y State Street son pasivos de indice y se muestran pero no cuentan). Gestores: Berkshire, Bridgewater, Renaissance, Pershing Square, Tiger Global, Coatue, Third Point, Appaloosa, Soros, Viking, Lone Pine, Baupost (+ Vanguard, State Street como pasivos).
2. **Proxy de ejecucion (limitacion explicita):** el 13F no da fechas de compra. Se usa el rango intercuartil (P25-P75) ponderado por volumen del precio tipico diario del trimestre, junto con VWAP y [Low, High].
3. **Point-in-Time:** rangos con datos anteriores al inicio del trimestre (snapshot estricto) y al inicio de cada mes del trimestre (rodante). "Toque" = el Low del trimestre <= r_sup; "solapa" = el rango PIT intersecta el IQR de ejecucion.
4. **Control:** rango aleatorio del mismo ancho entre -40 % y -0.5 % del precio (Monte Carlo, semilla 0).
5. Si EDGAR no responde, el modulo degrada: evalua las ventanas documentadas con el proxy de precios y marca `verificado_13f = False`.

### 3.2 Resultados (EDGAR disponible: las 16 observaciones de acumulacion estan verificadas con 13F)

Berkshire en AAPL (acciones 13F sin ajustar por el split 4:1 de 2020; precios ajustados): 0 -> 9.81 M (Q1 2016), 15.23 M (Q2), 15.23 M (Q3, sin compra), 57.36 M (Q4), 129.36 M (Q1 2017), 130.19 M (Q2), 134.09 M (Q3), 165.33 M (Q4), 239.57 M (Q1 2018). Los rangos PIT del inicio de cada trimestre estaban 20-30 % bajo el precio (p. ej. 2017Q1: 20.1-21.6 vs IQR de ejecucion 27.8-32.1): **0 de 9 trimestres evaluados (0 de 8 con compra neta) solapan** y solo 2016Q2 toco el rango.

Berkshire en OXY (2022): 0 -> 136.37 M (Q1), 158.55 M (Q2), 194.35 M (Q3), 194.35 M (Q4, sin compra). 0 de 3 trimestres de compra solapan en modo estricto; en modo rodante 2 de 3 (Q1 y Q3), p. ej. 2022Q3 rango PIT 39.7-54.8 vs IQR 56.7-63.2 en estricto.

Acumulacion institucional Q3-Q4 2022 (variacion neta de gestores activos): META solo Q4 (+1.40 M acciones; en Q3 -4.73 M); MSFT solo Q3 (+0.12 M; Q4 -1.97 M); UNH ambos (+0.59 M, +1.58 M); GOOGL solo Q3 (+5.46 M; Q4 -2.35 M). En META el unico solape estricto fue 2022Q3 (rango 129.3-158.9), trimestre sin compra neta activa.

Agregado (16 trimestres con compra neta):

| Metrica | Valor |
|---|---|
| Toque estricto del rango en el trimestre | 18.8 % |
| Solape estricto con el IQR de ejecucion | 0.0 % |
| Solape rodante (3 snapshots) | 25.0 % |
| Solape con algun cluster (no solo el primario) | 12.5 % |
| Control aleatorio (estricto / rodante) | 7.8 % / 34.8 % |
| Distancia mediana VWAP del trimestre a cluster mas cercano | 15.2 % |

Conclusion: los rangos PIT **no** coinciden mejor que el azar con las zonas de compra de los fondos (rodante 25.0 % vs control 34.8 %). Los fondos compraron cerca del mercado y en tendencia; el algoritmo propone zonas estructurales profundas.
Retornos desde el VWAP del trimestre a 252 ruedas fueron positivos en la mayoria de casos (p. ej. AAPL +48 % en 2016Q1, META +207 % desde 2022Q4, OXY +35 % en 2022Q1) pero con excepciones (AAPL -14 % en 2017Q4, OXY -3/-6/-12 % en 2022Q2-Q4).

## 4. Indice de confiabilidad y calibracion (`scoring.py`)

Indice 0-100 como promedio ponderado de cinco componentes (se renormaliza si falta alguno): respaldo 13F/Form 4, flujo STOCK Act (`pts`/15), absorcion, backtest (50 % bounce, 30 % retorno mediano 60d saturado en 15 %, 20 % MAE60) y confluencia (familias de ancla + anclas de valoracion).

Pesos: a-priori mezclados 50/50 con w ~ prior * exp(0.5 * clip(z, -3, 3)):

| Componente | Evidencia empirica | z / t | A-priori | Calibrado |
|---|---|---|---|---|
| respaldo_13f_form4 | solape rodante 25.0 % vs control 34.8 % (n=16) | -0.82 | 25.0 % | 23.4 % |
| stock_act | sin serie historica PIT | n/d | 10.0 % | 11.6 % |
| absorcion | ret60 con absorcion +8.12 % (n=62) vs sin +9.22 % (n=175); bounce 64.5 % vs 68.0 % | -0.36 | 20.0 % | 21.0 % |
| backtest | Spearman(bounce previo PIT del ticker, bounce actual) = -0.014 (n=206) | -0.19 | 30.0 % | 33.0 % |
| confluencia | Spearman(n_familias, ret60) = -0.149 (t=-2.30); t(has_logreg, ret60) = -2.10 | -2.10 | 15.0 % | 11.0 % |

Calibracion del modelo (score PIT = absorcion + historial PIT del ticker + confluencia; n = 237 fills):

| Tercil | n | Score medio | Bounce | Ret60 media | Ret60 mediana | MAE60 mediana |
|---|---|---|---|---|---|---|
| bajo | 79 | 22.5 | 70.9 % | 14.4 % | 10.7 % | -8.0 % |
| medio | 79 | 41.5 | 65.8 % | 4.8 % | 3.6 % | -8.7 % |
| alto | 79 | 65.2 | 64.6 % | 7.6 % | 8.1 % | -9.7 % |

Spearman(score, bounce) = -0.020; Spearman(score, ret60) = -0.028. **El score no discrimina resultados**: ningun componente probado tiene efecto positivo significativo. Los pesos calibrados son por tanto casi los a-priori; no deben interpretarse como evidencia de poder predictivo. El indice sirve como resumen estandarizado de respaldo, no como probabilidad calibrada.

## 5. Limitaciones

* Muestra: 6 tickers de gran capitalizacion, 10 anios mayormente alcistas, sesgo de supervivencia (se eligieron tickers ganadores). Ordenes mensuales con vida de 63 ruedas se solapan: las observaciones no son independientes y los t-stat estan inflados.
* Precios ajustados (`auto_adjust`): los niveles son coherentes en la serie ajustada, pero el ajuste por dividendos usa informacion posterior (distorsion pequena).
* Pisos de valoracion XBRL (P/E, EV/FCF, DCF) y el AVWAP al 10-K no se auditaron (no son PIT): su aporte empirico queda sin medir; `confluencia` solo usa log-regresion como ancla de valoracion.
* 13F: sin fecha de ejecucion (proxy de precio del trimestre, amplio), informacion rezagada hasta 45 dias, no incluye posiciones cortas ni no-EE. UU.; cambios agregados mezclan flujos de clientes; los gestores pasivos se excluyen del flag. Solo 16 trimestres: muestra muy pequena. El control Monte Carlo es un modelo simple.
* El umbral de bounce (+5 %/20d, invalidacion -3 %), la ventana de absorcion (5 ruedas) y las ponderaciones de normalizacion son parametros fijados a priori, no optimizados.
* Sin red: EDGAR (13F, Form 4) y STOCK Act dejan de estar disponibles; el backtest de rangos y los proxies de precio solo requieren yfinance.
* Costos de transaccion, impuestos y liquidez no modelados; Form 4 y STOCK Act no se probaron empiricamente por falta de series PIT.
