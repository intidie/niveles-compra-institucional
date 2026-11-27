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
Orden limite GTC en `r_sup`, vigencia 63 ruedas (las barras posteriores a la fecha de colocacion). Llenado en la primera rueda `k` con `Low <= r_sup`; entrada = min(Open del dia, r_sup). Las definiciones exactas siguientes son las de `evaluar_orden` (`backtest_ranges.py`).
* **Fill profundo (`<= r_inf`):** es `Low[k] <= r_inf`, evaluado **solo el dia del llenado** (`profunda`). No mira las ruedas posteriores: un precio que perfora `r_inf` despues del dia del llenado no cuenta como fill profundo (eso lo recoge el MAE, no esta columna). Esto explica en parte que la cifra sea baja (5.9 % en el pool).
* **Absorcion:** ventana = la vela del toque (`k`) **mas las 5 siguientes**, es decir 6 velas (`range(k, k + 5 + 1)`, truncada al final de los datos); donde este documento dice "5 ruedas desde el toque" se refiere a estas 5 posteriores al toque, ademas de la propia vela del toque. Una vela cuenta como absorcion si cumple a la vez: intersecta el rango (`Low <= r_sup` y `High >= r_inf`; no se exige que quede contenida en el), volumen > 1.35 x SMA(20) del volumen de las 20 ruedas **previas** (la propia vela no entra en la media), cierre en el tercio superior de la vela (`(Close - Low) / (High - Low) >= 2/3`) y `Close >= r_inf` (puede cerrar por encima de `r_sup`).
* **Bounce:** la meta es `High >= entrada x 1.05` y la invalidacion es el **minimo intradia** `Low < r_inf x (1 - 0.03)` (no el cierre). Se recorren las barras `k .. k + 20` (la del llenado y las 20 siguientes) en orden; si en una misma barra se cumplen ambas, cuenta como fallo (la invalidacion se evalua primero). Un llenado con gap cuyo Low ya cae bajo `r_inf - 3 %` el mismo dia es fallo inmediato. Solo se calcula si existen las 20 ruedas posteriores (censura por derecha).
* **Retorno forward** 20/60/120/252 ruedas = `Close[k + h] / entrada - 1`, solo si `k + h` existe en los datos (censura por derecha por horizonte: de ahi n = 229 y 214 a 120 y 252 ruedas). **MAE** = `min(Low[k .. k + h]) / entrada - 1` a 60 y 252 ruedas.
* **Linea base:** compra al Close de cada fecha de rebalanceo con los mismos horizontes. `resumen` la calcula sobre **todas** las fechas de rebalanceo con horizonte disponible (sin filtrar por existencia de rango ni por ventana de vida completa), por lo que su n no coincide con el de los llenados.

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

### 2.4 Nota sobre el tamano de muestra: 236 llenados en 2.3 frente a 237 en la calibracion de 4
Las dos cifras provienen de filtros distintos sobre las mismas 720 filas de ordenes (6 tickers x 120 fechas):
* `resumen` (tablas de 2.3) solo cuenta ordenes con rango y **ventana de vida completa** (`ventana_completa`: la barra de colocacion + 63 ruedas existe en los datos): 686 ordenes validas, **236 llenadas**.
* `evidencia` y `tabla_calibracion` (seccion 4) toman las ordenes llenadas con `ret60` disponible (`llenada & ret60 notna`) y **no filtran por `ventana_completa`**: **237**. La unica diferencia es 1 orden de META colocada el 2026-08-28 y llenada el 2026-08-31: su ventana de 63 ruedas aun no esta completa, pero ya existen 60 ruedas posteriores al llenado, asi que tiene `ret60`. Verificado sobre el cache de ordenes (hay 244 ordenes llenadas en total; las otras 7 sin ventana completa no tienen `ret60` y no entran en ninguno de los dos conteos).
* Otros n de la seccion 4 son submuestras de esos 237: n = 206 (backtest) exige al menos 5 llenados previos ya resueltos del mismo ticker (historial point-in-time); n = 16 (13F) son trimestres de acumulacion, no ordenes.
* La linea base usa todas las fechas de rebalanceo con horizonte disponible, no solo las que generaron un llenado.

## 3. Casos 13F-HR (`case_studies.py`)

### 3.1 Protocolo
1. Se descargan los 13F-HR de cada gestor (EDGAR) y se suma `SharesPrnAmount` del ticker (match por Ticker/CUSIP). La variacion trimestral de acciones (previo ajustado por splits, p. ej. GOOGL 20:1) determina si hubo compra neta. Para los casos institucionales el flag de acumulacion usa solo gestores **activos** (Vanguard y State Street son pasivos de indice y se muestran pero no cuentan). Gestores: Berkshire, Bridgewater, Renaissance, Pershing Square, Tiger Global, Coatue, Third Point, Appaloosa, Soros, Viking, Lone Pine, Baupost (+ Vanguard, State Street como pasivos).
2. **Proxy de ejecucion (limitacion explicita):** el 13F no da fechas de compra. Se usa el rango intercuartil (P25-P75) ponderado por volumen del precio tipico diario del trimestre, junto con VWAP y [Low, High].
3. **Point-in-Time:** rangos con datos anteriores al inicio del trimestre (snapshot estricto) y al inicio de cada mes del trimestre (rodante). "Toque" = el Low del trimestre <= r_sup; "solapa" = el rango PIT intersecta el IQR de ejecucion.
4. **Control:** rango aleatorio del mismo ancho entre -40 % y -0.5 % del precio (Monte Carlo, semilla 0).
5. Si EDGAR no responde, el modulo degrada: evalua las ventanas documentadas con el proxy de precios y marca `verificado_13f = False`.

**Aviso: gestores pasivos y nomina de gestores (estudios de caso frente al screener en vivo).**
* `case_studies.py` excluye del flag de acumulacion a los gestores pasivos de indice, lista exacta `PASIVOS = {"Vanguard Group", "State Street"}` (el flag es `delta_activos > 0`; los pasivos se muestran en el detalle pero no cuentan). El caso Berkshire usa solo `Berkshire Hathaway`; los casos institucionales (META, MSFT, UNH, GOOGL) usan los 14 gestores de `GESTORES_13F` + `GESTORES_EXTRA`, de los cuales 12 son activos.
* El screener en vivo (`main.py`, `sec_edgar.obtener_13f_ticker`) consulta solo **5 gestores** (`GESTORES_13F`): Berkshire Hathaway, Vanguard Group, State Street, Bridgewater y Renaissance Technologies; es decir, como maximo **3 activos** frente a los 12 de los estudios de caso (los otros 9 solo se usan en `case_studies.py`). La fraccion de gestores que acumulan es por tanto muy gruesa en el screener (0, 1/3, 2/3 o 1) y su universo no coincide con el de la evidencia (n = 16 trimestres) que calibro el peso del componente.
* Correccion del screener en vivo (contrato de datos vigente): antes de esta correccion Vanguard y State Street **si** contaban como "acumulando" en el screener, lo que era inconsistente con los estudios de caso. Ahora `obtener_13f_ticker` devuelve todas las filas con una columna booleana `pasivo` y los atributos `attrs['consultados']` y `attrs['fallidos']`; `main.py` calcula `n_con_posicion` y `n_acumulando` **solo con gestores activos** y trata el caso "todos los gestores fallaron" como **dato no disponible** (`None`), no como "0 gestores con posicion" (ver 4.2). La definicion de pasivos esta duplicada en `sec_edgar.GESTORES_PASIVOS` y `case_studies.PASIVOS` (por dominio de modulos) y un test de datos comprueba que coincidan.

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

### 4.1 Componentes, pesos y calibracion empirica

Indice 0-100 como promedio ponderado de cinco componentes (los componentes sin dato se excluyen y los pesos se renormalizan, ver 4.2): respaldo 13F/Form 4 (50 % fraccion de gestores activos con posicion que aumentaron + 50 % compras Form 4 codigo P saturadas en USD 1 M), flujo STOCK Act (`pts`/15), absorcion, backtest (50 % bounce, 30 % retorno mediano 60d saturado en 15 %, 20 % MAE60) y confluencia (60 % familias de ancla de la zona + 40 % anclas de valoracion en la zona; si la valoracion no se consulto, ver 4.2).

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

### 4.2 Datos no disponibles no son cero: exclusion, renormalizacion y cobertura

Regla (implementada en `scoring.py`): un dato que **no se pudo obtener** se representa con `None` y se **excluye**; no se sustituye por 0 ni por un valor neutro. El indice es el promedio ponderado de los componentes con dato, con los pesos renormalizados para sumar 1 (`peso_efectivo`). Un dato obtenido que vale cero (0 USD de compras de insiders, 0 gestores con posicion, 0 puntos STOCK Act, 0 anclas de valoracion en la zona) **si es un dato** y puntua cero.

| Situacion | Tratamiento |
|---|---|
| STOCK Act no consultable (`pts=None`, `disponible=False`) | Componente `stock_act` excluido (no se cuenta como neutro). El dict con `disponible=False` se excluye aunque traiga `pts`. |
| STOCK Act consultado correctamente sin operaciones del ticker | Dato valido (en el estado actual de `stock_act.py`: `pts=0.0`, `disponible=True`); puntua 0/100. Antes de la correccion devolvia 7.5 pts (neutro, 50/100) tanto para este caso como para los fallos, y contaba como dato. |
| Form 4 fallido (`compras_form4_usd=None`) | Se excluye la mitad Form 4 de `respaldo_13f_form4`: el componente vale el 100 % de la mitad de gestores (antes contaba como 0 USD y bajaba el score). |
| Form 4 consultado sin compras de insiders | `compras_form4_usd=0.0`: dato valido, la mitad Form 4 vale 0. |
| 13F sin respuesta de ningun gestor (`n_con_posicion=None`, `n_acumulando=None`) | Se excluye la mitad de gestores: el componente vale el 100 % de la mitad Form 4 (antes la mitad de gestores se contaba como 0 y el componente salia al 50 % de lo debido). |
| 13F consultado, 0 gestores activos con posicion (`n_con_posicion=0`) | Dato valido: fraccion 0. |
| Ambas mitades no disponibles | Componente `respaldo_13f_form4` excluido y pesos renormalizados. |
| `n_con_posicion > 0` pero `n_acumulando=None` | La fraccion no es calculable: mitad de gestores no disponible. |
| Backtest sin fills / sin bounce | Componentes `backtest` y `absorcion` excluidos (NaN). |
| Valoracion no consultada (`n_valoracion=None`; p. ej. falla `calcular_pisos_valoracion`) | Se excluye la mitad de valoracion (40 %) de `confluencia` y la parte de familias se reescala a 0-100: `100 x clip((n_familias - 1) / 3, 0, 1)` (equivale a dividir su parte, `0.6 x (n_familias - 1) / 3`, por su peso 0.6; 4 o mas familias dan 100 y 1 familia da 0). `main.py` pasa `None` en este caso (`n_valoracion_para_scoring`). |
| Valoracion consultada sin anclas en la zona (`n_valoracion=0`) | Dato valido: la mitad de valoracion vale 0 y la confluencia es `100 x clip(0.6 x (n_familias - 1) / 3 + 0.4 x min(1, n_valoracion / 2), 0, 1)` (formula completa sin cambios; el valor por defecto de `n_valoracion` sigue siendo 0). |
| `n_familias=None` (sin zona) | `confluencia` = None: componente excluido. |

Cuando las dos mitades de `respaldo_13f_form4` existen, cada una pesa 50 %; con una sola, esa pesa 100 % del componente. Esto **no** cambia la cobertura (el componente sigue presente), aunque descanse en la mitad de la informacion. Lo mismo ocurre con la valoracion no consultada: `confluencia` sigue presente con solo su parte de familias (reescalada), la cobertura no baja y el componente descansa en menos informacion. Como la parte de familias se reescala, una zona con la valoracion **no consultada** puede puntuar mas en confluencia que la misma zona con la valoracion **consultada y sin anclas** (p. ej. 3 familias: 66.7 frente a 40.0); por eso la confluencia (y el ranking, donde pesa ~55.5 %) tampoco es comparable entre tickers con valoracion consultada y no consultada.

**Cobertura** (`calcular_indice_confiabilidad`): suma de los pesos de los componentes con dato dividida por la suma de todos los pesos. 100 % = los cinco componentes presentes; baja cuando falta alguno (con los pesos calibrados vigentes, sin STOCK Act = 100 % - 11.6 % = 88.4 %; sin SEC = 65.0 %; sin backtest = 46.0 %). Mide cuanta parte del modelo tuvo dato, no la calidad de esos datos. Con cobertura baja el indice es el promedio de lo poco disponible y no es comparable con el de otro ticker con cobertura distinta. Un componente con valor 0 no baja la cobertura (es un dato); solo los ausentes la bajan. La cobertura no se infla con ceros ni neutros: un fallo de STOCK Act, Form 4, 13F o valoracion la reduce o deja el componente con menos informacion, nunca la sube.

Excepciones (i y iii residuales, no corregidas aqui; ii corregida): (i) dentro del componente `backtest`, si hay bounce pero faltan los medianos de retorno o MAE a 60 ruedas se usan 0 y 0.5 (neutro) respectivamente; (ii) valoracion: **corregida**. Antes, si fallaba la etapa de valoracion (`calcular_pisos_valoracion`), la mitad de valoracion de la confluencia (40 %) puntuaba 0 como si se hubiera consultado y no hubiera anclas. Ahora `s_confluencia` acepta `n_valoracion=None` (valoracion no consultada: se excluye esa mitad y se reescala la de familias) y distingue `0` (consultada, sin anclas en la zona: dato valido que puntua 0); `indice_desde_modulos` lo pasa tal cual; (iii) el screener en vivo compara los dos ultimos 13F-HR sin ajustar por splits (`case_studies.py` si ajusta), de modo que un split entre ambos trimestres puede aparecer como acumulacion o reduccion espuria.

## 5. Limitaciones

* Muestra: 6 tickers de gran capitalizacion, 10 anios mayormente alcistas, sesgo de supervivencia (se eligieron tickers ganadores). Ordenes mensuales con vida de 63 ruedas se solapan: las observaciones no son independientes y los t-stat estan inflados.
* Precios ajustados (`auto_adjust`): los niveles son coherentes en la serie ajustada, pero el ajuste por dividendos usa informacion posterior (distorsion pequena).
* Pisos de valoracion XBRL (P/E, EV/FCF, DCF) y el AVWAP al 10-K no se auditaron (no son PIT): su aporte empirico queda sin medir; `confluencia` solo usa log-regresion como ancla de valoracion.
* 13F: sin fecha de ejecucion (proxy de precio del trimestre, amplio), informacion rezagada hasta 45 dias, no incluye posiciones cortas ni no-EE. UU.; cambios agregados mezclan flujos de clientes; en los estudios de caso los gestores pasivos (`Vanguard Group`, `State Street`) se excluyen del flag y el screener en vivo, tras la correccion, tambien los excluye del conteo de "acumulando" (ver el aviso de 3.1; el screener consulta solo 5 gestores, como maximo 3 activos). Solo 16 trimestres: muestra muy pequena. El control Monte Carlo es un modelo simple.
* El umbral de bounce (+5 %/20d, invalidacion -3 % sobre el minimo intradia), la ventana de absorcion (la vela del toque + 5 siguientes) y las ponderaciones de normalizacion son parametros fijados a priori, no optimizados.
* Sin red: EDGAR (13F, Form 4) y STOCK Act dejan de estar disponibles y sus componentes se **excluyen** del indice (la cobertura baja; no se cuentan como cero ni como neutro, ver 4.2); el backtest de rangos y los proxies de precio solo requieren yfinance.
* Costos de transaccion, impuestos y liquidez no modelados; Form 4 y STOCK Act no se probaron empiricamente por falta de series PIT.
* Las metricas por ticker del screener no son comparables con las del pool (ver 6.1).
* **Diseno frente a evidencia (decision abierta).** El reparto de capital (`asignar_capital`: capital x `n_familias` de la zona / suma de `n_familias` de las 3 primeras zonas), el orden N1-N4 (mas familias, luego mas anclas, luego mas cercana) y el ranking de la watchlist dependen de `n_familias` / confluencia. La calibracion propia (seccion 4) **no** hallo que mas familias mejoren el retorno a 60 ruedas: Spearman(n_familias, ret60) = -0.149 (t = -2.30, ademas t inflado por el solapamiento de ordenes), y t(has_logreg, ret60) = -2.10; es el componente con la peor evidencia. Aun asi la confluencia pesa **~55.5 %** del score de ranking: `ranking = 0.5 x confluencia + 0.5 x confiabilidad`, y la confluencia es tambien un componente de la confiabilidad, de modo que su peso total es 0.5 + 0.5 x 0.1096 = 0.5548 (con el peso calibrado 0.1096 de `pesos_calibrados.json` y cobertura 100 %). Con componentes ausentes los pesos se renormalizan y su peso sube (56.2 % sin STOCK Act; 58.4 % con `--sin-sec`; 61.9 % con `--sin-backtest`; 100 % si es el unico componente con dato). **No se modificaron las formulas** (reparto, orden ni ranking): la evidencia desfavorable a la confluencia no respalda por si misma ninguna alternativa concreta (tampoco el resto de componentes discrimina, ver 4.1) y cambiarlas sin evidencia a favor seria arbitrario. Es una decision de diseno abierta que debe reevaluarse con series PIT de valoracion y mas tickers/periodos; hasta entonces, el reparto por familias y el ranking son un criterio de organizacion, no un resultado respaldado empiricamente.

## 6. Screener en vivo (`main.py`) frente a las cifras auditadas

### 6.1 Que audita el screener y por que sus metricas no son comparables con el pool
Las cifras de las secciones 2 a 4 (686 ordenes, 236 llenadas, 34.4 % de llenado, 66.9 % de bounce, etc.) son del **pool**: 6 tickers (AAPL, MSFT, META, OXY, GOOGL, UNH) a **10 anios**, 120 fechas de rebalanceo por ticker, con el historial completo (`"max"`) de yfinance. El screener (`python main.py --ticker T`) calcula un backtest distinto por ticker:
* **Duracion:** `--anios-backtest` vale **5 por defecto** (`backtest_ticker(ticker, 5, df=df)`), es decir unas **60 fechas mensuales** por ticker (5 x 252 / 21 = 60, cada 21 ruedas), sobre el historial de `--periodo` (10y por defecto) y no sobre el historial completo.
* **Alcance:** solo se audita el **rango primario (N1)** (`rango_primario`: mas familias, luego mas anclas, luego mas cercano; es la misma regla con la que se ordena N1). Las zonas N2-N4 y el reparto de capital entre zonas **no** se auditan.
* **Muestra pequena:** descontando las ultimas ~3 fechas sin ventana de vida completa y las fechas sin rango, quedan del orden de 55-57 ordenes por ticker; con las tasas de llenado por ticker del pool (17.7 %-58.1 %), eso son aproximadamente 10 a 33 llenados (estimacion de orden de magnitud, no medida), y menos aun para los horizontes largos por la censura por derecha (las ordenes de los ultimos 12 meses no tienen retorno a 252 ruedas). Con ~10-20 llenados el error estandar binomial de un bounce rate ronda 10-15 puntos porcentuales antes de contar el solapamiento entre ordenes consecutivas, que lo aumenta.
* **Consecuencia:** Fill Rate, Bounce, retornos y MAE por ticker del screener son **mas ruidosos** y **no son comparables** con las cifras del pool (otro universo, otra ventana y otro tamano de muestra); ni siquiera son comparables entre dos tickers del propio screener con distinto numero de llenados. Son una referencia de la historia reciente de ese activo, no una estimacion de la distribucion del pool.
* El componente `backtest` y el componente `absorcion` del indice del screener se alimentan de esa misma muestra pequena (ver 4.2 para su tratamiento cuando no hay llenados).

### 6.2 Modo batch (`--watchlist`)
En modo batch **cada ticker recibe el capital indicado completo** (`analizar_ticker(t, args.capital, ...)` para cada ticker): `--capital` es "capital por ticker" y **no se reparte entre tickers**. Si las ordenes de N activos se llenaran, harian falta N veces el capital. El presupuesto total y su division entre tickers son decision del usuario; el sistema solo reparte el capital de cada ticker entre las zonas de ese ticker (ver la nota de diseno frente a evidencia en la seccion 5).

## 7. Cambios de correcciones (2026-11-26)

Documentacion (este archivo), cada punto verificado contra el codigo; no se modifico ninguna cifra ya citada ni los pesos de `pesos_calibrados.json`; no se volvieron a ejecutar el backtest ni la calibracion:
* 2.2: el bounce usa el **minimo intradia** (`Low < r_inf x 0.97`), no el cierre; umbral sin cambios. Se aclaran la meta (`High >= entrada x 1.05`), la ventana (barras `k .. k + 20`) y el orden de evaluacion dentro de una barra.
* 2.2: la ventana de absorcion son 6 velas (la del toque + 5 siguientes) y la condicion de rango es de interseccion, no de contencion.
* 2.2: definicion exacta de "fill profundo" (`Low[k] <= r_inf` solo el dia del llenado).
* 2.4 (nueva): origen de la diferencia 236 (`resumen`, solo ventana completa) frente a 237 (`evidencia` / `tabla_calibracion`, sin ese filtro): 1 orden de META.
* 3.1: aviso sobre gestores pasivos (lista exacta, exclusion en estudios de caso y en el screener tras la correccion) y sobre la nomina de gestores (5 en el screener, 14 en los casos).
* 4.1 / 4.2: regla "dato no disponible no es cero": STOCK Act fallido, Form 4 fallido y 13F sin respuesta se excluyen y los pesos se renormalizan (la cobertura baja); definicion de cobertura; excepciones residuales.
* 4.2 (valoracion): `s_confluencia(n_familias, None)` (valoracion no consultada) excluye la mitad de valoracion (40 %) y reescala las familias con `100 x clip((n_familias - 1) / 3)`; `n_valoracion=0` sigue siendo dato valido (consultada, sin anclas). La excepcion residual (ii), que decia que la valoracion fallida puntuaba 0, queda corregida y se documenta que la confluencia no es comparable entre valoracion consultada y no consultada.
* 5: nota de diseno frente a evidencia (reparto de capital y ranking dependen de la confluencia; peso real ~55.5 % calculado de `pesos_calibrados.json`; formulas sin modificar; decision abierta).
* 6.1: diferencias screener (5 anios, ~60 fechas, solo N1) frente al pool (6 tickers, 10 anios).
* 6.2: modo batch, capital completo por ticker.

Codigo (`src/validation/scoring.py`): `cargar_pesos` y `main()` cierran los ficheros (`with open`); `s_13f` promedia solo las mitades con dato (gestores activos / Form 4) reponderando 50/50 entre las disponibles y devuelve `None` sin ninguna; `s_stock_act` y `calcular_indice_confiabilidad` tratan `None`/NaN como dato no disponible; `indice_desde_modulos` excluye `stock_act` con `pts=None` o `disponible=False`, acepta `respaldo13f` con claves `None` y `n_valoracion=None` (se pasa tal cual a `s_confluencia`); `s_confluencia` acepta `n_valoracion=None` (ver arriba); se documenta `cobertura`. Pruebas en `tests/test_scoring_fixes.py`.
