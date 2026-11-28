# Guía: qué hacer después de obtener los resultados

Explicación sencilla del resultado de `main.py`. Las cifras provienen de `docs/METODOLOGIA.md` (ejecución real de
`src/validation/`, datos del 2026-11-26) y de los parámetros del código. Los ejemplos con ticker "EJEMPLO" son
**ilustrativos**: no son datos reales.

## 1. Resumen en 5 líneas
1. El sistema te entrega hasta 4 zonas de precio (N1 a N4) por debajo del precio actual, donde coinciden varias huellas históricas (volumen, VWAP anclado, order blocks, mínimos, regresión).
2. También reparte un capital de ejemplo en hasta 3 órdenes límite escalonadas y muestra cuánto sufrió el precio, en el pasado, tras llegar a una zona similar (MAE).
3. NO predice rebotes: en 10 años y 6 acciones, la mediana a 252 ruedas desde el llenado fue casi igual a comprar sin esperar (22.6 % vs 22.5 %).
4. NO da probabilidades: el índice de confiabilidad (0-100) no separó buenos de malos resultados.
5. Úsalo para planificar entradas y medir riesgo con disciplina. La decisión y el riesgo son tuyos.

## 2. Cómo leer los resultados en 2 minutos

| Campo | Qué significa (en simple) | Dónde lo ves |
|---|---|---|
| N1-N4 | Zonas ordenadas por **fuerza** (más familias coinciden; si empatan, más anclas; luego la más cercana). **No** por cercanía. Máximo 4. | Consola, `_zonas.csv` |
| r_inf / r_sup | Borde bajo y borde alto de la zona. Tu orden límite va en **r_sup**. | Consola, gráfico |
| Distancia % (`dist. a r_sup`) | Qué tan abajo del precio actual está **r_sup**, el precio de tu orden límite GTC. `-12.0 %` = 12 % más barato. Es la **misma referencia** en consola, gráfico, UI y export (nunca al centro de la zona). Las zonas están entre 0.5 % y 30 % bajo el precio. | Consola, gráfico, UI, export (`dist_pct`) |
| Familias | Tipos distintos de ancla que coinciden: volumen, avwap, ob, swing, logreg. Más familias = zona con más "razones", no más seguridad. | Consola |
| Capital y acciones | Reparto del `--capital` **de ese ticker** entre sus 3 primeras zonas, proporcional a sus familias. Acciones = capital / r_sup. N4 recibe $0. En modo watchlist cada ticker recibe el capital completo (no se reparte entre tickers). | Consola |
| Fill Rate | De cada 100 órdenes en el historial, cuántas se llenaron en 63 ruedas. Solo del **rango primario (N1)**, con un backtest de 5 años (`--anios-backtest`). | Consola (línea Backtest), export (`bt_fill_rate`) |
| Bounce Rate | Tras llenarse: % de casos que subió +5 % antes de caer bajo r_inf menos 3 %, dentro de 20 ruedas. | Export (`bt_bounce`) |
| MAE | Peor caída desde la entrada (mínimo intradía) en 60 o 252 ruedas. | Export (`bt_mae60_mediana`; JSON: mae252) |
| Confluencia (0-100) | 60 % por familias de la zona + 40 % por anclas de valoración en la zona (máx. 2). | Consola |
| Confiabilidad (0-100) | Promedio ponderado de 5 componentes (13F/Form 4, STOCK Act, absorción, backtest, confluencia). Un componente sin dato se **excluye**; no cuenta como 0 ni como neutro. | Consola, UI |
| Cobertura | Qué parte de los pesos tenía dato. 100 % = los 5 componentes presentes. Junto a ella sale `sin dato: ...` con los componentes excluidos. | Consola (`cobertura`), UI, export (`confiabilidad_sin_dato`) |
| Ranking | Promedio simple de Confluencia y Confiabilidad: ayuda para ordenar, no una señal. Solo en modo watchlist. | Tabla batch, pestaña Watchlist |
| Avisos | Datos que faltan o son poco fiables: `muestra pequeña: N fills`, `13f: no respondieron: ...`, `STOCK Act no disponible`. | Consola (`[aviso]`), UI, export (`avisos`) |
| Aviso earnings | Alerta si el reporte de resultados está a 10 días o menos. | JSON (`earnings`) y `app_ui.py` |

Nota: en consola salen zonas, confluencia, confiabilidad (con cobertura y `sin dato`), una línea de Backtest, los
avisos y el **conteo** de etapas con fallo. Retornos y MAE a 120/252 ruedas están en el CSV/JSON exportado. El
backtest del screener es solo del **rango primario (N1)** y usa 5 años por defecto (`--anios-backtest 5`, unas 60
fechas mensuales): una muestra pequeña y ruidosa por ticker, **no comparable con el pool** de `docs/METODOLOGIA.md`
(6 tickers, 10 años). Con menos de 10 fills sale el aviso `muestra pequeña: N fills`: trata esas cifras como ruido.

### Ejemplo ilustrativo (ticker ficticio, no son datos reales)
```
=== EJEMPLO | Precio $100.00 | Capital por ticker $1,000 ===
  N1: $85.00 - $88.00 (dist. a r_sup -12.0%) | 3 familias [avwap, ob, volumen] | $500 -> 5.682 acc. @ $88.00
  N2: $92.00 - $94.00 (dist. a r_sup -6.0%)  | 2 familias [swing, volumen]     | $333 -> 3.546 acc. @ $94.00
  N3: $79.00 - $81.00 (dist. a r_sup -19.0%) | 1 familias [logreg]             | $167 -> 2.058 acc. @ $81.00
  N4: $70.00 - $72.00 (dist. a r_sup -28.0%) | 1 familias [swing]              | $0 -> 0.000 acc. @ $72.00
  Confluencia: 60.0/100 | Confiabilidad institucional: 42.5/100 (cobertura 100%) | Ranking: 51.3
  Backtest (rango primario (N1), backtest de 5 años; no comparable con el pool de docs/METODOLOGIA.md): 14 fills de 40 ordenes | fill 35% | bounce 67% | MAE60 mediana -8%
```
* N1 es la zona más "fuerte" (3 familias), pero N2 está más cerca y se llenaría antes. El número N no indica cercanía.
* Reparto: familias 3, 2 y 1 suman 6, así que 50 %, 33 % y 17 % de $1,000. N4 queda fuera del reparto.
* Confiabilidad ilustrativa: respaldo 25, STOCK Act 50, absorción 26, backtest 57, confluencia 60. Con los pesos
  calibrados da 42.5. Ranking = (60 + 42.5) / 2 = 51.3. Un 42.5 **no** significa "42.5 % de probabilidad".
* Backtest ilustrativo del ticker: Fill Rate 35 %, Bounce 67 %, MAE60 mediano -8 %, con 14 fills (por encima de 10,
  así que no sale el aviso de muestra pequeña). Son valores parecidos al pool real (34.8 %, 61.2 %, -9.1 %) solo
  para que el ejemplo sea creíble; el pool **no** es comparable con la cifra de un ticker (otra muestra y otra ventana).

## 3. Checklist paso a paso

### (a) Revisar avisos, errores, datos faltantes y cobertura
- [ ] Si la consola dice "Advertencias: N etapa(s)", abre el CSV (columna `errores`) o el JSON. Etapas posibles: mercado, earnings, zonas, valoracion, form4, 13f, stock_act, backtest, scoring.
- [ ] Lee las líneas `[aviso]` de la consola (columna `avisos` del CSV): `muestra pequeña: N fills` (menos de 10 ejecuciones simuladas: Bounce, Fill y MAE son ruido), `13f: no respondieron: X, Y`, `STOCK Act no disponible` y, si falló la valoración, `valoracion no disponible`.
- [ ] Si falla "mercado" no hay zonas; en batch el ticker aparece en "Sin datos" y queda fuera del ranking.
- [ ] Mira la **cobertura** y la lista `sin dato: ...`. Con los pesos actuales, sin STOCK Act queda ~88 %, `--sin-sec` ~65 % y `--sin-backtest` ~46 %. Con poca cobertura el índice es el promedio de lo poco que hay; no lo compares con otro ticker de cobertura distinta (la pestaña Watchlist lo avisa).
- [ ] **Un dato no disponible no cuenta como cero ni como neutro; se excluye y la cobertura baja:**
  - STOCK Act que no responde (`pts=None`, `disponible=False`) se excluye (ya no vale 7.5 neutro). Si se consultó bien y no hay operaciones del ticker, da 0 pts y **sí** es dato (puntúa 0).
  - Form 4 fallido ya no cuenta como 0 USD: se excluye esa mitad del componente 13F/Form 4 (queda la mitad de gestores). Consultado sin compras de insiders sí vale 0.
  - 13F: los gestores que no respondieron se listan en el aviso; si no respondió ninguno, o no respondió ningún gestor **activo** (aunque sí respondan Vanguard y State Street, que no cuentan), la mitad de gestores es "no disponible" (no "0 gestores"): sale el aviso `13f: no respondió ningún gestor activo (...)`. Si respondió al menos un activo y ninguno tiene posición, 0 es un dato válido.
  - Valoración (P/E, EV/FCF, DCF): si no se pudo consultar (yfinance), esa parte de la confluencia se **excluye** y el score se reescala con las familias, con el aviso `valoracion ... no disponible`. Si se consultó y ninguna ancla cae en la zona, vale 0 (es un dato).
- [ ] El 13F del screener cruza solo 5 gestores (Berkshire, Vanguard, State Street, Bridgewater, Renaissance). Vanguard y State Street son pasivos de índice y **ya no cuentan** como "acumulando" (ni en el conteo ni en los marcadores del gráfico): quedan 3 activos, así que la fracción es muy gruesa (0, 1/3, 2/3 o 1).
- [ ] Limitaciones que aún quedan: el 13F en vivo compara los dos últimos 13F-HR **sin ajustar `cambio_pct` por splits** (`case_studies.py` sí lo hace), así que un split entre ambos trimestres puede verse como compra o venta. Dentro del componente backtest, con bounce disponible pero sin retorno o MAE mediano a 60 ruedas se usan 0 y 0.5 (excepción residual). La lista `sin dato` se deriva de los componentes que faltan en el detalle del índice.

### (b) Revisar la proximidad de earnings
- [ ] Busca `earnings` en el JSON (o las tarjetas de `app_ui.py`): la consola de `main.py` no lo imprime.
- [ ] Si faltan 10 días o menos hay alerta; el propio mensaje del sistema dice "evitar dejar órdenes límite GTC expuestas a gap de resultados" (y "pausar órdenes límite" si reportan hoy).
- [ ] Si dice "No disponible", confirma la fecha en la web de la empresa.
- [ ] El backtest no tiene filtro de earnings: no mide cómo se comportan las órdenes alrededor de resultados.

### (c) Contrastar con el gráfico y los marcadores
Abre `resultados/<TICKER>_niveles.png`.
- Banda de color = zona; línea discontinua = **r_sup** (tu límite GTC); la etiqueta `N1: $inf-$sup | dist. a r_sup -X% | …` mide la distancia de r_sup al precio actual, igual que la consola, la UI y el export.
- Triángulo verde = compra de insider Form 4 código P (últimos 365 días, hasta 60 filings).
- Diamante y banda violeta = trimestre 13F con compra neta de algún gestor **activo** (Vanguard y State Street no cuentan; el 13F no da fecha exacta y llega con hasta 45 días de rezago).
- Barras naranjas laterales = nodos de alto volumen (HVN) de los últimos 5 años. El precio del gráfico está ajustado por splits y dividendos.
- Pregúntate: ¿el precio ya reaccionó ahí antes? ¿La zona queda muy lejos? Los marcadores 13F/Form 4 son contexto, **no** confirmación (ver sección 5).

### (d) Cuánto capital usar y en cuántos tramos
- `--capital` es el dinero para **ese ticker** (por defecto $1,000), no tu patrimonio.
- Fórmula del sistema: `capital × familias de la zona / suma de familias de las 3 primeras zonas`. Es un criterio simple (más confluencia, más peso), **no probado**: en el backtest, más familias no dio mejores retornos a 60 ruedas (Spearman(n_familias, ret60) = -0.012, sin efecto detectable). Además, con las zonas estrechas solo el 9.4 % de las órdenes tiene 3 o más familias (antes 40.3 %): lo normal son zonas de 1 o 2 familias, así que el reparto se acerca a partes iguales y a veces solo hay 1 o 2 zonas.
- Es una guía, no una orden: puedes dividir a partes iguales o reducir tramos lejanos.
- En modo watchlist **cada ticker recibe el capital completo** (`analizar_ticker(t, args.capital)`; la ayuda de `--capital`, el encabezado del ranking, el export y la pestaña Watchlist lo dicen). Si escaneas 10 y todo se llena necesitarías 10 veces el capital. Decide primero tu presupuesto total y divídelo tú.
- El sistema no calcula stop ni % de riesgo. Traduce el MAE: a 252 ruedas el peor punto fue -16.3 % (mediana) y -21.4 % (media). En $1,000 son unos $163 a $214 de pérdida temporal. Dimensiona tu posición según lo que podrías soportar.

### (e) Órdenes límite escalonadas
- Orden límite: "compra solo si el precio baja a X o menos". **GTC** (Good Till Canceled): sigue activa hasta que se llene o la canceles; algunos brókers la vencen antes (verifica con el tuyo).
- El backtest asume: orden GTC en **r_sup**, vida de **63 ruedas** (~3 meses). Cuenta como llenada si el mínimo del día es menor o igual a r_sup. El precio de entrada es el menor entre la apertura y r_sup (si abre con gap hacia abajo, compras más barato que tu límite).
- Escalonar = una orden por zona, con las acciones indicadas. En el ejemplo: 5.682 @ $88.00, 3.546 @ $94.00 y 2.058 @ $81.00 (si tu bróker no admite fracciones, redondea).
- Solo se auditó el rango primario (N1), con 5 años de historial por defecto (unos 10 a 33 fills, estimación; la consola muestra los reales); la combinación de órdenes y las zonas N2-N4 no se probaron una por una.
- Expectativas del pool: se llena ~1 de cada 3 (34.8 %); MSFT 18.6 % y OXY 59.5 %; tarda 15.1 ruedas de media. Las zonas actuales son más estrechas (mediana ~4 % del precio, tope 5 % y 2 ATR) y por eso se tocan menos: en 32 tickers el fill rate bajó 3.7 puntos frente al diseño anterior, y más en acciones de baja volatilidad. Menos llenados es el precio de una zona más precisa; no se demostró mejor rendimiento.
- El backtest es optimista: un toque exacto cuenta como llenado y no modela comisiones, impuestos ni liquidez.
- **Si no se llena** en ~63 ruedas: no la dejes olvidada. Cancela, recalcula el screener y coloca órdenes nuevas con las zonas vigentes. No subas el límite "para asegurar" el llenado.

### (f) Si el precio toca la zona, la perfora o nunca llega
| Situación | Qué mostró el historial (pool, 6 tickers) | Qué hacer |
|---|---|---|
| **Toca y se llena** | MAE60 mediano -9.1 % (media -12.2 %); MAE252 mediano -15.9 % (media -21.1 %). Rebote +5 % en 20 ruedas: 61.2 % (casi 4 de cada 10 no lo logró). | Asume que el precio **probablemente baje más** antes de subir: compraste tras una caída. Ejemplo: entrada $88.00 con MAE60 mediano -9.1 % = mínimo típico cerca de $80.00 (la mitad de los casos fue peor). No agregues dinero fuera del plan; para eso están las zonas inferiores. |
| **Confirmación vs. recibir el golpe** | Hubo "absorción" (vela de rechazo con volumen > 1.35 x SMA20, cierre en el tercio superior, en las 6 velas desde el toque: la del toque y las 5 siguientes) en 28.7 % de los llenados. Con absorción el retorno medio a 60 ruedas fue mayor (+10.9 % vs +5.2 % sin ella), pero esa vela se mide **después** de entrar: no es una señal disponible al colocar la orden. Midiendo desde la 6.ª rueda la diferencia es +9.5 % vs +4.5 % (p≈0.06): sugerente, no concluyente (posible look-ahead; límite abierto). | Esperar confirmación **no tiene respaldo demostrado** (la asociación con el retorno es post-entrada y débil) y sale más caro. Elige antes una vía (límite y aceptar MAE, o entrar tras confirmar y aceptar menos llenados) y no la cambies a mitad. Ninguna regla de confirmación fue probada. |
| **La perfora** | "Rebote fallido" = mínimo bajo r_inf menos 3 % (ej. r_inf $85 -> $82.45) en 20 ruedas. Es una regla de medición, no un stop probado. Solo el 11.4 % de los llenados ya tocaba r_inf el mismo día del llenado. | Es la señal de que la estructura dejó de sostener. Pausa los tramos pendientes, recalcula zonas, revisa fundamentales (nuevo 10-Q) y decide con una regla escrita de antemano. |
| **Nunca llega** | ~65.6 % de las órdenes no se llenó en 63 ruedas. El costo de oportunidad de esos casos no se midió. | No perseguir: no subir el límite ni comprar por impulso. Recalcula en la fecha prevista. Acepta que puedes quedarte fuera. |

### (g) Cuándo reevaluar o recalcular
| Evento | Por qué |
|---|---|
| Cada ~21 ruedas (1 mes) | Es la cadencia del backtest: rango recalculado cada 21 ruedas y orden con vida de 63. |
| Nuevo 10-Q / 10-K | Cambian P/E propio, EV/FCF y DCF, que alimentan las anclas de valoración de la confluencia (no auditadas en el backtest). |
| Nuevo 13F trimestral | Hasta 45 días después del cierre del trimestre; es información rezagada. |
| Cambio de estructura | Nuevo mínimo o máximo, o zona perdida: el perfil de volumen y los mínimos se recalculan y los N cambian. |
| Tras ejecutar una zona | Esa zona ya se "consumió"; recalcula antes de poner órdenes nuevas. |
| Earnings en menos de 10 días | Revisa órdenes GTC abiertas (ver b). |

### (h) Gestión a largo plazo
- **Horizontes:** 60, 120 y 252 ruedas = ~3 meses, ~6 meses y ~1 año. Medianas desde el llenado: 6.2 %, 12.4 % y 22.6 % (línea base: 5.1 %, 11.1 % y 22.5 %). Vienen de 6 empresas grandes ganadoras en 10 años alcistas: **no son una expectativa futura**.
- **Tamaño de posición:** el que aguantes con una caída de -16 % a -21 % en el primer año (MAE252).
- **Diversificación:** el backtest cubrió 6 tickers de gran capitalización. No concentres todo en un ticker ni en un sector. Reparte tu presupuesto total entre varios tickers de la watchlist.
- **Revisión periódica:** mensual (zonas y órdenes vivas), trimestral (10-Q, 13F, earnings) y anual (tesis). Anota fecha, zona, precio de entrada y MAE vivido para compararlos con lo esperado.

## 4. Cómo usar el ranking de la Watchlist
- Ranking = promedio de Confluencia y Confiabilidad (si falta una, usa la otra; sin score va al final). Se muestran `--top` activos (10 por defecto) con su Zona 1 (= N1), la distancia a r_sup y la cobertura.
- **Es una ayuda de organización, no una señal.** Úsalo para decidir qué ticker revisar primero y dedicarle tiempo de análisis. No para decidir cuánto comprar ni cuál subirá.
- Cada ticker lleva el **capital completo**: el ranking no reparte dinero entre tickers.
- Qué NO concluir:
  - La Zona 1 puede estar muy abajo: mira la columna de distancia a r_sup (consola y pestaña Watchlist) o `z1_dist_pct` del CSV.
  - El ranking está **dominado por la confluencia**: pesa 56.7 % del score (0.5 directo + 0.5 x 0.1342 como componente de la confiabilidad; el peso sale de `pesos_calibrados.json`, con cobertura 100 %); sube con menos cobertura (59.6 % con `--sin-sec`, 65.5 % con `--sin-backtest`). Ese componente no tiene respaldo propio (Spearman con ret60 = -0.012, t = -0.18): **no hay evidencia de que más familias mejoren el retorno**, y como ahora los valores altos de confluencia son raros (solo 9.4 % de las órdenes con 3 o más familias), la mayoría de los activos tendrá confluencia baja o media.
  - Score alto no significa mejores resultados (ver sección 5).
  - No incorpora calidad del negocio, valoración completa ni earnings.
  - Comparar scores con coberturas distintas (SEC o backtest apagados, componentes `sin dato`) no es justo.

## 5. Lo que los datos dicen con honestidad
| Hallazgo (METODOLOGIA.md) | Dato | Consecuencia práctica |
|---|---|---|
| El índice no discrimina resultados en conjunto | Spearman(score, ret60) = +0.154 pero Spearman(score, bounce) = -0.084 (n = 236 llenados). Terciles bajo/medio/alto: bounce 65.0 / 61.0 / 57.0 %; ret60 media 4.5 / 5.4 / 10.6 %; ret60 mediana 1.4 / 5.6 / 10.3 %; MAE60 mediana -9.1 / -8.7 / -10.0 %. | El retorno a 60 ruedas sube algo con el score, pero el rebote y la MAE empeoran y el t está inflado por el solapamiento. No es una probabilidad: no lo uses para elegir ni para dimensionar. |
| Absorción: posible look-ahead | El peso de la absorción (33.2 %) sale de una calibración que mide la señal en las 5 velas **posteriores** al llenado, es decir después de entrar. El auditor midió desde el cierre k+5: retorno medio con absorción +9.5 % vs sin +4.5 % (t≈1.9, p≈0.06). | Sugerente pero no concluyente; es una limitación abierta (la fórmula no se cambió). No leas el peso como poder predictivo. |
| Mediana a 252 ruedas ≈ línea base | Llenado 22.6 % vs base 22.5 %. Media 30.3 % vs 24.2 %. A 20 ruedas, mediana 2.2 % vs 2.3 %. | La ventaja está en las medias, no en la mediana. No hay evidencia independiente de alfa. |
| MAE peor que la base | 60 ruedas: -8.4 % vs -6.6 %. 252 ruedas: -16.3 % vs -11.9 %. | Esperar el nivel implica comprar mientras cae. Cuenta con sufrir más que si comprases "a mercado". |
| Caso OXY | Mediana negativa a 60/120/252 ruedas. | Una zona puede fallar durante meses. |
| Muestra 13F pequeña | 16 trimestres con compra neta. Solape estricto 6.2 % (1 de 16); rodante 25.0 % vs 29.8 % de un control aleatorio; toque estricto 37.5 %. | Los fondos compraron cerca del mercado y en tendencia; el algoritmo propone zonas profundas. Los niveles **no** coinciden mejor que el azar con las compras 13F. No los uses como "dónde compra el dinero grande". |
| Sesgos | 6 tickers grandes y ganadores (supervivencia), 10 años mayormente alcistas, órdenes mensuales solapadas (t-stat inflados), sin costos/impuestos/liquidez, Form 4 y STOCK Act sin probar, valoración XBRL no auditada, umbrales fijados a priori (+5 %/20 d, -3 %, ventana de absorción de 6 velas). | Los números son orientativos y optimistas. Los pesos calibrados son casi los a priori: no son evidencia de poder predictivo. |
| Backtest del screener ≠ pool | Solo rango primario (N1), 5 años por defecto, ~60 fechas mensuales: del orden de 10 a 33 fills por ticker (estimación). El pool usa 6 tickers y 10 años. | Fill, Bounce y MAE de un ticker son más ruidosos y **no comparables** con el pool; con menos de 10 fills la consola avisa `muestra pequeña`. |

Conclusión práctica:
- Los niveles sirven como **herramienta de disciplina**: definir de antemano dónde estaría tu entrada, cuánto repartes y cuánta caída aceptas. No son una señal predictiva.
- No sobreponderes el índice. Según la propia metodología, es "un resumen estandarizado de respaldo, no una probabilidad calibrada".
- Tu mejor control de riesgo es el tamaño de posición, la diversificación y la paciencia, no el número 0-100.

## 6. Errores comunes
| Error | Por qué es un problema |
|---|---|
| Comprar a mercado "porque tocó" la zona | El backtest asume orden límite en r_sup; comprar a mercado en otro punto es otra estrategia, no auditada. |
| Ignorar el MAE | A 252 ruedas el peor punto fue -16.3 % (mediana) y -21.4 % (media). Si no lo soportas, reduce el tamaño. |
| Concentrar todo en un ticker | El capital es por ticker; el sistema no diversifica por ti. |
| Sumar el capital de varios tickers como un solo presupuesto | En modo watchlist cada ticker recibe el capital completo: con 10 tickers llenos necesitarías 10 veces ese capital. |
| Comparar scores con cobertura distinta | Un score con backtest y otro sin él (o con componentes `sin dato`) promedian cosas distintas. |
| Tratar el ranking como señal | Está dominado por la confluencia (56.7 %), que no mostró relación positiva con el retorno. Solo ordena tu lista de revisión. |
| Leer el backtest del ticker como el pool | 5 años, solo N1 y pocos fills; con `muestra pequeña: N fills` es ruido. |
| Olvidar los earnings | La consola no los imprime; un gap de resultados puede llenar o romper la orden. |
| Tratar el índice como probabilidad | 42.5 no es "42.5 % de acierto"; el índice no discrimina resultados. |
| Creer que N1 es la más cercana | N1 es la más "fuerte"; puede estar a -20 % y no llenarse en meses. |
| Dejar una GTC olvidada | Tras ~63 ruedas las zonas ya cambiaron: recalcula. |
| Leer 13F/Form 4 como confirmación | El 13F no da fecha de compra y va con rezago; los rangos no coincidieron mejor que el azar. |

## 7. Glosario
| Término | Significado |
|---|---|
| Rueda | Sesión bursátil. 21 ruedas ≈ 1 mes, 63 ≈ 3 meses, 252 ≈ 1 año. |
| Orden límite / GTC | Compra solo a un precio máximo. GTC = permanece activa hasta cancelarla. |
| Zona (r_inf, r_sup) | Rango de precio donde coinciden varias anclas (ancho máx. 2 ATR semanales y 5 % del precio; ATR = rango típico de una semana). |
| Volume Profile / HVN | Mapa de dónde se negoció más volumen (Steidlmayer). HVN = nodo de alto volumen. |
| Anchored VWAP (AVWAP) | Precio medio ponderado por volumen desde un evento clave (Shannon). |
| Order Block | Zona semanal de actividad fuerte aún no "mitigada" (concepto Smart Money; sin respaldo académico formal). |
| Confluencia | Cuántas familias de ancla coinciden en una zona (volumen, avwap, ob, swing, logreg). |
| Fill Rate | % de órdenes límite que se llenan dentro de su vida (63 ruedas). En el screener, del rango primario (N1) y 5 años. |
| Bounce Rate | % de llenados que suben +5 % antes de perder r_inf menos 3 %, en 20 ruedas. |
| MAE | Maximum Adverse Excursion: la peor caída desde la entrada (mínimo intradía). |
| Línea base | Comprar al cierre en cada fecha de recálculo, sin esperar la zona. Sirve para comparar. |
| 13F-HR / Form 4 (P) | 13F: tenencias trimestrales de gestores grandes, sin fecha de compra; el screener cuenta solo los gestores **activos** (no Vanguard ni State Street, pasivos de índice). Form 4 código P: compras de insiders en mercado abierto (SEC EDGAR). |
| Cobertura / `sin dato` | Parte del peso de la confiabilidad que tuvo dato. Lo que no se pudo consultar se excluye (no es 0 ni neutro) y baja la cobertura. |
| `dist. a r_sup` | Distancia % del límite GTC (r_sup) al precio actual; negativa = por debajo. Misma referencia en consola, gráfico, UI y export. |

---
**Material educativo; no constituye asesoría financiera ni recomendación de inversión.**

## Nota sobre zonas más estrechas (cambio 2026-11-26)

- Las zonas ahora son más estrechas (HVN acotado a su pico, Order Block acotado a 1 ATR desde su techo, ancho máximo 2 ATR y 5 % del precio; las anclas sobre el precio o con menos de la mitad de su rango en la banda se descartan). Ejemplo real con fecha 2026-11-25 (NVDA): N1 pasó de 104.51-114.81 (más una AVWAP sobre el precio) a un único N1 107.35-114.81.
- Consecuencias: **menos zonas por activo** (a veces 1 o 2), **menor fill rate** (-3.7 puntos en 32 tickers; más en baja volatilidad) y valores altos de confluencia más raros. La calidad medida no empeoró de forma distinguible del ruido, pero tampoco hay una mejora demostrada.
- El "núcleo" aún más estrecho de cada zona existe en los datos, pero no se usa como rango operativo.
- La tabla de la interfaz muestra el ancho de cada zona en % del precio.

## Nota sobre el Valor Justo (pestaña "Valor Justo")

- La pestaña **Valor Justo** (y la línea `Fair Value Medio: $XXX (+XX% Upside)` del gráfico de Análisis, con el interruptor "Mostrar Valor Justo", activado por defecto, que dibuja SOLO el Escenario Medio; si el ticker no tiene datos aún, el botón "Calcular valor justo" de la pestaña los obtiene en segundos) calcula 15 modelos de valoración (DCF, múltiplos, dividendos, EPV, Graham y Lynch) con los estados financieros de SEC EDGAR. Es una implementación propia «estilo InvestingPro»: no replica sus cifras.
- Hay **dos escenarios, Medio y Pesimista** (no hay optimista). Son **supuestos de escenario, no predicciones**: el WACC sale de ^TNX, beta de Blume y ERP de 4,5 % (Medio acotado a 8,0-11,5 %; Pesimista = Medio + 200 pb, mínimo 12,5 %); el DCF es de 10 años en 2 etapas con convención de mitad de año; el crecimiento se topa a 42 % (Medio) / 15 % (Pesimista), el margen a 55 % / 30 % y g a 3,5 % / 2,5 %; los múltiplos mezclan la mediana histórica propia con un benchmark sectorial fijado a mano.
- **Cómo leer las cuatro tarjetas:** Precio actual; Valor justo (Medio); Piso fundamental (Pesimista) y Diagnóstico: «Sobrevalorada» si el precio está 10 % o más sobre el valor justo Medio, «Infravalorada» si está 10 % o más bajo, «En línea» entre ambos. Con incertidumbre Alta o menos de 3 modelos el diagnóstico sale atenuado.
- **Cuidado con los extremos.** En negocios de hipercrecimiento el Medio puede quedar muy por encima del precio (NVDA: ~220 USD frente a ~149 USD, con DCF de 10 años entre 397 y 475). El Pesimista es un **estrés duro, no un «peor caso realista»** (en KO queda ~61 % bajo el precio). Ninguno es un precio objetivo ni una predicción.
- **Es un valor estimado con supuestos, no es un precio objetivo ni una predicción**, y es independiente de las zonas N1-N4: no las mueve ni entra en los scores. Úsalo como contraste: si una zona cae por debajo del valor justo con margen de seguridad, la estructura de precio y la valoración coinciden; si discrepan, revisa los supuestos.
- **Gráfico histórico con bandas (±10/20/30 %):** es una reconstrucción con el modelo actual. Cada punto usa solo reportes ya publicados (causal) y el último punto es el valor justo de hoy (ancla, lo único no point-in-time). Rf, ERP, beta, sector y los sliders son los vigentes en todas las fechas (no point-in-time). La línea punteada a 12 meses es ilustrativa, no una predicción. En la versión web de presentación solo hay 3 años de historia.
- Mira la **incertidumbre** antes que el número: "Alta" significa que los modelos divergen (o que quedan menos de 3). Los sliders de WACC, g y margen de seguridad muestran lo sensible que es el resultado (el valor terminal domina los DCF).
- No aplica bien a bancos y aseguradoras, a empresas con pérdidas ni a dividendos simbólicos (cada modelo excluido muestra su motivo en el acordeón de explicación). No fue auditado con backtesting como los niveles técnicos.
- El consolidado **iguala el peso de las familias** (1/3 DCF, 1/3 Múltiplos, 1/3 Valor/Dividendos) tras filtrar outliers con IQR; la mediana y el promedio simple se muestran solo como información complementaria. Los modelos **no son independientes** (los 6 DCF comparten WACC, g y crecimiento; los de Valor comparten EPS y valor contable). El **DCF inverso** indica qué WACC o crecimiento descuenta el precio actual.
- Los modelos con un **tope activo** (múltiplo, crecimiento, margen, beta o WACC acotados) dependen del tope "por construcción". En empresas cíclicas el flujo se normaliza (media de varios años) en vez de extrapolar un pico.
- Si la moneda de reporte difiere de la de cotización (por ejemplo TSM: TWD frente a USD) los modelos no aplican (no hay conversión FX); si los datos por acción de otra clase de acciones no cuadran con el precio (BRK-B) se descartan y el modelo lo explica.
