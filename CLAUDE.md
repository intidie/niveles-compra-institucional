# Proyecto: Screener Cuantitativo de Niveles Institucionales (Largo Plazo)

## Reglas Globales de Arquitectura y Rigor
1. **Invarianza de Niveles:** Ningún nivel de compra puede depender del cierre intradía aleatorio; todo soporte debe anclarse a estructuras acumuladas (HVN, Swing Lows macro, Anchored VWAP de ciclo, Order Blocks no mitigados o múltiplos de valoración histórica/DCF).
2. **Fuentes Primarias:** Priorizar SEC EDGAR (edgartools) sobre yfinance para estados financieros (10-Q/10-K), tenencias institucionales (13F-HR) y compras reales de ejecutivos (Form 4, filtrando código P de compra en mercado abierto).
3. **Verificación Empírica Obligatoria:** Todo rango propuesto debe ser auditado contra:
   - Backtesting histórico de órdenes límite (Fill Rate, Max Drawdown posterior al toque y retorno a 60/120/252 ruedas).
   - Casos prácticos verificables de grandes fondos (13F) y mesas institucionales.
4. **Control de Contexto:** Cada subagente solo modifica los archivos asignados a su módulo en src/. No reescribir archivos fuera de su dominio.
