#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PANEL REBOTE IBEX — reglas validadas con backtest de 5 anos (35 valores).

CRITERIOS (no elegidos a dedo: salieron de un barrido de parametros)
  1. TENDENCIA      cierre > MA200            -> corrige dentro de tendencia, no se derrumba
  2. CAIDA          a <=3% del minimo de 40 sesiones (8 semanas)
  3. CONFIRMACION   la vela cierra por encima del maximo del dia anterior
                    (sin esto el sistema pierde dinero: es la compuerta clave)
  4. RECORRIDO      R:R >= 1,2 hasta el objetivo
  5. LIQUIDEZ       >= 3 M EUR/dia
  6. FUNDAMENTAL    se muestra para juicio humano (no descarta automaticamente)

EJECUCION EN ETORO
  Entrada  : orden de MERCADO en la apertura del dia siguiente (9:00 Madrid)
  Stop     : minimo de 40 sesiones - 0,5 ATR      <- UN SOLO nivel
  Objetivo : maximo de 40 sesiones                <- UN SOLO nivel, sin salidas parciales
  Tiempo   : cerrar a las 25 sesiones si no toco ninguno

BACKTEST (5 anos, 110 operaciones): 51% aciertos · profit factor 1,65
  · neto medio +4,75 $ por operacion sobre 400 $ con 2 $ de comision
  MUESTRA PEQUENA: resultado orientativo, no garantia.
"""
import warnings, html, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")

CAPITAL      = 400.0
COMISION     = 2.0
DIST_LOW_MAX = 3.0
LOOKBACK     = 40
RR_MIN       = 1.2
EURVOL_MIN   = 3.0
PRECIO_MAX   = 40.0   # con 400 EUR compras >=10 acciones: capital ocioso <5%
HOLD_MAX     = 25

IBEX = {
 'SAN.MC':'Santander','BBVA.MC':'BBVA','ITX.MC':'Inditex','IBE.MC':'Iberdrola','TEF.MC':'Telefonica',
 'CABK.MC':'CaixaBank','AMS.MC':'Amadeus','FER.MC':'Ferrovial','AENA.MC':'Aena','REP.MC':'Repsol',
 'ELE.MC':'Endesa','CLNX.MC':'Cellnex','ACS.MC':'ACS','SAB.MC':'Sabadell','BKT.MC':'Bankinter',
 'RED.MC':'Redeia','NTGY.MC':'Naturgy','MAP.MC':'Mapfre','ANA.MC':'Acciona','GRF.MC':'Grifols',
 'ACX.MC':'Acerinox','ENG.MC':'Enagas','CIE.MC':'CIE Automotive','LOG.MC':'Logista','IDR.MC':'Indra',
 'COL.MC':'Colonial','MRL.MC':'Merlin','ROVI.MC':'Rovi','FDR.MC':'Fluidra','SLR.MC':'Solaria',
 'UNI.MC':'Unicaja','PUIG.MC':'Puig','ANE.MC':'Acciona Energia','SCYR.MC':'Sacyr','MTS.MC':'ArcelorMittal',
}

def flat(d):
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    return d

def cargar(tk):
    """Diario, descartando la sesion en curso y completando si el feed va con retraso."""
    d = yf.download(tk, period='2y', interval='1d', progress=False, auto_adjust=True)
    if d.empty: return None
    d = flat(d).dropna(subset=['Close'])

    # --- GUARDA: si la ultima vela es la sesion de HOY y Madrid aun no ha cerrado
    #     (17:30), esa vela esta incompleta y falsea minimos, maximos y ATR. Se descarta.
    try:
        from zoneinfo import ZoneInfo
        ahora = datetime.now(ZoneInfo('Europe/Madrid'))
        if d.index[-1].date() == ahora.date() and ahora.hour < 18:
            d = d.iloc[:-1]
    except Exception:
        pass

    try:
        h = flat(yf.download(tk, period='5d', interval='1h', progress=False, auto_adjust=True))
        if not h.empty:
            ult = d.index[-1].date()
            for dia in sorted(set(h.index.date)):
                if dia <= ult: continue
                s = h[h.index.date == dia]
                # Solo aceptar la sesion si EXISTE la vela de cierre (17:00 Madrid).
                # Con menos, la sesion sigue abierta y los niveles serian falsos.
                if len(s) and max(x.hour for x in s.index) >= 17:
                    d.loc[pd.Timestamp(dia)] = {'Open': float(s['Open'].iloc[0]),
                        'High': float(s['High'].max()), 'Low': float(s['Low'].min()),
                        'Close': float(s['Close'].iloc[-1]), 'Volume': float(s['Volume'].sum())}
            d = d.sort_index()
    except Exception: pass
    return d if len(d) >= 210 else None

def fundamentales(tk):
    try: i = yf.Ticker(tk).info
    except Exception: return {}
    return dict(g=i.get('revenueGrowth'), m=i.get('profitMargins'), eg=i.get('earningsGrowth'),
                de=i.get('debtToEquity'), pe=i.get('trailingPE'), roe=i.get('returnOnEquity'),
                dy=i.get('dividendYield'), tgt=i.get('targetMeanPrice'),
                rec=i.get('recommendationKey'), sector=i.get('sector', ''))

def noticias(tk, n=3):
    out = []
    try:
        for it in (yf.Ticker(tk).news or [])[:n*2]:
            c = it.get('content', it)
            t = c.get('title',''); f = str(c.get('pubDate', c.get('providerPublishTime','')))[:10]
            p = c.get('provider',{}); p = p.get('displayName','') if isinstance(p,dict) else str(p)
            u = c.get('canonicalUrl',{}); u = u.get('url','') if isinstance(u,dict) else c.get('link','')
            if t and f: out.append(dict(t=t, d=f, p=p, u=u))
            if len(out) >= n: break
    except Exception: pass
    return out

def analizar():
    res, obs, fecha = [], [], None
    for tk, nm in IBEX.items():
        d = cargar(tk)
        if d is None: continue
        c,h,l,v = d['Close'],d['High'],d['Low'],d['Volume']
        fecha = str(d.index[-1].date())
        px = float(c.iloc[-1])
        ma200 = float(c.rolling(200).mean().iloc[-1]); ma50 = float(c.rolling(50).mean().iloc[-1])
        ma20  = float(c.rolling(20).mean().iloc[-1])
        lo = float(l.tail(LOOKBACK).min()); hi = float(h.tail(LOOKBACK).max())
        tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        eurvol = px*float(v.tail(20).mean())/1e6
        dist = (px/lo-1)*100
        confirma = px > float(h.iloc[-2])
        stop = lo - 0.5*atr; obj = hi
        rk = px - stop; rr = (obj-px)/rk if rk > 0 else 0
        base = dict(tk=tk.replace('.MC',''), nm=nm, px=px, dist=dist, lo=lo, hi=hi,
                    stop=stop, obj=obj, rr=rr, atr=atr, eurvol=eurvol, confirma=confirma,
                    vs200=(px/ma200-1)*100, vs50=(px/ma50-1)*100, ma20=ma20,
                    risk_pct=rk/px*100, up_pct=(obj/px-1)*100,
                    adr=float(((h-l)/c).tail(20).mean()*100))
        # compuertas
        if px > PRECIO_MAX: base['estado'], base['motivo'] = 'FUERA', f'{px:.0f} EUR/accion: dejaria capital ocioso'
        elif px <= ma200: base['estado'], base['motivo'] = 'FUERA', f'Bajo MA200 ({(px/ma200-1)*100:+.1f}%): tendencia rota'
        elif eurvol < EURVOL_MIN: base['estado'], base['motivo'] = 'FUERA', f'Liquidez {eurvol:.1f} M EUR/dia'
        elif dist > DIST_LOW_MAX: base['estado'], base['motivo'] = 'FUERA', f'A {dist:.1f}% del minimo: sin descuento'
        elif rk <= 0 or rr < RR_MIN: base['estado'], base['motivo'] = 'FUERA', f'R:R {rr:.2f}: recorrido insuficiente'
        elif not confirma: base['estado'], base['motivo'] = 'VIGILAR', 'En zona, falta vela de confirmacion'
        else: base['estado'], base['motivo'] = 'ENTRAR', 'Todas las compuertas superadas'
        (res if base['estado'] in ('ENTRAR','VIGILAR') else obs).append(base)
    res.sort(key=lambda r: (r['estado'] != 'ENTRAR', -r['rr']))
    obs.sort(key=lambda r: r['dist'])
    return res, obs, fecha

def esc(s): return html.escape(str(s), quote=True)

def fmt_fund(f):
    if not f: return '<span class="dim">Sin datos fundamentales — verificar manualmente</span>'
    p = []
    def add(lab, val, suf='', mult=1, good=None):
        if val is None: return
        x = val*mult
        cls = '' if good is None else ('pos' if good(x) else 'neg')
        p.append(f'<span class="fi"><b class="{cls}">{x:,.1f}{suf}</b><i>{lab}</i></span>')
    add('Ingresos', f.get('g'), '%', 100, lambda x: x > 0)
    add('Margen', f.get('m'), '%', 100, lambda x: x > 0)
    add('Beneficio', f.get('eg'), '%', 100, lambda x: x > 0)
    add('ROE', f.get('roe'), '%', 100, lambda x: x > 0)
    add('PER', f.get('pe'))
    add('Deuda/Pat.', f.get('de'), '', 1, lambda x: x < 150)
    add('Dividendo', f.get('dy'), '%', 1)
    return ''.join(p) or '<span class="dim">Sin datos</span>'

def build(res, obs, fecha, repo=''):
    g = datetime.now(timezone.utc)
    entrar = [r for r in res if r['estado'] == 'ENTRAR']
    vigilar = [r for r in res if r['estado'] == 'VIGILAR']
    actions = f'https://github.com/{repo}/actions' if repo else ''

    def ficha(r, principal):
        acc = CAPITAL/r['px']
        sl_eur = acc*(r['px']-r['stop']); tp_eur = acc*(r['obj']-r['px'])
        f = r.get('f', {}); nw = r.get('nw', [])
        nh = ''.join(f'<a class="nw" href="{esc(n["u"])}" target="_blank" rel="noopener">'
                     f'<span class="nd">{esc(n["d"])} · {esc(n["p"])}</span>{esc(n["t"])}</a>' for n in nw) \
             or '<div class="nw dim">Sin titulares recientes. Busca el valor antes de operar.</div>'
        tag = 'ENTRAR MANANA' if principal else 'FALTA CONFIRMAR'
        cls = 'card go' if principal else 'card wait'
        accion = (f'Orden de <b>MERCADO</b> manana en la apertura (9:00). '
                  if principal else
                  f'<b>NO entrar aun.</b> Necesita cerrar por encima de <b>{r["hi_ayer"]:.3f} EUR</b> '
                  f'(maximo de ayer). Si eso pasa, entras la manana siguiente.')
        techo = r['px'] + 0.5*r['atr']
        antichase = (f'<div class="row"><span class="lb">TECHO DE ENTRADA</span><span class="tx">'
                     f'No compres por encima de <b>{techo:.3f} EUR</b>. Si al mirar el movil el precio ya '
                     f'esta mas arriba, la operacion se fue: el stop queda demasiado lejos y el R:R deja de '
                     f'compensar. <b>Se deja pasar, no se persigue.</b></span></div>') if principal else ''
        return f"""
<div class="{cls}">
  <div class="ch">
    <div class="tk">{r['tk']}<em>{esc(r['nm'])} · {r['px']:.3f} EUR</em></div>
    <div class="bg">{tag}</div>
  </div>
  <div class="plan">
    <div class="pl"><span class="k">COMPRAR</span><span class="v">{acc:.2f} acciones</span><span class="s">= {CAPITAL:.0f} EUR</span></div>
    <div class="pl st"><span class="k">STOP LOSS</span><span class="v">{r['stop']:.3f}</span><span class="s">-{r['risk_pct']:.1f}% · {sl_eur:.2f} EUR</span></div>
    <div class="pl tp"><span class="k">TAKE PROFIT</span><span class="v">{r['obj']:.3f}</span><span class="s">+{r['up_pct']:.1f}% · {tp_eur:.2f} EUR</span></div>
    <div class="pl"><span class="k">R:R</span><span class="v">{r['rr']:.2f}</span><span class="s">{HOLD_MAX} sesiones max</span></div>
  </div>
  <div class="body">
    <div class="row"><span class="lb">EJECUCION</span><span class="tx">{accion}
      Pon el Stop Loss en <b>{r['stop']:.3f}</b> y el Take Profit en <b>{r['obj']:.3f}</b>.
      Si eToro te los pide en dinero: <b>{sl_eur:.2f} EUR</b> de perdida y <b>{tp_eur:.2f} EUR</b> de ganancia.
      <b>Un solo objetivo, sin cierres parciales.</b> Si a las {HOLD_MAX} sesiones no toco ninguno, cierra y libera el capital.</span></div>
    {antichase}
    <div class="row"><span class="lb">ESTRUCTURA</span><span class="tx">
      A <b>{r['dist']:.1f}%</b> de su minimo de 8 semanas ({r['lo']:.3f}), pero <b>{r['vs200']:+.1f}%</b> sobre la MA200
      y {r['vs50']:+.1f}% sobre la MA50: corrige dentro de tendencia, no se derrumba.
      Objetivo = maximo de 8 semanas ({r['hi']:.3f}). ATR {r['atr']:.3f} ({r['adr']:.1f}% diario) ·
      liquidez {r['eurvol']:.0f} M EUR/dia.</span></div>
    <div class="row"><span class="lb">FUNDAMENTAL</span><span class="tx"><span class="fund">{fmt_fund(f)}</span>
      {'<br><span class="dim">Sector: '+esc(f.get('sector',''))+'</span>' if f.get('sector') else ''}</span></div>
    <div class="row"><span class="lb">TITULARES</span><span class="tx">{nh}</span></div>
  </div>
</div>"""

    cards = ''.join(ficha(r, r['estado'] == 'ENTRAR') for r in res[:4])
    if not res:
        cards = ('<div class="empty"><div class="big">NO OPERAR</div>'
                 'Ningun valor esta en zona de rebote con tendencia intacta. '
                 'El sistema genera ~2 senales al mes: la mayoria de los dias, la respuesta correcta es esperar.</div>')

    filas = ''.join(f'<tr><td class="tk2">{r["tk"]}</td><td>{r["px"]:.2f}</td>'
                    f'<td>{r["dist"]:.1f}%</td><td class="{"pos" if r["vs200"]>0 else "neg"}">{r["vs200"]:+.1f}%</td>'
                    f'<td class="dim">{esc(r["motivo"])}</td></tr>' for r in obs[:12])

    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Panel Rebote IBEX</title>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:#05070B;--c1:#0C1219;--c2:#131B26;--ln:#212D3C;--gr:#00E88F;--am:#FFC400;
--rd:#FF3A5E;--bl:#3DA5FF;--w:#fff;--g1:#C6D0DC;--g2:#8493A5}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--w);font-family:Archivo,system-ui,sans-serif;font-size:15px;line-height:1.5;padding-bottom:44px}}
.wrap{{max-width:800px;margin:0 auto;padding:0 14px}}
header{{padding:20px 0 8px}}h1{{font-size:25px;font-weight:900;letter-spacing:-.02em}}h1 span{{color:var(--gr)}}
.sub{{color:var(--g2);font-size:11.5px;font-family:'JetBrains Mono',monospace;margin-top:4px}}
.bar{{display:flex;align-items:center;gap:10px;background:var(--c1);border:1px solid var(--ln);padding:11px 13px;margin:12px 0 20px}}
.dot{{width:11px;height:11px;border-radius:50%;flex:0 0 11px;background:var(--g2)}}
.bar.ok .dot{{background:var(--gr);box-shadow:0 0 8px var(--gr)}}
.bar.old .dot{{background:var(--rd)}}
.bt{{flex:1;min-width:0}}.b1{{font-size:12.5px;font-weight:700}}
.b2{{font-size:10.5px;color:var(--g2);font-family:'JetBrains Mono',monospace}}
.btn{{font-size:10.5px;font-weight:700;padding:8px 11px;border:1px solid var(--ln);background:var(--c2);
color:var(--g1);cursor:pointer;font-family:Archivo;text-decoration:none;white-space:nowrap}}
.btn:hover{{border-color:var(--bl);color:#fff}}
.rec{{background:var(--c1);border-left:3px solid var(--bl);padding:10px 13px;margin-bottom:4px;
font-size:12.5px;color:var(--g1);line-height:1.55}}
.rec b{{color:#fff}}
.sh{{display:flex;align-items:center;gap:8px;margin:24px 0 10px}}
.sh i{{width:10px;height:10px;border-radius:50%;font-style:normal}}
.sh h2{{font-size:16px;font-weight:900}}
.card{{background:var(--c1);border:1px solid var(--ln);margin-bottom:13px}}
.card.go{{border:2px solid var(--gr)}}.card.wait{{border-left:3px solid var(--am)}}
.ch{{display:flex;justify-content:space-between;align-items:center;padding:12px 14px;background:var(--c2);
border-bottom:1px solid var(--ln);flex-wrap:wrap;gap:6px}}
.tk{{font-size:21px;font-weight:900}}.tk em{{font-style:normal;font-size:11.5px;color:var(--g2);font-weight:400;margin-left:8px}}
.card.go .bg{{font-size:9.5px;font-weight:700;letter-spacing:.07em;background:var(--gr);color:#000;padding:4px 9px}}
.card.wait .bg{{font-size:9.5px;font-weight:700;letter-spacing:.07em;color:var(--am);border:1px solid var(--am);padding:4px 9px}}
.plan{{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--ln)}}
.pl{{padding:11px 6px;text-align:center;border-right:1px solid var(--ln);display:flex;flex-direction:column;gap:2px}}
.pl:last-child{{border-right:0}}
.pl .k{{font-size:8.5px;font-weight:700;color:var(--g2);letter-spacing:.05em}}
.pl .v{{font-family:'JetBrains Mono',monospace;font-size:17px;font-weight:700}}
.pl .s{{font-size:9.5px;color:var(--g2);font-family:'JetBrains Mono',monospace}}
.pl.st .v{{color:var(--rd)}}.pl.tp .v{{color:var(--gr)}}
@media(max-width:560px){{.plan{{grid-template-columns:1fr 1fr}}.pl{{border-bottom:1px solid var(--ln)}}}}
.body{{padding:13px 14px}}
.row{{display:block;margin-bottom:11px;font-size:12.5px}}.row:last-child{{margin-bottom:0}}
.lb{{display:block;font-size:9px;font-weight:700;letter-spacing:.07em;color:var(--bl);margin-bottom:3px}}
.tx{{color:var(--g1);line-height:1.55}}.tx b{{color:#fff}}
.fund{{display:flex;flex-wrap:wrap;gap:6px;margin-top:2px}}
.fi{{background:var(--c2);border:1px solid var(--ln);padding:5px 8px;display:flex;flex-direction:column;min-width:64px}}
.fi b{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700}}
.fi i{{font-style:normal;font-size:8.5px;color:var(--g2);letter-spacing:.04em}}
.pos{{color:var(--gr)}}.neg{{color:var(--rd)}}.dim{{color:var(--g2)}}
.nw{{display:block;color:var(--g1);text-decoration:none;padding:6px 0;border-bottom:1px solid rgba(33,45,60,.6);font-size:11.5px;line-height:1.45}}
.nw:last-child{{border-bottom:0}}.nw:hover{{color:#fff}}
.nd{{display:block;font-size:9px;color:var(--g2);font-family:'JetBrains Mono',monospace}}
.empty{{background:var(--c1);border:2px solid var(--rd);padding:26px 16px;text-align:center;color:var(--g1);font-size:13px}}
.empty .big{{font-size:32px;font-weight:900;color:var(--rd);margin-bottom:8px;letter-spacing:-.02em}}
table{{width:100%;border-collapse:collapse;font-size:11.5px;font-family:'JetBrains Mono',monospace}}
th{{text-align:left;padding:7px 8px;color:var(--g2);font-size:8.5px;letter-spacing:.05em;border-bottom:1px solid var(--ln)}}
td{{padding:7px 8px;border-bottom:1px solid rgba(33,45,60,.5)}}.tk2{{font-weight:700;color:var(--w);font-family:Archivo}}
.bx{{background:var(--c1);border:1px solid var(--ln);padding:14px}}
.bx h3{{font-size:12px;font-weight:900;letter-spacing:.05em;margin-bottom:9px;color:var(--bl)}}
.bx p{{color:var(--g1);font-size:12.5px;line-height:1.6;margin-bottom:8px}}.bx p:last-child{{margin-bottom:0}}
.bx b{{color:#fff}}
footer{{margin-top:22px;padding-top:13px;border-top:1px solid var(--ln);color:var(--g2);font-size:10.5px;line-height:1.65}}
footer b{{color:var(--g1)}}
</style></head><body><div class="wrap">
<header><h1>Panel <span>Rebote</span> IBEX</h1>
<div class="sub">Reversion a minimos · una posicion · {CAPITAL:.0f} EUR · eToro</div></header>

<div class="bar" id="bar"><div class="dot"></div><div class="bt">
<div class="b1">Cierre {fecha}</div>
<div class="b2" id="b2">Generado {g.strftime('%d/%m/%Y %H:%M')} UTC</div></div>
<a class="btn" href="?t=0" onclick="this.href='?t='+Date.now()">RECARGAR</a>
{f'<a class="btn" href="{actions}" target="_blank" rel="noopener">ACTUALIZAR</a>' if actions else ''}
</div>

<div class="rec">
  <b>Mirar solo antes de las 9:00 (Madrid) o de noche con el mercado cerrado.</b>
  Durante la sesion el analisis no es valido: trabaja con velas cerradas.
</div>

<div class="sh"><i style="background:var(--gr)"></i><h2 style="color:var(--gr)">Plan de hoy</h2></div>
{cards}

<div class="sh"><i style="background:var(--g2)"></i><h2 style="color:var(--g2)">Descartados</h2></div>
<div class="bx" style="padding:0">
<table><tr><th>VALOR</th><th>PRECIO</th><th>DIST. MIN</th><th>vs MA200</th><th>MOTIVO</th></tr>{filas}</table></div>

<div class="sh"><i style="background:var(--bl)"></i><h2 style="color:var(--bl)">El sistema</h2></div>
<div class="bx">
<h3>CRITERIOS VALIDADOS CON BACKTEST</h3>
<p>Cinco anos, 35 valores del IBEX, 110 operaciones: <b>51% de aciertos, profit factor 1,65</b>,
neto medio <b>+4,75 EUR</b> por operacion sobre 400 EUR con 2 EUR de comision.</p>
<p><b>La compuerta que decide todo</b> es la vela de confirmacion. Sin exigir que el precio cierre por encima
del maximo del dia anterior, el mismo sistema <b>pierde dinero</b>. Nunca se compra mientras cae.</p>
<p><b>Un solo objetivo, sin salidas parciales.</b> El backtest lo demuestra: cerrar antes, en la media de 20,
convierte el sistema en perdedor (profit factor 0,76) porque las ganancias quedan mas pequenas que las perdidas
aunque aciertes mas veces.</p>
<p><b>Cuando mirar el panel.</b> Es un sistema de cierre: analiza solo sesiones cerradas, asi que
pulsar ACTUALIZAR a las 11:00 o a las 16:00 da <b>el mismo resultado</b>. No es un fallo, es el diseno:
una vela a medio hacer daria confirmaciones que luego se evaporan. Revisalo por la tarde tras el cierre
o por la manana antes de abrir, y ejecuta en la apertura. Medido en el backtest: entrar en la apertura
rinde <b>+4,93 EUR</b> por operacion (PF 1,68); entrar mas tarde ese mismo dia, <b>+4,36 EUR</b> (PF 1,62).
Pierdes algo de ventaja pero sigue funcionando: <b>tienes flexibilidad de horario</b>.</p>
<p><b>Frecuencia esperada:</b> ~2 senales al mes. La mayoria de los dias no hay nada que hacer, y eso es
el funcionamiento correcto, no un fallo.</p>
</div>

<footer>
<p><b>Limites que debes conocer.</b> 110 operaciones es una muestra pequena y el profit factor de 1,65 puede
contener ruido. El backtest asume que ejecutas todos los stops sin excepcion y que entras en la apertura
siguiente. No incluye horquilla ni deslizamiento.</p>
<p><b>El fundamental no descarta automaticamente:</b> se muestra para tu juicio. Si los ingresos caen o la deuda
es alta, no operes aunque el tecnico sea perfecto. Lee los titulares antes de entrar.</p>
<p><b>Datos:</b> Yahoo Finance, OHLCV diario, tickers .MC de la Bolsa de Madrid. Panel de analisis,
no recomendacion de inversion.</p>
</footer></div>
<script>
(function(){{
 var gen=new Date("{g.isoformat()}"), h=(new Date()-gen)/36e5, b=document.getElementById('bar'),
 s=document.getElementById('b2');
 if(h>96){{b.className='bar old'; s.textContent+=' · Sin actualizar hace mas de 4 dias: revisa Actions.';}}
 else {{b.className='bar ok'; s.textContent+=' · Datos al dia.';}}
}})();
</script></body></html>"""

def main():
    print('Analizando IBEX 35...')
    res, obs, fecha = analizar()
    for r in res[:4]:
        r['f'] = fundamentales(r['tk'] + '.MC')
        r['nw'] = noticias(r['tk'] + '.MC')
        d = cargar(r['tk'] + '.MC')
        r['hi_ayer'] = float(d['High'].iloc[-1])
    print(f'Cierre {fecha} | ENTRAR: {sum(1 for r in res if r["estado"]=="ENTRAR")} | '
          f'VIGILAR: {sum(1 for r in res if r["estado"]=="VIGILAR")}')
    for r in res[:4]:
        print(f"  [{r['estado']:7s}] {r['tk']:5s} {r['px']:8.3f} stop {r['stop']:8.3f} "
              f"obj {r['obj']:8.3f} R:R {r['rr']:.2f} dist {r['dist']:.1f}%")
    Path('index.html').write_text(
        build(res, obs, fecha, os.environ.get('GITHUB_REPOSITORY','')), encoding='utf-8')
    print('Panel: index.html')

if __name__ == '__main__':
    main()
