from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd, streamlit as st
import streamlit.components.v1 as components

# ---------- small utils ----------
_root = lambda: next((p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
                      if (p/"data/common_data").exists() or (p/"app.py").exists()), Path(__file__).resolve().parents[2])
_find_mc = lambda d: (sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower()==".xlsx"
                              and any(n in p.stem.lower() for n in ("mesures completes","mesures complètes","mesures complete","mesures compl"))],
                             key=lambda x:x.stat().st_mtime, reverse=True)[:1] or [None])[0] if d.exists() else None
_sheet0 = lambda xlsx: (lambda x: x.sheet_names[0] if x.sheet_names else (_ for _ in ()).throw(ValueError("Classeur Excel sans onglets.")))(pd.ExcelFile(xlsx, engine="openpyxl"))
_clean = lambda v: None if v is None or (isinstance(v,float) and np.isnan(v)) or not str(v).strip() else str(v).strip()
_fmt_date = lambda dt: "—" if dt is None or pd.isna(dt) else pd.Timestamp(dt).strftime("%d/%m/%Y")
_fmt_mm = lambda v: "—" if v is None or (isinstance(v,float) and (np.isnan(v) or np.isinf(v))) else f"{v*1000.0:.1f} mm"
_fmt_hyp_mm = lambda dx,dy,dz=None: "—" if not (np.isfinite(dx) and np.isfinite(dy)) or (dz is not None and not np.isfinite(dz)) else _fmt_mm(float(np.sqrt(dx*dx+dy*dy+(0 if dz is None else dz*dz))))
_esc = lambda s: str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;").replace("'","&#39;")

def _rb(vals, q0=1.0, q1=99.0):
    v=vals[np.isfinite(vals)]
    if v.size==0: return 0.0,1.0
    lo,hi=float(np.percentile(v,q0)),float(np.percentile(v,q1))
    if hi-lo<1e-12: lo,hi=float(v.min()),float(v.max()); hi=hi if hi-lo>=1e-12 else lo+1.0
    return lo,hi

def _mapper(points,w,h):
    w,h=float(w),float(h); pad=30.0; iw,ih=max(w-2*pad,1.0),max(h-2*pad,1.0)
    xs,ys=points["x"].to_numpy(float),points["y"].to_numpy(float)
    bx0,bx1=_rb(xs); by0,by1=_rb(ys); sx,sy=iw/max(bx1-bx0,1e-12),ih/max(by1-by0,1e-12); s=min(sx,sy)
    cw,ch=(bx1-bx0)*s,(by1-by0)*s; ox,oy=(w-cw)/2,(h-ch)/2
    clamp=lambda v,a,b: a if v<a else (b if v>b else v)
    def to_px(x,y):
        x,y=clamp(x,bx0,bx1),clamp(y,by0,by1)
        return ox+(x-bx0)*s, oy+(by1-y)*s
    return to_px,float(s)

def _auto_vec_scale(points,deltas,to_px,desired=25.0):
    if points.empty or deltas.empty: return 1.0
    dd=deltas.drop_duplicates("name",keep="last").set_index("name",drop=False)
    L=[]
    for _,r in points.iterrows():
        n=str(r["name"])
        if n not in dd.index: continue
        dx,dy=float(dd.loc[n,"dx"]),float(dd.loc[n,"dy"])
        if not (np.isfinite(dx) and np.isfinite(dy)) or (abs(dx)<1e-15 and abs(dy)<1e-15): continue
        x,y=float(r["x"]),float(r["y"]); px,py=to_px(x,y); p2x,p2y=to_px(x+dx,y+dy)
        l=float(np.hypot(p2x-px,p2y-py))
        if np.isfinite(l) and l>0: L.append(l)
    if not L: return 1.0
    med=float(np.median(L))
    return 1.0 if med<1e-9 else max(min(desired/med,500.0),0.02)

# ---------- excel parsing ----------
@st.cache_data(show_spinner=False)
def _groups(xlsx,mtime):
    sh=_sheet0(xlsx)
    h=pd.read_excel(xlsx,sheet_name=sh,header=None,nrows=1,engine="openpyxl").iloc[0,:]
    starts=[(_clean(h.iloc[j]),j) for j in range(1,len(h)) if _clean(h.iloc[j]) is not None]
    g=[(n,j,j+1,j+2) for n,j in starts if j+2<len(h)]
    return sh,g

@st.cache_data(show_spinner=False)
def _date_range(xlsx,mtime):
    sh=_sheet0(xlsx)
    d=pd.read_excel(xlsx,sheet_name=sh,header=None,skiprows=1,usecols=[0],engine="openpyxl")
    dates=pd.to_datetime(d.iloc[:,0],errors="coerce",dayfirst=True).dropna()
    return (None,None) if dates.empty else (dates.min(),dates.max())

def _win(last,days):
    if last is None or pd.isna(last): return None,None
    last=pd.Timestamp(last); return last-pd.Timedelta(days=days), last

@st.cache_data(show_spinner=False)
def _xy_medians(xlsx,mtime):
    sh,g=_groups(xlsx,mtime)
    if not g: return pd.DataFrame(columns=["name","x","y"])
    use=sorted({0,*[j for _,jx,jy,_ in g for j in (jx,jy)]})
    df=pd.read_excel(xlsx,sheet_name=sh,header=None,skiprows=1,usecols=use,engine="openpyxl").apply(pd.to_numeric,errors="coerce")
    pos={c:i for i,c in enumerate(use)}
    rows=[]
    for n,jx,jy,_ in g:
        x=df.iloc[:,pos[jx]].to_numpy(float,copy=False); y=df.iloc[:,pos[jy]].to_numpy(float,copy=False)
        if np.isfinite(x).sum()==0 or np.isfinite(y).sum()==0: continue
        rows.append({"name":n,"x":float(np.nanmedian(x)),"y":float(np.nanmedian(y))})
    out=pd.DataFrame(rows)
    return out.dropna(subset=["name"]).drop_duplicates("name",keep="last").sort_values("name").reset_index(drop=True) if not out.empty else out

def _deltas_from_df(df,g,use):
    df=df.apply(pd.to_numeric,errors="coerce"); pos={c:i for i,c in enumerate(use)}; rows=[]
    for n,jx,jy,jz in g:
        x=df.iloc[:,pos[jx]].to_numpy(float,copy=False); y=df.iloc[:,pos[jy]].to_numpy(float,copy=False); z=df.iloc[:,pos[jz]].to_numpy(float,copy=False)
        ix,iy,iz=np.where(np.isfinite(x))[0],np.where(np.isfinite(y))[0],np.where(np.isfinite(z))[0]
        if ix.size==0 or iy.size==0 or iz.size==0: continue
        rows.append({"name":n,"dx":float(x[ix[-1]]-x[ix[0]]),"dy":float(y[iy[-1]]-y[iy[0]]),"dz":float(z[iz[-1]]-z[iz[0]])})
    out=pd.DataFrame(rows)
    return out.dropna(subset=["name"]).drop_duplicates("name",keep="last").sort_values("name").reset_index(drop=True) if not out.empty else out

def _read_xyz_df(xlsx,mtime):
    sh,g=_groups(xlsx,mtime)
    if not g: return sh,[],[0],pd.DataFrame(),pd.Series(dtype="datetime64[ns]")
    use=sorted({0,*[j for _,jx,jy,jz in g for j in (jx,jy,jz)]})
    df=pd.read_excel(xlsx,sheet_name=sh,header=None,skiprows=1,usecols=use,engine="openpyxl")
    dates=pd.to_datetime(df.iloc[:,0],errors="coerce",dayfirst=True)
    return sh,g,use,df,dates

@st.cache_data(show_spinner=False)
def _deltas_first_last(xlsx,mtime):
    sh,g,use,df,dates=_read_xyz_df(xlsx,mtime)
    if not g: return pd.DataFrame(columns=["name","dx","dy","dz"])
    if dates.notna().sum():
        order=np.argsort(dates.fillna(pd.Timestamp.min).to_numpy())
        df=df.iloc[order].reset_index(drop=True)
    return _deltas_from_df(df,g,use)

@st.cache_data(show_spinner=False)
def _deltas_last_days(xlsx,mtime,days):
    sh,g,use,df,dates=_read_xyz_df(xlsx,mtime)
    if not g: return pd.DataFrame(columns=["name","dx","dy","dz"])
    if dates.notna().sum()==0: return pd.DataFrame(columns=["name","dx","dy","dz"])
    last=dates.max(); start=last-pd.Timedelta(days=days)
    m=(dates>=start)&(dates<=last); dfw,datesw=df.loc[m].copy(),dates.loc[m].copy()
    if dfw.empty: return pd.DataFrame(columns=["name","dx","dy","dz"])
    order=np.argsort(datesw.fillna(pd.Timestamp.min).to_numpy()); dfw=dfw.iloc[order].reset_index(drop=True)
    return _deltas_from_df(dfw,g,use)

@st.cache_data(show_spinner=False)
def _timeseries_xyz(xlsx,mtime):
    sh,g,use,df,dates=_read_xyz_df(xlsx,mtime)
    if not g: return pd.Series(dtype="datetime64[ns]"),[],[0],pd.DataFrame()
    m=dates.notna(); df,dates=df.loc[m].reset_index(drop=True),dates.loc[m].reset_index(drop=True)
    order=np.argsort(dates.to_numpy()); df,dates=df.iloc[order].reset_index(drop=True),dates.iloc[order].reset_index(drop=True)
    return dates,g,use,df.apply(pd.to_numeric,errors="coerce")

# ---------- HTML (minified-ish; 1 big string to save python lines) ----------
# zoom mini = 1.0 (stop dézoom une fois toutes les cibles visibles)
_TPL = r'''<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;padding:0;background:#fff}#wrap{width:__W__px;height:__H__px;overflow:hidden;display:flex;flex-direction:column}#viewport{position:relative;width:__W__px;height:__VIEW_H__px;background:#fff;flex:1 1 auto}svg{display:block;background:#fff;touch-action:none}.target{fill:rgba(255,0,0,.88);stroke:rgba(0,0,0,.92);stroke-width:1.2px;cursor:default}.target:hover{fill:rgba(255,0,0,.98)}.vec{stroke:rgba(255,145,0,.55);stroke-width:2.2px;marker-end:url(#arrow);stroke-linecap:round}.vec:hover{stroke:rgba(255,145,0,.88)}.tooltip{position:absolute;pointer-events:none;background:rgba(20,20,20,.92);color:#fff;font:12px/1.2 system-ui,-apple-system,Segoe UI,Roboto,Arial;padding:8px 10px;border-radius:10px;white-space:nowrap;transform:translate(10px,10px);opacity:0;transition:opacity 60ms linear;z-index:80}.tooltip .title{font-weight:700;margin-bottom:4px}.tooltip .row{opacity:.95}#controls{position:absolute;top:12px;right:12px;width:170px;padding:10px;border-radius:14px;background:rgba(255,255,255,.86);box-shadow:0 6px 18px rgba(0,0,0,.08);backdrop-filter:blur(6px);display:flex;flex-direction:column;gap:10px;user-select:none;z-index:70}#controls.hidden{display:none}.rng{width:100%;margin:0;-webkit-appearance:none;appearance:none;height:4px;border-radius:999px;outline:none}.rng::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;height:14px;border-radius:50%;border:2px solid rgba(255,255,255,.96);box-shadow:0 4px 10px rgba(0,0,0,.15);cursor:pointer}.rng::-moz-range-thumb{width:14px;height:14px;border-radius:50%;border:2px solid rgba(255,255,255,.96);box-shadow:0 4px 10px rgba(0,0,0,.15)}.rng-orange{background:rgba(255,145,0,.35)}.rng-orange::-webkit-slider-thumb{background:rgba(255,145,0,.95)}.rng-orange::-moz-range-thumb{background:rgba(255,145,0,.95)}.scalebox{position:absolute;bottom:12px;padding:8px 10px 7px;border-radius:12px;background:rgba(255,255,255,.86);box-shadow:0 6px 18px rgba(0,0,0,.08);backdrop-filter:blur(6px);user-select:none;z-index:75;pointer-events:none}#scaleXY{left:12px}#scaleV{right:12px}.scalebox.hidden{display:none}.bar{height:8px;border-radius:999px;overflow:hidden;display:inline-block}.bar-fill{height:100%;width:100%;background:rgba(0,0,0,.85)}.bar.orange .bar-fill{background:rgba(255,145,0,.95)}.labelsbar{position:relative;margin-top:6px;height:14px;font:12px/1.1 system-ui,-apple-system,Segoe UI,Roboto,Arial;color:rgba(0,0,0,.78)}#scaleV .labelsbar{color:rgba(255,145,0,.95)}.labelsbar .mid{position:absolute;left:50%;transform:translateX(-50%);white-space:nowrap}.labelsbar .end{position:absolute;right:0;white-space:nowrap}.customTop{margin:12px 12px 0;padding:10px 12px;border-radius:14px;background:rgba(255,255,255,.86);box-shadow:0 6px 18px rgba(0,0,0,.08);backdrop-filter:blur(6px);user-select:none;flex:0 0 auto;z-index:60}.customTitle{display:flex;align-items:baseline;gap:10px;margin-bottom:8px;font:12px/1.1 system-ui,-apple-system,Segoe UI,Roboto,Arial;color:rgba(0,0,0,.82)}.customTitle .t{font-weight:750}.dualWrap{position:relative;height:28px;margin-top:2px}.trackBase{position:absolute;left:0;right:0;top:50%;transform:translateY(-50%);height:6px;border-radius:999px;background:rgba(122,14,14,.18)}.trackSel{position:absolute;top:50%;transform:translateY(-50%);height:6px;border-radius:999px;background:rgba(122,14,14,.92);left:20%;width:60%}.range{position:absolute;left:0;right:0;top:0;width:100%;height:28px;margin:0;background:transparent;pointer-events:none;-webkit-appearance:none;appearance:none;outline:none}.range::-webkit-slider-runnable-track{height:6px;background:transparent}.range::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;border:2px solid rgba(255,255,255,.96);box-shadow:0 4px 10px rgba(0,0,0,.18);background:rgba(122,14,14,.95);cursor:pointer;pointer-events:auto;margin-top:-5px}.range::-moz-range-track{height:6px;background:transparent}.range::-moz-range-thumb{width:16px;height:16px;border-radius:50%;border:2px solid rgba(255,255,255,.96);box-shadow:0 4px 10px rgba(0,0,0,.18);background:rgba(122,14,14,.95);cursor:pointer;pointer-events:auto}</style></head><body><div id="wrap">__CUSTOM_TOP__<div id="viewport" data-base-px-per-m="__BASE__" data-vec-base-scale="__VBASE__" data-tooltip-mode="__TM__"><div id="tip" class="tooltip"></div>__CONTROLS__<div id="scaleXY" class="scalebox __SXY__"><div class="bar" id="barXY"><div class="bar-fill"></div></div><div class="labelsbar" id="labelsXY"><span class="mid" id="labXYa">—</span><span class="end" id="labXYb">—</span></div></div><div id="scaleV" class="scalebox __SV__"><div class="bar orange" id="barV"><div class="bar-fill"></div></div><div class="labelsbar" id="labelsV"><span class="mid" id="labVa">—</span><span class="end" id="labVb">—</span></div></div><svg id="svg" width="__W__" height="__VIEW_H__" viewBox="0 0 __W__ __VIEW_H__" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="rgba(255,145,0,.70)"></path></marker></defs><g id="scene">__SVG__</g></svg></div></div><script>const CUSTOM=__CUSTOM__;const vp=document.getElementById('viewport'),tip=document.getElementById('tip'),svg=document.getElementById('svg'),scene=document.getElementById('scene');const tm=(vp.getAttribute('data-tooltip-mode')||'name').toLowerCase(),vs=document.getElementById('vecScale');const base=parseFloat(vp.getAttribute('data-base-px-per-m'))||1,vbase=parseFloat(vp.getAttribute('data-vec-base-scale'))||1;const barXY=document.getElementById('barXY'),barV=document.getElementById('barV'),labelsXY=document.getElementById('labelsXY'),labelsV=document.getElementById('labelsV'),labXYa=document.getElementById('labXYa'),labXYb=document.getElementById('labXYb'),labVa=document.getElementById('labVa'),labVb=document.getElementById('labVb');const esc=s=>String(s).split('&').join('&amp;').split('<').join('&lt;').split('>').join('&gt;').split('"').join('&quot;').split("'").join('&#39;');const hide=()=>tip.style.opacity='0';const showName=(n,x,y)=>{tip.innerHTML='<div class="title">'+esc(n)+'</div>';tip.style.left=x+'px';tip.style.top=y+'px';tip.style.opacity='1'};const showD=(n,dx,dy,dz,dp,ds,x,y)=>{tip.innerHTML='<div class="title">'+esc(n)+'</div><div class="row">ΔX : '+esc(dx)+'</div><div class="row">ΔY : '+esc(dy)+'</div><div class="row">ΔZ : '+esc(dz)+'</div><div class="row">Δ plan : '+esc(dp)+'</div><div class="row">Δ spatial : '+esc(ds)+'</div>';tip.style.left=x+'px';tip.style.top=y+'px';tip.style.opacity='1'};svg.addEventListener('mousemove',e=>{const t=e.target;if(t&&t.classList&&(t.classList.contains('target')||t.classList.contains('vec'))){const n=t.getAttribute('data-name')||'';if(!n)return hide();const r=vp.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(tm==='deltas')showD(n,t.getAttribute('data-dx')||'—',t.getAttribute('data-dy')||'—',t.getAttribute('data-dz')||'—',t.getAttribute('data-dp')||'—',t.getAttribute('data-ds')||'—',x,y);else showName(n,x,y);return}hide()});svg.addEventListener('mouseleave',hide);const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));const MIN_ZOOM=1;const vMult=()=>{if(!vs)return 1;const v=parseFloat(vs.value);return Number.isFinite(v)?v:1};const applyV=m=>{if(!vs)return;svg.querySelectorAll('.vec').forEach(l=>{const x1=parseFloat(l.getAttribute('data-x1')),y1=parseFloat(l.getAttribute('data-y1')),vx=parseFloat(l.getAttribute('data-vx')),vy=parseFloat(l.getAttribute('data-vy'));if(!Number.isFinite(x1)||!Number.isFinite(y1)||!Number.isFinite(vx)||!Number.isFinite(vy))return;l.setAttribute('x2',(x1+vx*m).toFixed(3));l.setAttribute('y2',(y1+vy*m).toFixed(3))})};let panX=0,panY=0,zoom=1;const applyT=()=>scene.setAttribute('transform','translate('+panX.toFixed(3)+','+panY.toFixed(3)+') scale('+zoom.toFixed(6)+')');const nice=x=>{if(!Number.isFinite(x)||x<=0)return 1;const e=Math.floor(Math.log10(x)),k=Math.pow(10,e),u=x/k;let n=1; if(u<=1)n=1; else if(u<=2)n=2; else if(u<=5)n=5; else n=10;const o=n*k;return (Number.isFinite(o)&&o>0)?o:1};const fmt=(v,u)=>{if(!Number.isFinite(v))return'—';if(v>=10)return Math.round(v)+' '+u;if(v>=1)return (Math.round(v*10)/10).toFixed(1).replace('.0','')+' '+u;let s=(Math.round(v*100)/100).toFixed(2);while(s.includes('.')&&(s.endsWith('0')||s.endsWith('.'))){s=s.slice(0,-1);if(s.endsWith('.')){s=s.slice(0,-1);break}}return s+' '+u};const upd=()=>{if(!barXY||!labelsXY||!labXYa||!labXYb)return;const pxPerM=base*zoom, target=170,min=110,max=230;let L=nice(target/Math.max(pxPerM,1e-12)),w=L*pxPerM;for(let i=0;i<24&&Number.isFinite(w)&&w<min;i++){L=nice(L*2);w=L*pxPerM}for(let i=0;i<24&&Number.isFinite(w)&&w>max&&L>0;i++){L=nice(L/2);w=L*pxPerM}if(!Number.isFinite(w)||w<=0){L=1;w=clamp(target,min,max)}barXY.style.width=w.toFixed(1)+'px';labelsXY.style.width=w.toFixed(1)+'px';labXYa.textContent=fmt(L/2,'m');labXYb.textContent=fmt(L,'m');if(barV&&labelsV&&labVa&&labVb){let pxPerMm=(pxPerM*vbase*vMult())/1000;if(!Number.isFinite(pxPerMm)||pxPerMm<=0)pxPerMm=1e-12;let Lm=nice(target/pxPerMm),wm=Lm*pxPerMm;for(let i=0;i<24&&Number.isFinite(wm)&&wm<min;i++){Lm=nice(Lm*2);wm=Lm*pxPerMm}for(let i=0;i<24&&Number.isFinite(wm)&&wm>max&&Lm>0;i++){Lm=nice(Lm/2);wm=Lm*pxPerMm}if(!Number.isFinite(wm)||wm<=0){Lm=1;wm=clamp(target,min,max)}barV.style.width=wm.toFixed(1)+'px';labelsV.style.width=wm.toFixed(1)+'px';labVa.textContent=fmt(Lm/2,'mm');labVb.textContent=fmt(Lm,'mm')}};if(vs){applyV(vMult());vs.addEventListener('input',()=>{applyV(vMult());upd()})}svg.addEventListener('wheel',e=>{e.preventDefault();const r=svg.getBoundingClientRect(),cx=e.clientX-r.left,cy=e.clientY-r.top,z0=zoom,f=Math.exp(-e.deltaY*0.0012),z1=clamp(zoom*f,MIN_ZOOM,25),sx=(cx-panX)/z0,sy=(cy-panY)/z0;zoom=z1;panX=cx-sx*z1;panY=cy-sy*z1;applyT();upd();CUSTOM&&CUSTOM.enabled&&req()}, {passive:false});let pan=false,sx0=0,sy0=0,px0=0,py0=0;svg.addEventListener('pointerdown',e=>{const p=e.composedPath?e.composedPath():[];if(p.some(el=>el&&el.id==='controls')||p.some(el=>el&&el.id==='customTop'))return;pan=true;sx0=e.clientX;sy0=e.clientY;px0=panX;py0=panY;svg.setPointerCapture(e.pointerId)});svg.addEventListener('pointermove',e=>{if(!pan)return;panX=px0+(e.clientX-sx0);panY=py0+(e.clientY-sy0);applyT()});svg.addEventListener('pointerup',e=>{pan=false;try{svg.releasePointerCapture(e.pointerId)}catch(_){} });svg.addEventListener('pointercancel',()=>pan=false);svg.addEventListener('dblclick',()=>{panX=0;panY=0;zoom=1;applyT();upd();CUSTOM&&CUSTOM.enabled&&req()});let raf=0;const mm=m=>Number.isFinite(m)?(m*1000).toFixed(1)+' mm':'—',h2=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)?Math.sqrt(a*a+b*b):NaN,h3=(a,b,c)=>Number.isFinite(a)&&Number.isFinite(b)&&Number.isFinite(c)?Math.sqrt(a*a+b*b+c*c):NaN,ds=ms=>{const d=new Date(ms);return String(d.getDate()).padStart(2,'0')+'/'+String(d.getMonth()+1).padStart(2,'0')+'/'+d.getFullYear()};function req(){if(!CUSTOM||!CUSTOM.enabled)return;if(raf)cancelAnimationFrame(raf);raf=requestAnimationFrame(()=>{raf=0;recomp()})}function recomp(){if(!CUSTOM||!CUSTOM.enabled)return;const a=document.getElementById('rangeA'),b=document.getElementById('rangeB');if(!a||!b)return;let ia=parseInt(a.value,10),ib=parseInt(b.value,10);if(!Number.isFinite(ia))ia=0;if(!Number.isFinite(ib))ib=0;if(ia>ib){const t=ia;ia=ib;ib=t}const dates=CUSTOM.dates_ms,N=dates.length;ia=clamp(ia,0,Math.max(N-1,0));ib=clamp(ib,0,Math.max(N-1,0));const s=dates[ia],e=dates[ib],lab=document.getElementById('customDates');if(lab)lab.textContent='du '+ds(s)+' au '+ds(e);const sel=document.getElementById('trackSel'),den=Math.max(N-1,1),p0=100*ia/den,p1=100*ib/den;if(sel){sel.style.left=p0.toFixed(2)+'%';sel.style.width=Math.max(p1-p0,0).toFixed(2)+'%'}const mult=vMult();svg.querySelectorAll('.vec').forEach(line=>{const name=line.getAttribute('data-name')||'';if(!name)return;const ax=CUSTOM.x[name],ay=CUSTOM.y[name],az=CUSTOM.z[name];if(!ax||!ay||!az)return;let fx=-1,lx=-1,fy=-1,ly=-1,fz=-1,lz=-1;for(let i=ia;i<=ib;i++){const x=ax[i],y=ay[i],z=az[i];if(fx<0&&Number.isFinite(x))fx=i;if(fy<0&&Number.isFinite(y))fy=i;if(fz<0&&Number.isFinite(z))fz=i;if(fx>=0&&fy>=0&&fz>=0)break}for(let i=ib;i>=ia;i--){const x=ax[i],y=ay[i],z=az[i];if(lx<0&&Number.isFinite(x))lx=i;if(ly<0&&Number.isFinite(y))ly=i;if(lz<0&&Number.isFinite(z))lz=i;if(lx>=0&&ly>=0&&lz>=0)break}const x1=parseFloat(line.getAttribute('data-x1')),y1=parseFloat(line.getAttribute('data-y1'));if(fx<0||lx<0||fy<0||ly<0||fz<0||lz<0){line.setAttribute('data-vx','0');line.setAttribute('data-vy','0');['dx','dy','dz','dp','ds'].forEach(k=>line.setAttribute('data-'+k,'—'));if(Number.isFinite(x1)&&Number.isFinite(y1)){line.setAttribute('x2',x1.toFixed(3));line.setAttribute('y2',y1.toFixed(3))}return}const dx=ax[lx]-ax[fx],dy=ay[ly]-ay[fy],dz=az[lz]-az[fz],ux=CUSTOM.vx_unit[name],uy=CUSTOM.vy_unit[name],vx=(dx*ux)*vbase,vy=(dy*uy)*vbase;line.setAttribute('data-vx',Number.isFinite(vx)?vx.toFixed(3):'0');line.setAttribute('data-vy',Number.isFinite(vy)?vy.toFixed(3):'0');const dp=h2(dx,dy),sp=h3(dx,dy,dz);line.setAttribute('data-dx',mm(dx));line.setAttribute('data-dy',mm(dy));line.setAttribute('data-dz',mm(dz));line.setAttribute('data-dp',mm(dp));line.setAttribute('data-ds',mm(sp));const c=svg.querySelector('.target[data-name="'+CSS.escape(name)+'"]');if(c){c.setAttribute('data-dx',mm(dx));c.setAttribute('data-dy',mm(dy));c.setAttribute('data-dz',mm(dz));c.setAttribute('data-dp',mm(dp));c.setAttribute('data-ds',mm(sp))}});applyV(mult);upd()}function init(){if(!CUSTOM||!CUSTOM.enabled)return;const a=document.getElementById('rangeA'),b=document.getElementById('rangeB'),N=(CUSTOM.dates_ms||[]).length;if(!a||!b||N<=1)return;a.min='0';a.max=String(N-1);b.min='0';b.max=String(N-1);a.value=String(clamp(CUSTOM.default_i0||0,0,N-1));b.value=String(clamp(CUSTOM.default_i1||N-1,0,N-1));const norm=()=>{let ia=parseInt(a.value,10),ib=parseInt(b.value,10);if(ia>ib){const t=ia;ia=ib;ib=t;a.value=String(ia);b.value=String(ib)}};const on=()=>{norm();req()};a.addEventListener('input',on);b.addEventListener('input',on);req()}applyT();upd();init();</script></body></html>'''

def _wrap(width,height,svg_inner,with_vec,base_px,vec_base,tm,custom=None,hide_v=False,show_xy=True):
    top=""; top_h=0; cjson="null"
    if custom is not None:
        cjson=json.dumps(custom,ensure_ascii=False); top_h=88
        top=r'''<div id="customTop" class="customTop"><div class="customTitle"><span class="t">Période custom</span><span class="d" id="customDates">—</span></div><div class="dualWrap"><div class="trackBase" id="trackBase"></div><div class="trackSel" id="trackSel"></div><input id="rangeA" class="range rangeA" type="range" min="0" max="100" value="20" step="1"/><input id="rangeB" class="range rangeB" type="range" min="0" max="100" value="80" step="1"/></div></div>'''
    view_h=int(height-top_h) if height-top_h>160 else max(int(height),260)
    controls = f'''<div id="controls" class="{'hidden' if not with_vec else ''}">{"<input id='vecScale' class='rng rng-orange' type='range' min='0.1' max='20' value='1' step='0.05'/>" if with_vec else ""}</div>'''
    return (_TPL.replace("__W__",str(int(width))).replace("__H__",str(int(height))).replace("__VIEW_H__",str(int(view_h)))
            .replace("__SVG__",svg_inner).replace("__CONTROLS__",controls).replace("__BASE__",f"{base_px:.10f}")
            .replace("__VBASE__",f"{vec_base:.10f}").replace("__TM__",tm)
            .replace("__SV__","hidden" if (hide_v or not with_vec) else "").replace("__SXY__","" if show_xy else "hidden")
            .replace("__CUSTOM__",cjson).replace("__CUSTOM_TOP__",top))

# ---------- SVG builders ----------
def _svg_targets(points,w=950,h=700):
    if points.empty: return "<div style='padding:12px;font-family:system-ui'>Aucune cible détectée.</div>"
    to_px,base=_mapper(points,w,h)
    cir="".join([f"<circle class='target' cx='{to_px(float(r.x),float(r.y))[0]:.3f}' cy='{to_px(float(r.x),float(r.y))[1]:.3f}' r='7' data-name='{_esc(r.name)}'></circle>"
                 for r in points.itertuples(index=False)])
    return _wrap(w,h,cir,False,base,1.0,"name",None,True,True)

def _svg_mouv(points,deltas,w=950,h=700):
    if points.empty: return "<div style='padding:12px;font-family:system-ui'>Aucune cible détectée.</div>"
    if deltas.empty: return "<div style='padding:12px;font-family:system-ui'>Aucun mouvement calculé.</div>"
    p=points.drop_duplicates("name",keep="last").reset_index(drop=True)
    d=deltas.drop_duplicates("name",keep="last").reset_index(drop=True)
    dd=d.set_index("name")
    to_px,base=_mapper(p,w,h); vec_base=_auto_vec_scale(p,d,to_px,25.0)
    cir=[]; vec=[]
    for r in p.itertuples(index=False):
        name=str(r.name); x,y=float(r.x),float(r.y); px,py=to_px(x,y)
        extra=""
        if name in dd.index:
            dx,dy,dz=float(dd.loc[name,"dx"]),float(dd.loc[name,"dy"]),float(dd.loc[name,"dz"])
            if np.isfinite(dx) and np.isfinite(dy) and np.isfinite(dz) and (abs(dx)+abs(dy)+abs(dz)>1e-15):
                dp=_fmt_hyp_mm(dx,dy,None); ds=_fmt_hyp_mm(dx,dy,dz)
                pex,pey=to_px(x+dx,y+dy); vx=(pex-px)*vec_base; vy=(pey-py)*vec_base
                extra=f" data-dx='{_fmt_mm(dx)}' data-dy='{_fmt_mm(dy)}' data-dz='{_fmt_mm(dz)}' data-dp='{dp}' data-ds='{ds}'"
                vec.append(f"<line class='vec' x1='{px:.3f}' y1='{py:.3f}' x2='{px:.3f}' y2='{py:.3f}' data-x1='{px:.3f}' data-y1='{py:.3f}' data-vx='{vx:.3f}' data-vy='{vy:.3f}' data-name='{_esc(name)}' data-dx='{_fmt_mm(dx)}' data-dy='{_fmt_mm(dy)}' data-dz='{_fmt_mm(dz)}' data-dp='{dp}' data-ds='{ds}'></line>")
        cir.append(f"<circle class='target' cx='{px:.3f}' cy='{py:.3f}' r='7' data-name='{_esc(name)}'{extra}></circle>")
    return _wrap(w,h,"".join(cir)+ "".join(vec),True,base,vec_base,"deltas",None,False,True)

def _svg_custom(points,xlsx,mtime,w=950,h=700):
    if points.empty: return "<div style='padding:12px;font-family:system-ui'>Aucune cible détectée.</div>"
    dates,groups,use,df=_timeseries_xyz(xlsx,mtime)
    if dates.empty or not groups or df.empty: return "<div style='padding:12px;font-family:system-ui'>Aucune série temporelle exploitable.</div>"
    p=points.drop_duplicates("name",keep="last").reset_index(drop=True)
    to_px,base=_mapper(p,w,h)
    try: dfull=_deltas_first_last(xlsx,mtime)
    except Exception: dfull=pd.DataFrame(columns=["name","dx","dy","dz"])
    vec_base=_auto_vec_scale(p,dfull,to_px,25.0)
    pos={c:i for i,c in enumerate(use)}
    dates_ms=(dates.astype("int64")//1_000_000).to_list(); N=len(dates_ms)
    x_map,y_map,z_map,vxu,vyu={}, {}, {}, {}, {}
    for name,jx,jy,jz in groups:
        if name is None or jx not in pos or jy not in pos or jz not in pos: continue
        ax=df.iloc[:,pos[jx]].to_numpy(float,copy=False); ay=df.iloc[:,pos[jy]].to_numpy(float,copy=False); az=df.iloc[:,pos[jz]].to_numpy(float,copy=False)
        x_map[name]=[None if not np.isfinite(v) else float(v) for v in ax]
        y_map[name]=[None if not np.isfinite(v) else float(v) for v in ay]
        z_map[name]=[None if not np.isfinite(v) else float(v) for v in az]
        pp=p[p["name"]==name]
        if pp.empty: continue
        x0,y0=float(pp.iloc[0]["x"]),float(pp.iloc[0]["y"])
        px0,py0=to_px(x0,y0); px1,_=to_px(x0+1.0,y0); _,py2=to_px(x0,y0+1.0)
        vxu[name]=float(px1-px0); vyu[name]=float(py2-py0)
    if N<=1: i0=i1=0
    else:
        last=dates_ms[-1]; start=last-int(30*24*3600*1000)
        i0=int(np.searchsorted(np.array(dates_ms,np.int64),start,side="left")); i0=max(min(i0,N-1),0); i1=N-1
    cir=[]; vec=[]
    for r in p.itertuples(index=False):
        name=str(r.name); x,y=float(r.x),float(r.y); px,py=to_px(x,y)
        cir.append(f"<circle class='target' cx='{px:.3f}' cy='{py:.3f}' r='7' data-name='{_esc(name)}'></circle>")
        vec.append(f"<line class='vec' x1='{px:.3f}' y1='{py:.3f}' x2='{px:.3f}' y2='{py:.3f}' data-x1='{px:.3f}' data-y1='{py:.3f}' data-vx='0' data-vy='0' data-name='{_esc(name)}' data-dx='—' data-dy='—' data-dz='—' data-dp='—' data-ds='—'></line>")
    payload={"enabled":True,"dates_ms":dates_ms,"default_i0":i0,"default_i1":i1,"x":x_map,"y":y_map,"z":z_map,"vx_unit":vxu,"vy_unit":vyu}
    return _wrap(w,h,"".join(cir)+ "".join(vec),True,base,vec_base,"deltas",payload,False,True)

# ---------- entrypoint ----------
def render_topo_projections(selected_xlsx: str, xlsx_abs_path: str, force_key: int):
    root=_root(); wp=None
    if xlsx_abs_path:
        p=Path(xlsx_abs_path).expanduser(); p=(root/p).resolve() if not p.is_absolute() else p.resolve()
        wp=p if p.exists() and p.is_file() and p.suffix.lower()==".xlsx" else None
    if wp is None and selected_xlsx:
        q=Path(selected_xlsx).expanduser()
        if not q.is_absolute():
            for base in (root, root/"data", root/"data/common_data", root/"data/topo"):
                p=(base/q.name).resolve()
                if p.exists() and p.is_file() and p.suffix.lower()==".xlsx": wp=p; break
            if wp is None:
                p=(root/q).resolve()
                if p.exists() and p.is_file() and p.suffix.lower()==".xlsx": wp=p
        else:
            q=q.resolve()
            if q.exists() and q.is_file() and q.suffix.lower()==".xlsx": wp=q
    if wp is None:
        cd=root/"data/common_data"; wp=_find_mc(cd)
        if wp is None or not wp.exists():
            st.error("Fichier Excel introuvable.\n\n"
                     f"- Chemin demandé (xlsx_abs_path) : {xlsx_abs_path or '—'}\n"
                     f"- Sélection (selected_xlsx)      : {selected_xlsx or '—'}\n"
                     f"- Fallback attendu               : {cd}/Mesures Completes*.xlsx")
            return
    mtime=float(wp.stat().st_mtime)+float(force_key or 0)
    try: dmin,dmax=_date_range(str(wp),mtime)
    except Exception: dmin=dmax=None
    try: points=_xy_medians(str(wp),mtime)
    except Exception as e: st.error("Impossible de lire le classeur (médianes)."); st.caption(str(e)); return

    tab1,tab2,tab3,tab4,tab5=st.tabs(["Cibles","Mouvement connu","Mois passé","Semaine passée","Période custom"])
    with tab1:
        st.markdown("### Ensemble des cibles répertoriées")
        components.html(_svg_targets(points,950,700),height=720,scrolling=False)

    with tab2:
        st.markdown(f"### Mouvements connus au {_fmt_date(dmax)} depuis le {_fmt_date(dmin)}")
        try: d=_deltas_first_last(str(wp),mtime)
        except Exception as e: st.error("Impossible de calculer le mouvement connu (first/last)."); st.caption(str(e)); return
        components.html(_svg_mouv(points,d,950,700),height=720,scrolling=False)

    with tab3:
        s,e=_win(dmax,30)
        st.markdown(f"### Mouvements connus au cours du mois passé (du {_fmt_date(s)} au {_fmt_date(e)})")
        try: d=_deltas_last_days(str(wp),mtime,30)
        except Exception as ex: st.error("Impossible de calculer le mouvement sur les 30 derniers jours."); st.caption(str(ex)); return
        components.html(_svg_mouv(points,d,950,700),height=720,scrolling=False)

    with tab4:
        s,e=_win(dmax,7)
        st.markdown(f"### Mouvements connus au cours de la semaine passée (du {_fmt_date(s)} au {_fmt_date(e)})")
        try: d=_deltas_last_days(str(wp),mtime,7)
        except Exception as ex: st.error("Impossible de calculer le mouvement sur les 7 derniers jours."); st.caption(str(ex)); return
        components.html(_svg_mouv(points,d,950,700),height=720,scrolling=False)

    with tab5:
        st.markdown("### Mouvements sur une période personnalisée")
        try: html=_svg_custom(points,str(wp),mtime,950,700)
        except Exception as ex: st.error("Impossible de préparer le mode live."); st.caption(str(ex)); return
        components.html(html,height=780,scrolling=False)
