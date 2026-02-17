from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = None

# ✅ bump version (anti-cache dur)
_BASE = Path(tempfile.gettempdir()) / "geodashboard_selection_cibles_component_v19_front"


def selection_cibles_component(data: dict, key: str):
    _ensure_component()
    return _COMPONENT(data=data, default=None, key=key)  # type: ignore[name-defined]


def _ensure_component() -> None:
    global _COMPONENT
    _BASE.mkdir(parents=True, exist_ok=True)

    html = _index_html()
    idx = _BASE / "index.html"

    # ✅ Écrit seulement si différent (évite I/O à chaque run)
    try:
        if idx.exists():
            old = idx.read_text(encoding="utf-8")
            if old != html:
                idx.write_text(html, encoding="utf-8")
        else:
            idx.write_text(html, encoding="utf-8")
    except Exception:
        idx.write_text(html, encoding="utf-8")

    if _COMPONENT is None:
        # ✅ name changé => Streamlit ne peut PAS réutiliser l’ancien composant
        _COMPONENT = components.declare_component("selection_cibles_component_v19", path=str(_BASE))


def _index_html() -> str:
    return r"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
  html, body { margin:0; padding:0; background:#fff; font-family: system-ui; }
  .root { width: 100%; display:flex; justify-content:center; }
  .panel { width:100%; max-width:1200px; }
  .topbar { display:flex; align-items:center; justify-content:space-between; padding:8px 2px 10px 2px; gap:10px; }
  .btn { appearance:none; border:0; background:#111827; color:#fff; font-weight:800; font-size:13px;
         padding:10px 12px; border-radius:12px; cursor:pointer; box-shadow:0 10px 20px rgba(0,0,0,.10); }
  .btn:active { transform: translateY(1px); }
  .ver { font-size:12px; font-weight:800; color:rgba(17,24,39,.55); padding:0 10px; user-select:none; }

  .wrap { width:100%; height:680px; position:relative; user-select:none; overflow:hidden; border-radius:14px; }
  svg { width:100%; height:100%; background:#fff; border-radius:14px; touch-action:none; }

  .hud { position:absolute; right:10px; top:10px; bottom:10px; width:290px; background:rgba(255,255,255,.92);
         border:1px solid rgba(0,0,0,.10); border-radius:12px; padding:10px; font-size:11px;
         box-shadow:0 10px 22px rgba(0,0,0,.10); opacity:0; pointer-events:none; transform:translateX(6px);
         transition:opacity .10s linear, transform .10s linear; display:flex; flex-direction:column; }
  .hud.show { opacity:1; transform:translateX(0); pointer-events:auto; }
  .hudTop { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:6px; }
  .hud .title { font-weight:900; margin:0; }
  .hud .meta { opacity:.8; margin-bottom:8px; }
  .hud .list { overflow:auto; flex:1; padding-right:2px; }
  .pill { display:inline-block; margin:2px 6px 2px 0; padding:2px 7px; border-radius:999px;
          border:1px solid rgba(0,0,0,.10); background:rgba(0,0,0,.03); max-width:100%;
          white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

  .bgImg { opacity:.92; }

  .pt { fill:#eb3838; stroke:rgba(0,0,0,.15); stroke-width:1; }
  .pt.sel { fill: var(--sel-color, #146EFF); }

  .zonePoly { fill: var(--zone-weak); stroke: var(--zone-stroke); stroke-width:2; stroke-dasharray:6 4; pointer-events:none; }
  .corner { fill: var(--zone-color); stroke: rgba(0,0,0,.14); stroke-width:1; cursor:pointer; }
  .zoneLabel { fill: var(--zone-color); font-weight:900; text-anchor:middle; dominant-baseline:middle; pointer-events:none;
               paint-order:stroke fill; stroke:rgba(0,0,0,.92); stroke-width:3px; stroke-linejoin:round; }

  .vecLine { stroke: var(--zone-color); stroke-linecap: round; pointer-events:none; }
  .vecHead { fill: var(--zone-color); pointer-events:none; }
  .vecHandle { fill: rgba(255,255,255,.92); stroke: var(--zone-color); cursor:pointer; }
  .vecPerp { stroke: var(--zone-color); stroke-linecap: round; opacity:.95; pointer-events:none; }

  .vecTools {
    position:absolute;
    display:none;
    align-items:center;
    gap:6px;
    padding:6px 7px;
    border-radius:12px;
    border:1px solid rgba(0,0,0,.10);
    background:rgba(255,255,255,.96);
    box-shadow:0 10px 22px rgba(0,0,0,.12);
    pointer-events:auto;
    user-select:none;
    z-index: 50;
  }
  .toolBtn{
    width:28px; height:28px;
    border-radius:11px;
    border:1px solid rgba(0,0,0,.10);
    background:rgba(0,0,0,.03);
    cursor:pointer;
    display:flex; align-items:center; justify-content:center;
    font-size:14px; line-height:1;
  }
  .toolBtn:active{ transform:translateY(1px); }
  .toolBtn.locked { background:rgba(17,24,39,.10); border-color:rgba(17,24,39,.20); }
  .toolBtn.delete { background:rgba(235,56,56,.10); border-color:rgba(235,56,56,.22); color:#eb3838; }

  .hint {
    position:absolute;
    left:10px; bottom:10px;
    font-size:11px;
    color:rgba(17,24,39,.60);
    background:rgba(255,255,255,.85);
    border:1px solid rgba(0,0,0,.08);
    border-radius:12px;
    padding:6px 9px;
    user-select:none;
    pointer-events:none;
  }
</style>
</head>

<body>
<div class="root">
  <div class="panel" id="panel">
    <div class="topbar">
      <button class="btn" id="frontBtn">Mettre à jour</button>
      <div class="ver" id="ver">FRONT v19 (auto alpha from backend; fallback angle_deg)</div>
    </div>

    <div class="wrap" id="wrap">
      <svg id="svg" viewBox="0 0 1200 680" preserveAspectRatio="xMidYMid meet"><g id="vp"></g></svg>

      <div class="vecTools" id="vecTools">
        <button class="toolBtn" id="toolLock" title="Verrouiller / Déverrouiller">🔒</button>
        <button class="toolBtn delete" id="toolDel" title="Supprimer le vecteur">✖</button>
      </div>

      <div class="hud" id="hud">
        <div class="hudTop">
          <div class="title" id="hudTitle">—</div>
        </div>
        <div class="meta">
          α = <b><span id="hudAlpha">—</span>°</b>
          <span style="opacity:.7;margin-left:8px;">| cibles: <b><span id="hudCount">0</span></b></span>
        </div>
        <div class="list" id="hudList"></div>
      </div>

      <div class="hint">Molette=zoom • Drag=pan • Shift+drag=déplacer fond • Shift+molette=zoom fond</div>
    </div>
  </div>
</div>

<script>
// -------- streamlit bridge
const API=1;
const post=(type,extra)=>window.parent.postMessage(Object.assign({isStreamlitMessage:true,type,apiVersion:API},extra||{}),"*");
post("streamlit:componentReady");
const setH=h=>post("streamlit:setFrameHeight",{height:h});
const setV=v=>post("streamlit:setComponentValue",{value:v});

// state global page (pan/zoom)
const STATE = window.__GD_FRONT_STATE__ || (window.__GD_FRONT_STATE__ = {tx:0,ty:0,sc:1,active:null});

addEventListener("message",(e)=>{
  const m=e.data;
  if(!m||m.type!=="streamlit:render")return;
  const D=m.args&&m.args.data;
  if(D) boot(D);
});

function boot(D){
  console.log("[GeoDashboard] FRONT v19 loaded");

  const W=+D.w||1200, H=+D.h||680;
  document.getElementById("panel").style.maxWidth=W+"px";
  document.getElementById("wrap").style.height=H+"px";
  setH(H+70);

  const svg=document.getElementById("svg");
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  const vp=document.getElementById("vp");
  while(vp.firstChild)vp.removeChild(vp.firstChild);

  const wrapEl=document.getElementById("wrap");
  const hud=document.getElementById("hud");
  const hudTitle=document.getElementById("hudTitle");
  const hudAlpha=document.getElementById("hudAlpha");
  const hudCount=document.getElementById("hudCount");
  const hudList=document.getElementById("hudList");

  const vecTools=document.getElementById("vecTools");
  const toolLock=document.getElementById("toolLock");
  const toolDel=document.getElementById("toolDel");

  const zonesIn=Array.isArray(D.zones)?D.zones:[];
  const pointsIn=Array.isArray(D.points)?D.points:[];

  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const n2=(x,y)=>Math.sqrt(x*x+y*y);
  const deg=(r)=>r*180/Math.PI;
  const aTrig=(dx,dy)=>Math.atan2(-dy,dx);
  const signed=(d)=>((d+180)%360+360)%360-180;

  // alpha (deg) -> unit direction (dx,dy) compatible avec alpha_from_vector
  const dirFromAlpha=(alphaDeg)=>{
    const rad=(+alphaDeg||0)*Math.PI/180;
    const ux=Math.cos(rad);
    const uy=-Math.sin(rad);
    const nn=Math.sqrt(ux*ux+uy*uy);
    if(nn<1e-12) return {ux:1,uy:0};
    return {ux:ux/nn, uy:uy/nn};
  };

  const hex2rgb=(hex)=>{
    let h=String(hex||"").trim().replace("#","");
    if(h.length===3)h=h.split("").map(c=>c+c).join("");
    if(h.length!==6)return {r:20,g:110,b:255};
    return {r:parseInt(h.slice(0,2),16),g:parseInt(h.slice(2,4),16),b:parseInt(h.slice(4,6),16)};
  };
  const zoneVars=(color)=>{
    const rgb=hex2rgb(color);
    const c=(color||"#146EFF");
    return {color:c,weak:`rgba(${rgb.r},${rgb.g},${rgb.b},0.10)`,stroke:`rgba(${rgb.r},${rgb.g},${rgb.b},0.78)`};
  };

  const svgPt=(cx,cy)=>{
    const pt=svg.createSVGPoint(); pt.x=cx; pt.y=cy;
    const ctm=svg.getScreenCTM(); if(!ctm)return {x:0,y:0};
    const p=pt.matrixTransform(ctm.inverse());
    return {x:p.x,y:p.y};
  };

  let tx=+STATE.tx||0, ty=+STATE.ty||0, sc=clamp(+STATE.sc||1,0.15,25);
  let active=(STATE.active ?? null);

  const world=(cx,cy)=>{const p=svgPt(cx,cy); return {x:(p.x-tx)/sc,y:(p.y-ty)/sc};};
  const toClient=(x,y)=>{
    const pt=svg.createSVGPoint(); pt.x=x*sc+tx; pt.y=y*sc+ty;
    const ctm=svg.getScreenCTM(); if(!ctm)return {x:0,y:0};
    const p=pt.matrixTransform(ctm);
    return {x:p.x,y:p.y};
  };

  const inPoly=(x,y,P)=>{
    let inside=false;
    for(let i=0,j=P.length-1;i<P.length;j=i++){
      const xi=P[i].x,yi=P[i].y,xj=P[j].x,yj=P[j].y;
      const hit=((yi>y)!==(yj>y))&&(x<(xj-xi)*(y-yi)/(yj-yi+1e-12)+xi);
      if(hit)inside=!inside;
    }
    return inside;
  };
  const quadC=(Q)=>({x:(Q[0].x+Q[1].x+Q[2].x+Q[3].x)/4,y:(Q[0].y+Q[1].y+Q[2].y+Q[3].y)/4});

  const closestOnSeg=(P,A,B)=>{
    const abx=B.x-A.x, aby=B.y-A.y;
    const apx=P.x-A.x, apy=P.y-A.y;
    const ab2=abx*abx+aby*aby;
    if(ab2<1e-12) return {x:A.x,y:A.y,t:0};
    let t=(apx*abx+apy*aby)/ab2;
    t=Math.max(0,Math.min(1,t));
    return {x:A.x+t*abx, y:A.y+t*aby, t};
  };

  const clampToPoly=(P,poly)=>{
    if(inPoly(P.x,P.y,poly)) return {x:P.x,y:P.y,clamped:false};
    let best=null, bestD=1e99;
    for(let i=0;i<poly.length;i++){
      const A=poly[i], B=poly[(i+1)%poly.length];
      const Q=closestOnSeg(P,A,B);
      const dx=P.x-Q.x, dy=P.y-Q.y;
      const d=dx*dx+dy*dy;
      if(d<bestD){ bestD=d; best=Q; }
    }
    if(!best) return {x:P.x,y:P.y,clamped:false};
    return {x:best.x,y:best.y,clamped:true};
  };

  const linePolyIntersections=(P, d, poly)=>{
    const out=[];
    const eps=1e-9;
    for(let i=0;i<poly.length;i++){
      const A=poly[i], B=poly[(i+1)%poly.length];
      const ex=B.x-A.x, ey=B.y-A.y;
      const det = d.x*(-ey) - d.y*(-ex);
      if(Math.abs(det) < eps) continue;
      const rx=A.x-P.x, ry=A.y-P.y;
      const t = (rx*(-ey) - ry*(-ex)) / det;
      const u = (d.x*ry - d.y*rx) / det;
      if(u >= -1e-6 && u <= 1+1e-6){
        out.push({t, x:P.x + t*d.x, y:P.y + t*d.y});
      }
    }
    return out;
  };

  // ✅ localStorage ONLY for quads + vectors (NOT for background transform)
  const Mem={
    p:"geodashboard.front.v1",
    k:(s,n)=>`${Mem.p}.${s}.${n}`,
    load:(s,n,fb)=>{try{const r=localStorage.getItem(Mem.k(s,n));return r?JSON.parse(r):fb;}catch(_){return fb;}},
    save:(s,n,v)=>{try{localStorage.setItem(Mem.k(s,n),JSON.stringify(v));}catch(_){}}
  };
  const scope="selection_cibles";

  // -----------------------------
  // Background (backend-only truth)
  // -----------------------------
  const bgIn = (D.bg && typeof D.bg==="object") ? D.bg : null;
  const bgEnabled = !!(bgIn && bgIn.enabled && bgIn.img_url);

  const bgG=document.createElementNS("http://www.w3.org/2000/svg","g");
  vp.appendChild(bgG);

  let bgX=0, bgY=0, bgS=1;
  let bgApply=null;

  if(bgEnabled){
    const img=document.createElementNS("http://www.w3.org/2000/svg","image");
    img.setAttribute("class","bgImg");
    img.setAttribute("href", bgIn.img_url);

    const iw = +bgIn.img_w || W;
    const ih = +bgIn.img_h || H;

    img.setAttribute("x","0");
    img.setAttribute("y","0");
    img.setAttribute("width", String(iw));
    img.setAttribute("height", String(ih));
    img.setAttribute("preserveAspectRatio","xMidYMid meet");
    bgG.appendChild(img);

    // ✅ ONLY payload transform (no localStorage)
    const trPayload = (bgIn.transform && typeof bgIn.transform==="object") ? bgIn.transform : null;
    const tr = trPayload || {x:0,y:0,scale:1};

    bgX = +tr.x || 0;
    bgY = +tr.y || 0;
    bgS = clamp(+tr.scale || 1, 0.02, 50);

    bgApply=()=>bgG.setAttribute("transform",`translate(${bgX} ${bgY}) scale(${bgS})`);
    bgApply();
  }

  // -----------------------------
  // Points
  // -----------------------------
  const PTS=pointsIn.map(p=>({name:String(p.name||""),x:+p.px,y:+p.py,z:+p.z,el:null}));
  for(const p of PTS){
    const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
    c.setAttribute("class","pt"); c.setAttribute("r","5.8");
    c.setAttribute("cx",p.x); c.setAttribute("cy",p.y);
    vp.appendChild(c); p.el=c;
  }

  // -----------------------------
  // Zones + vectors
  // -----------------------------
  const defQuad=(i)=> i===0 ? [
    {x:W*0.28,y:H*0.22},{x:W*0.55,y:H*0.20},{x:W*0.58,y:H*0.55},{x:W*0.31,y:H*0.58},
  ] : [
    {x:W*0.55,y:H*0.28},{x:W*0.80,y:H*0.26},{x:W*0.82,y:H*0.62},{x:W*0.57,y:H*0.65},
  ];

  const zones=[];
  for(let i=0;i<zonesIn.length;i++){
    const zi=zonesIn[i];
    const id=String(zi.id ?? (i+1));
    const name=String(zi.name ?? id);
    const idx=(zi.idx==null)?(i+1):+zi.idx;
    const zv=zoneVars(zi.color || "#146EFF");
    const memQuad=String(zi.mem_quad || ("quad_"+id));
    const memVec=String("vec_"+id);

    const quad=Mem.load(scope, memQuad, defQuad(i));
    const vecState=Mem.load(scope, memVec, null);

    // ✅ PRIORITÉ: alpha_backend (Python). Fallback: angle_deg (JSON). Fallback final: 0.0
    const a1 = (zi.alpha_backend != null && isFinite(+zi.alpha_backend)) ? +zi.alpha_backend : null;
    const a2 = (zi.angle_deg     != null && isFinite(+zi.angle_deg))     ? +zi.angle_deg     : null;
    const autoAlpha = (a1 != null) ? a1 : ((a2 != null) ? a2 : 0.0);

    const poly=document.createElementNS("http://www.w3.org/2000/svg","polygon");
    poly.setAttribute("class","zonePoly");
    poly.style.setProperty("--zone-color",zv.color);
    poly.style.setProperty("--zone-weak",zv.weak);
    poly.style.setProperty("--zone-stroke",zv.stroke);
    vp.appendChild(poly);

    const corners=[];
    for(let k=0;k<4;k++){
      const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
      c.setAttribute("class","corner");
      c.style.setProperty("--zone-color",zv.color);
      c.dataset.k=String(k);
      vp.appendChild(c);
      corners.push(c);
    }

    const label=document.createElementNS("http://www.w3.org/2000/svg","text");
    label.setAttribute("class","zoneLabel");
    label.style.setProperty("--zone-color",zv.color);
    label.textContent=name;
    vp.appendChild(label);

    const vecG=document.createElementNS("http://www.w3.org/2000/svg","g");
    vecG.style.display="none";
    vp.appendChild(vecG);

    // perpendicular segment
    const vPerp=document.createElementNS("http://www.w3.org/2000/svg","line");
    vPerp.setAttribute("class","vecPerp");
    vPerp.style.setProperty("--zone-color",zv.color);
    vecG.appendChild(vPerp);

    const vLine=document.createElementNS("http://www.w3.org/2000/svg","line");
    vLine.setAttribute("class","vecLine");
    vLine.style.setProperty("--zone-color",zv.color);
    vecG.appendChild(vLine);

    const vHead=document.createElementNS("http://www.w3.org/2000/svg","polygon");
    vHead.setAttribute("class","vecHead");
    vHead.style.setProperty("--zone-color",zv.color);
    vecG.appendChild(vHead);

    const h0=document.createElementNS("http://www.w3.org/2000/svg","circle");
    h0.setAttribute("class","vecHandle");
    h0.style.setProperty("--zone-color",zv.color);
    h0.dataset.role="v0";
    h0.style.display="none";
    vecG.appendChild(h0);

    const h1=document.createElementNS("http://www.w3.org/2000/svg","circle");
    h1.setAttribute("class","vecHandle");
    h1.style.setProperty("--zone-color",zv.color);
    h1.dataset.role="v1";
    h1.style.display="none";
    vecG.appendChild(h1);

    zones.push({
      id,name,idx,color:zv.color,memQuad,quad,
      memVec, vecState,
      autoAlpha,
      poly,corners,label,vecG,
      vPerp,vLine,vHead,h0,h1,
      selectedNames:[], selectedPts:[], alphaSigned:null,
      selSig:"",
      dragCorner:null,
      vx0:null,vy0:null,vx1:null,vy1:null,
    });
  }

  const syncGeom=(z)=>{
    z.poly.setAttribute("points",z.quad.map(p=>p.x+","+p.y).join(" "));
    for(let k=0;k<4;k++){
      const p=z.quad[k];
      z.corners[k].setAttribute("cx",p.x);
      z.corners[k].setAttribute("cy",p.y);
      z.corners[k].setAttribute("r",String(7/sc));
      z.corners[k].setAttribute("stroke-width",String(1/sc));
    }
    const c=quadC(z.quad);
    z.label.setAttribute("x",c.x); z.label.setAttribute("y",c.y);
    z.label.setAttribute("font-size",String(16/sc));
    z.label.setAttribute("stroke-width",String(3/sc));
    z.poly.setAttribute("stroke-width",String(2/sc));
  };

  const sel=(z)=>{
    z.selectedNames=[]; z.selectedPts=[];
    for(const p of PTS){
      if(inPoly(p.x,p.y,z.quad)){
        z.selectedNames.push(p.name);
        z.selectedPts.push({x:p.x,y:p.y,z:p.z,name:p.name});
      }
    }
    z.selectedNames.sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
    z.selSig = z.selectedNames.join("|");
  };

  const setVecWorld=(z, x0,y0,x1,y1)=>{ z.vx0=x0; z.vy0=y0; z.vx1=x1; z.vy1=y1; };

  const maybeResetCustomOnSelChange=(z)=>{
    const st=z.vecState || null;
    if(!st) return;
    if(!st.custom) return;
    if(st.deleted) return;
    if(st.locked) return;

    const prevStored = (typeof st.selSig==="string") ? st.selSig : "";
    if(prevStored && prevStored === z.selSig){
      return;
    }

    const next = Object.assign({}, st, {custom:false, selSig:z.selSig});
    z.vecState = next;
    Mem.save(scope, z.memVec, next);
  };

  // ✅ AUTO vector = uniquement basé sur alpha (backendAlpha/angle_deg/0)
  const computeAutoVec=(z)=>{
    if(z.selectedPts.length<3) return null;

    // base = min Z
    let base=z.selectedPts[0];
    for(const p of z.selectedPts){ if(p.z<base.z) base=p; }

    const a = (typeof z.autoAlpha==="number" && isFinite(z.autoAlpha)) ? z.autoAlpha : 0.0;
    const d = dirFromAlpha(a);
    const L = 170;

    const x0=base.x, y0=base.y;
    const x1=x0 + d.ux*L, y1=y0 + d.uy*L;
    return {x0,y0,x1,y1};
  };

  const renderVec=(z)=>{
    const st=z.vecState || {};
    const deleted=!!st.deleted;

    if(deleted || z.selectedPts.length<3){
      z.vecG.style.display="none";
      z.alphaSigned=null;
      return;
    }

    let x0,y0,x1,y1;
    if(st && st.custom && typeof st.x0==="number"){
      x0=+st.x0; y0=+st.y0; x1=+st.x1; y1=+st.y1;
    } else {
      const d=computeAutoVec(z);
      if(!d){ z.vecG.style.display="none"; z.alphaSigned=null; return; }
      x0=d.x0; y0=d.y0; x1=d.x1; y1=d.y1;
    }

    // base clamp inside zone
    const cl = clampToPoly({x:x0,y:y0}, z.quad);
    if(cl.clamped){
      const dx=x1-x0, dy=y1-y0;
      x0=cl.x; y0=cl.y;
      x1=x0+dx; y1=y0+dy;
      const next=Object.assign({}, z.vecState||{}, {x0,y0,x1,y1, custom:true, deleted:false, selSig:z.selSig});
      z.vecState=next; Mem.save(scope, z.memVec,next);
    }

    setVecWorld(z,x0,y0,x1,y1);
    z.vecG.style.display="block";

    // perp segment limited by polygon
    const dx=x1-x0, dy=y1-y0;
    const norm=Math.sqrt(dx*dx+dy*dy);
    if(norm<1e-9){ z.vecG.style.display="none"; z.alphaSigned=null; return; }

    const ux=dx/norm, uy=dy/norm;
    const px=-uy, py=ux;

    const inter = linePolyIntersections({x:x0,y:y0}, {x:px,y:py}, z.quad);
    if(inter.length >= 2){
      let neg=null, pos=null;
      for(const it of inter){
        if(it.t < 0){
          if(!neg || it.t > neg.t) neg=it;
        } else {
          if(!pos || it.t < pos.t) pos=it;
        }
      }
      if(!neg || !pos){
        inter.sort((a,b)=>a.t-b.t);
        neg=inter[0]; pos=inter[inter.length-1];
      }
      z.vPerp.style.display="block";
      z.vPerp.setAttribute("x1", String(neg.x));
      z.vPerp.setAttribute("y1", String(neg.y));
      z.vPerp.setAttribute("x2", String(pos.x));
      z.vPerp.setAttribute("y2", String(pos.y));
      z.vPerp.setAttribute("stroke-width",String(3/sc));
    } else {
      z.vPerp.style.display="none";
    }

    z.vLine.setAttribute("x1",x0); z.vLine.setAttribute("y1",y0);
    z.vLine.setAttribute("x2",x1); z.vLine.setAttribute("y2",y1);
    z.vLine.setAttribute("stroke-width",String(4/sc));

    const headLen=18/sc, headW=11/sc;
    const bx=x1-ux*headLen, by=y1-uy*headLen, nx=-uy, ny=ux;
    z.vHead.setAttribute("points",`${x1},${y1} ${bx+nx*headW},${by+ny*headW} ${bx-nx*headW},${by-ny*headW}`);

    const hr=8/sc;
    z.h0.setAttribute("cx",x0); z.h0.setAttribute("cy",y0);
    z.h1.setAttribute("cx",x1); z.h1.setAttribute("cy",y1);
    z.h0.setAttribute("r",String(hr));
    z.h1.setAttribute("r",String(hr));
    z.h0.setAttribute("stroke-width",String(2/sc));
    z.h1.setAttribute("stroke-width",String(2/sc));

    z.alphaSigned=signed(deg(aTrig(x1-x0,y1-y0)));
  };

  const recolor=()=>{
    for(const p of PTS){p.el.classList.remove("sel"); p.el.style.removeProperty("--sel-color");}
    for(const p of PTS){
      for(const z of zones){
        if(inPoly(p.x,p.y,z.quad)){
          p.el.classList.add("sel");
          p.el.style.setProperty("--sel-color", z.color);
          break;
        }
      }
    }
  };

  const updateAll=()=>{
    for(const z of zones){
      syncGeom(z);
      sel(z);
      maybeResetCustomOnSelChange(z);
      renderVec(z);
    }
    recolor();
  };

  const distSeg=(p,a,b)=>{
    const abx=b.x-a.x,aby=b.y-a.y, apx=p.x-a.x,apy=p.y-a.y, ab2=abx*abx+aby*aby;
    if(ab2<1e-9){const dx=p.x-a.x,dy=p.y-a.y;return Math.sqrt(dx*dx+dy*dy);}
    let t=(apx*abx+apy*aby)/ab2; t=Math.max(0,Math.min(1,t));
    const qx=a.x+t*abx,qy=a.y+t*aby, dx=p.x-qx,dy=p.y-qy;
    return Math.sqrt(dx*dx+dy*dy);
  };

  const showHandlesFor=(z)=>{
    for(const zz of zones){ zz.h0.style.display="none"; zz.h1.style.display="none"; }
    if(!z) return;
    const st=z.vecState || {};
    if(st.deleted) return;
    if(z.vx0==null) return;
    if(!!st.locked) return;
    z.h0.style.display="block";
    z.h1.style.display="block";
  };

  let hoveredZone=null;
  const pickZone=(cx,cy)=>{
    const w=world(cx,cy);
    for(let i=zones.length-1;i>=0;i--){
      if(inPoly(w.x,w.y,zones[i].quad)) return zones[i];
    }
    return null;
  };

  const showHud=(z)=>{
    hoveredZone=z||null;
    if(!z){
      hud.classList.remove("show");
      showHandlesFor(null);
      return;
    }
    hudTitle.textContent=z.name;
    hudAlpha.textContent=(z.alphaSigned==null)?"—":z.alphaSigned.toFixed(1);
    hudCount.textContent=String(z.selectedNames.length);
    hudList.innerHTML="";
    for(const n of z.selectedNames){
      const s=document.createElement("span"); s.className="pill"; s.title=n; s.textContent=n;
      hudList.appendChild(s);
    }
    hud.classList.add("show");
    showHandlesFor(z);
  };

  let hoveredVecZone=null;
  let toolsHover=false;
  let hideTimer=null;

  const cancelHide=()=>{ if(hideTimer){ clearTimeout(hideTimer); hideTimer=null; } };
  const scheduleHide=(ms=220)=>{
    if(toolsHover) return;
    cancelHide();
    hideTimer=setTimeout(()=>{
      if(toolsHover) return;
      vecTools.style.display="none";
      hoveredVecZone=null;
    }, ms);
  };

  const pickVectorHover=(cx,cy)=>{
    const P={x:cx,y:cy};
    for(const z of zones){
      const st=z.vecState||{};
      if(st.deleted) continue;
      if(z.vx0==null) continue;
      const a=toClient(z.vx0,z.vy0), b=toClient(z.vx1,z.vy1);
      if(distSeg(P,a,b)<=14) return z;
    }
    return null;
  };

  const placeToolsFor=(z)=>{
    if(!z){ scheduleHide(0); return; }
    const st=z.vecState||{};
    if(st.deleted || z.vx0==null || z.vx1==null){ scheduleHide(0); return; }

    cancelHide();

    const mx=(z.vx0+z.vx1)/2;
    const my=(z.vy0+z.vy1)/2;

    const p=toClient(mx,my);
    const wrap=wrapEl.getBoundingClientRect();
    const x=p.x - wrap.left;
    const y=p.y - wrap.top;

    vecTools.style.display="flex";

    const pad=8;
    const tw=vecTools.offsetWidth || 90;
    const th=vecTools.offsetHeight || 42;
    const cx=clamp(x - tw/2, pad, wrap.width - tw - pad);
    const cy=clamp(y - th/2, pad, wrap.height - th - pad);

    vecTools.style.left = cx + "px";
    vecTools.style.top  = cy + "px";

    toolLock.classList.toggle("locked", !!st.locked);
    toolLock.textContent = st.locked ? "🔒" : "🔓";
  };

  vecTools.addEventListener("pointerenter", ()=>{ toolsHover=true; cancelHide(); });
  vecTools.addEventListener("pointerleave", ()=>{
    toolsHover=false;
    if(!hoveredVecZone) scheduleHide(220);
  });

  vecTools.onpointerdown=(e)=>{ e.stopPropagation(); };

  toolLock.onclick=(e)=>{
    e.preventDefault(); e.stopPropagation();
    if(!hoveredVecZone) return;

    const z = hoveredVecZone;
    const st = z.vecState || {};
    const nowLocked = !st.locked;

    let next;
    if(nowLocked){
      if(typeof z.vx0 === "number" && typeof z.vx1 === "number"){
        next = Object.assign({}, st, {
          locked: true,
          deleted: false,
          custom: true,
          x0: z.vx0, y0: z.vy0, x1: z.vx1, y1: z.vy1,
          selSig: z.selSig
        });
      } else {
        next = Object.assign({}, st, {locked: true, selSig: z.selSig});
      }
    } else {
      next = Object.assign({}, st, {locked: false, selSig: z.selSig});
    }

    z.vecState = next;
    Mem.save(scope, z.memVec, next);

    placeToolsFor(z);
    showHandlesFor(hoveredZone);
  };

  toolDel.onclick=(e)=>{
    e.preventDefault(); e.stopPropagation();
    if(!hoveredVecZone) return;
    const st=hoveredVecZone.vecState || {};
    const next=Object.assign({}, st, {deleted:true, selSig: hoveredVecZone.selSig});
    hoveredVecZone.vecState=next;
    Mem.save(scope, hoveredVecZone.memVec, next);
    applyTf();
    hoveredVecZone=null;
    scheduleHide(0);
    if(hoveredZone) showHud(hoveredZone);
  };

  const applyTf=()=>{
    vp.setAttribute("transform",`translate(${tx} ${ty}) scale(${sc})`);
    updateAll();
    if(hoveredVecZone) placeToolsFor(hoveredVecZone);
  };

  // ✅ Front button => send click_id unique (anti-rerun loop côté Python)
  const frontBtn=document.getElementById("frontBtn");
  frontBtn.onclick=()=>{
    const snapshot = zones.map(z=>({
      idx: z.idx,
      name: z.name,
      alpha: (z.alphaSigned==null) ? null : +z.alphaSigned,
      targets: Array.isArray(z.selectedNames) ? z.selectedNames.slice() : [],
      locked: !!(z.vecState && z.vecState.locked),
      deleted: !!(z.vecState && z.vecState.deleted),
      custom:  !!(z.vecState && z.vecState.custom),
    }));

    // ✅ backend-only truth: we send current bg transform only on update
    const bg_transform = bgEnabled ? {x:bgX,y:bgY,scale:bgS} : null;

    const click_id = String(Date.now()) + ":" + Math.random().toString(16).slice(2);
    setV({type:"front_button", click_id, snapshot, bg_transform});
  };

  let pan=null;
  let dragVec=null;
  let dragBg=null;

  for(const z of zones){
    for(const c of z.corners){
      c.onpointerdown=(e)=>{
        e.preventDefault(); e.stopPropagation();
        svg.setPointerCapture(e.pointerId);
        z.dragCorner=parseInt(c.dataset.k,10);
        active=z.id; STATE.active=active;
      };
    }

    z.h0.onpointerdown=(e)=>{
      e.preventDefault(); e.stopPropagation();
      const st=z.vecState||{};
      if(st.locked || st.deleted) return;
      svg.setPointerCapture(e.pointerId);
      const w=world(e.clientX,e.clientY);
      dragVec={z, role:"v0", start:w, v0:{x:z.vx0,y:z.vy0}, v1:{x:z.vx1,y:z.vy1}};
      active=z.id; STATE.active=active;
    };
    z.h1.onpointerdown=(e)=>{
      e.preventDefault(); e.stopPropagation();
      const st=z.vecState||{};
      if(st.locked || st.deleted) return;
      svg.setPointerCapture(e.pointerId);
      const w=world(e.clientX,e.clientY);
      dragVec={z, role:"v1", start:w, v0:{x:z.vx0,y:z.vy0}, v1:{x:z.vx1,y:z.vy1}};
      active=z.id; STATE.active=active;
    };
  }

  svg.onpointerdown=(e)=>{
    const cls=e.target?.classList;
    if(cls && cls.contains("corner")) return;
    if(cls && cls.contains("vecHandle")) return;

    if(bgEnabled && e.shiftKey){
      svg.setPointerCapture(e.pointerId);
      const p0=svgPt(e.clientX,e.clientY);
      dragBg={x:p0.x,y:p0.y,bgX,bgY};
      return;
    }

    const zVec = pickVectorHover(e.clientX,e.clientY);
    if(zVec){
      const st=zVec.vecState||{};
      if(st.locked || st.deleted) return;
      svg.setPointerCapture(e.pointerId);
      const w=world(e.clientX,e.clientY);
      dragVec={z:zVec, role:"move", start:w, v0:{x:zVec.vx0,y:zVec.vy0}, v1:{x:zVec.vx1,y:zVec.vy1}};
      active=zVec.id; STATE.active=active;
      return;
    }

    svg.setPointerCapture(e.pointerId);
    const p0=svgPt(e.clientX,e.clientY);
    pan={x:p0.x,y:p0.y,tx,ty};
  };

  svg.onpointermove=(e)=>{
    cancelHide();

    // corner drag
    for(const z of zones){
      if(z.dragCorner!==null){
        const w=world(e.clientX,e.clientY);
        z.quad[z.dragCorner].x=w.x; z.quad[z.dragCorner].y=w.y;
        Mem.save(scope,z.memQuad,z.quad);
        applyTf();
        const zVH=pickVectorHover(e.clientX,e.clientY);
        if(zVH){ hoveredVecZone=zVH; placeToolsFor(zVH); showHud(zVH); }
        else { hoveredVecZone=null; scheduleHide(220); showHud(pickZone(e.clientX,e.clientY)); }
        return;
      }
    }

    // bg drag
    if(dragBg && bgEnabled){
      const p=svgPt(e.clientX,e.clientY);
      bgX = dragBg.bgX + (p.x - dragBg.x);
      bgY = dragBg.bgY + (p.y - dragBg.y);
      if(bgApply) bgApply();
      return;
    }

    // vector drag
    if(dragVec){
      const z=dragVec.z;
      const w=world(e.clientX,e.clientY);
      const dx=w.x-dragVec.start.x;
      const dy=w.y-dragVec.start.y;

      let x0=dragVec.v0.x, y0=dragVec.v0.y, x1=dragVec.v1.x, y1=dragVec.v1.y;

      if(dragVec.role==="v0"){
        const ndx = (x1-x0), ndy=(y1-y0);
        let bx = x0+dx, by=y0+dy;
        const cl = clampToPoly({x:bx,y:by}, z.quad);
        bx=cl.x; by=cl.y;
        x0=bx; y0=by;
        x1=x0+ndx; y1=y0+ndy;
      }
      else if(dragVec.role==="v1"){
        x1=x1+dx; y1=y1+dy;
      }
      else {
        const ndx=(x1-x0), ndy=(y1-y0);
        let bx=x0+dx, by=y0+dy;
        const cl = clampToPoly({x:bx,y:by}, z.quad);
        bx=cl.x; by=cl.y;
        x0=bx; y0=by;
        x1=x0+ndx; y1=y0+ndy;
      }

      const next=Object.assign({}, z.vecState||{}, {x0,y0,x1,y1, custom:true, deleted:false, selSig:z.selSig});
      z.vecState=next;
      Mem.save(scope, z.memVec, next);

      applyTf();

      hoveredVecZone=z;
      placeToolsFor(z);
      showHud(z);
      return;
    }

    // pan
    if(pan){
      const p=svgPt(e.clientX,e.clientY);
      tx=pan.tx+(p.x-pan.x);
      ty=pan.ty+(p.y-pan.y);
      STATE.tx=tx; STATE.ty=ty;
      applyTf();

      const zVH=pickVectorHover(e.clientX,e.clientY);
      if(zVH){ hoveredVecZone=zVH; placeToolsFor(zVH); showHud(zVH); }
      else { hoveredVecZone=null; scheduleHide(220); showHud(pickZone(e.clientX,e.clientY)); }
      return;
    }

    // hover: zone OR vector
    const zZone = pickZone(e.clientX,e.clientY);
    const zVH = pickVectorHover(e.clientX,e.clientY);

    if(zZone){
      showHud(zZone);
    } else if(zVH){
      showHud(zVH);
    } else {
      showHud(null);
    }

    if(zVH){
      hoveredVecZone=zVH;
      placeToolsFor(zVH);
    } else {
      hoveredVecZone=null;
      scheduleHide(220);
    }
  };

  svg.onpointerup=(e)=>{
    try{svg.releasePointerCapture(e.pointerId);}catch(_){}
    pan=null; dragVec=null; dragBg=null;
    for(const z of zones){z.dragCorner=null;}
  };

  const ZOOM_STEP_GLOBAL = 1.018;
  const ZOOM_STEP_BG     = 1.015;

  svg.onwheel=(e)=>{
    e.preventDefault();
    const sp=svgPt(e.clientX,e.clientY);

    // zoom BG
    if(bgEnabled && e.shiftKey){
      const zf=e.deltaY<0?ZOOM_STEP_BG:1/ZOOM_STEP_BG;
      const ns=clamp(bgS*zf,0.02,50);
      const bx=(sp.x-bgX)/bgS, by=(sp.y-bgY)/bgS;
      bgS=ns;
      bgX=sp.x-bx*bgS; bgY=sp.y-by*bgS;
      if(bgApply) bgApply();
      return;
    }

    const zf=e.deltaY<0?ZOOM_STEP_GLOBAL:1/ZOOM_STEP_GLOBAL;
    const ns=clamp(sc*zf,0.15,25);
    const bx=(sp.x-tx)/sc, by=(sp.y-ty)/sc;
    sc=ns;
    tx=sp.x-bx*sc; ty=sp.y-by*sc;
    STATE.tx=tx; STATE.ty=ty; STATE.sc=sc;
    applyTf();

    const zVH=pickVectorHover(e.clientX,e.clientY);
    if(zVH){ hoveredVecZone=zVH; placeToolsFor(zVH); showHud(zVH); }
    else { hoveredVecZone=null; scheduleHide(220); showHud(pickZone(e.clientX,e.clientY)); }
  };

  svg.onpointerleave=()=>{
    showHud(null);
    scheduleHide(220);
  };

  applyTf();
  if(active){
    const z=zones.find(zz=>zz.id===active)||null;
    showHud(z);
  }
}
</script>
</body>
</html>
"""
