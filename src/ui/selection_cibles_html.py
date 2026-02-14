from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = None

_BASE = Path(tempfile.gettempdir()) / "geodashboard_selection_cibles_component_v7_front"


def selection_cibles_component(data: dict, key: str):
    _ensure_component()
    return _COMPONENT(data=data, default=None, key=key)  # type: ignore[name-defined]


def _ensure_component() -> None:
    global _COMPONENT
    if _COMPONENT is not None:
        return

    _BASE.mkdir(parents=True, exist_ok=True)
    (_BASE / "index.html").write_text(_index_html(), encoding="utf-8")
    _COMPONENT = components.declare_component("selection_cibles_component", path=str(_BASE))


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
  .topbar { display:flex; align-items:center; justify-content:flex-start; padding:8px 2px 10px 2px; }
  .btn { appearance:none; border:0; background:#111827; color:#fff; font-weight:800; font-size:13px;
         padding:10px 12px; border-radius:12px; cursor:pointer; box-shadow:0 10px 20px rgba(0,0,0,.10); }
  .btn:active { transform: translateY(1px); }

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

  .pt { fill:#eb3838; stroke:rgba(0,0,0,.15); stroke-width:1; }
  .pt.sel { fill: var(--sel-color, #146EFF); }

  .zonePoly { fill: var(--zone-weak); stroke: var(--zone-stroke); stroke-width:2; stroke-dasharray:6 4; pointer-events:none; }
  .corner { fill: var(--zone-color); stroke: rgba(0,0,0,.14); stroke-width:1; cursor:pointer; }
  .zoneLabel { fill: var(--zone-color); font-weight:900; text-anchor:middle; dominant-baseline:middle; pointer-events:none;
               paint-order:stroke fill; stroke:rgba(0,0,0,.92); stroke-width:3px; stroke-linejoin:round; }

  .vecLine { stroke: var(--zone-color); stroke-linecap: round; }
  .vecHead { fill: var(--zone-color); }
  .vecHandle { fill: rgba(255,255,255,.92); stroke: var(--zone-color); cursor:pointer; }

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
</style>
</head>

<body>
<div class="root">
  <div class="panel" id="panel">
    <div class="topbar">
      <button class="btn" id="frontBtn">Mettre à jour</button>
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
    </div>
  </div>
</div>

<script>
const API=1;
const post=(type,extra)=>window.parent.postMessage(Object.assign({isStreamlitMessage:true,type,apiVersion:API},extra||{}),"*");
post("streamlit:componentReady");
const setH=h=>post("streamlit:setFrameHeight",{height:h});
const setV=v=>post("streamlit:setComponentValue",{value:v});

const STATE = window.__GD_FRONT_STATE__ || (window.__GD_FRONT_STATE__ = {tx:0,ty:0,sc:1,active:null});

addEventListener("message",(e)=>{
  const m=e.data;
  if(!m||m.type!=="streamlit:render")return;
  const D=m.args&&m.args.data;
  if(D) boot(D);
});

function boot(D){
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

  const pcaDir=(pts)=>{
    const n=pts.length; if(n<3)return null;
    let mx=0,my=0; for(const p of pts){mx+=p.x;my+=p.y;} mx/=n; my/=n;
    let sxx=0,syy=0,sxy=0;
    for(const p of pts){const dx=p.x-mx,dy=p.y-my; sxx+=dx*dx; syy+=dy*dy; sxy+=dx*dy;}
    sxx/=n; syy/=n; sxy/=n;
    const tr=sxx+syy, det=sxx*syy-sxy*sxy, disc=Math.max(tr*tr-4*det,0);
    const root=Math.sqrt(disc), l1=0.5*(tr+root);
    let vx,vy;
    if(Math.abs(sxy)>1e-12){vx=l1-syy;vy=sxy;} else {vx=(sxx>=syy)?1:0; vy=(sxx>=syy)?0:1;}
    const vn=n2(vx,vy); if(vn<1e-12)return null;
    return {vx:vx/vn, vy:vy/vn};
  };

  const PTS=pointsIn.map(p=>({name:String(p.name||""),x:+p.px,y:+p.py,z:+p.z,el:null}));
  for(const p of PTS){
    const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
    c.setAttribute("class","pt"); c.setAttribute("r","5.8");
    c.setAttribute("cx",p.x); c.setAttribute("cy",p.y);
    vp.appendChild(c); p.el=c;
  }

  const defQuad=(i)=> i===0 ? [
    {x:W*0.28,y:H*0.22},{x:W*0.55,y:H*0.20},{x:W*0.58,y:H*0.55},{x:W*0.31,y:H*0.58},
  ] : [
    {x:W*0.55,y:H*0.28},{x:W*0.80,y:H*0.26},{x:W*0.82,y:H*0.62},{x:W*0.57,y:H*0.65},
  ];

  const Mem={
    p:"geodashboard.front.v1",
    k:(s,n)=>`${Mem.p}.${s}.${n}`,
    load:(s,n,fb)=>{try{const r=localStorage.getItem(Mem.k(s,n));return r?JSON.parse(r):fb;}catch(_){return fb;}},
    save:(s,n,v)=>{try{localStorage.setItem(Mem.k(s,n),JSON.stringify(v));}catch(_){}}
  };
  const scope="selection_cibles";

  const loadVec=(memVec)=>Mem.load(scope, memVec, null);
  const saveVec=(memVec,v)=>Mem.save(scope, memVec, v);

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
    const vecState=loadVec(memVec);

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
      poly,corners,label,vecG,vLine,vHead,h0,h1,
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

  const computeDefaultVec=(z)=>{
    if(z.selectedPts.length<3) return null;
    let base=z.selectedPts[0];
    for(const p of z.selectedPts){ if(p.z<base.z) base=p; }

    const d=pcaDir(z.selectedPts); if(!d) return null;

    let mx=0,my=0; for(const p of z.selectedPts){mx+=p.x;my+=p.y;} mx/=z.selectedPts.length; my/=z.selectedPts.length;

    let dx=-d.vy, dy=d.vx;
    if(((mx-base.x)*dx+(my-base.y)*dy)<0){dx=-dx;dy=-dy;}
    const L=170; const norm=Math.sqrt(dx*dx+dy*dy); if(norm<1e-9) return null;
    const ux=dx/norm, uy=dy/norm;

    const x0=base.x,y0=base.y, x1=x0+ux*L, y1=y0+uy*L;
    return {x0,y0,x1,y1};
  };

  const setVecWorld=(z, x0,y0,x1,y1)=>{ z.vx0=x0; z.vy0=y0; z.vx1=x1; z.vy1=y1; };

  const maybeResetCustomOnSelChange=(z, prevSig)=>{
    const st=z.vecState || null;
    if(!st) return;
    if(!st.custom) return;
    if(st.deleted) return;
    if(st.locked) return; // ✅ locked => keep frozen
    if(prevSig === z.selSig) return;

    const next = {locked: !!st.locked, deleted: !!st.deleted, custom:false};
    z.vecState = next;
    saveVec(z.memVec, next);
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
      const d=computeDefaultVec(z);
      if(!d){ z.vecG.style.display="none"; z.alphaSigned=null; return; }
      x0=d.x0; y0=d.y0; x1=d.x1; y1=d.y1;
    }
    setVecWorld(z,x0,y0,x1,y1);

    z.vecG.style.display="block";
    z.vLine.setAttribute("x1",x0); z.vLine.setAttribute("y1",y0);
    z.vLine.setAttribute("x2",x1); z.vLine.setAttribute("y2",y1);
    z.vLine.setAttribute("stroke-width",String(4/sc));

    const dx=x1-x0, dy=y1-y0;
    const norm=Math.sqrt(dx*dx+dy*dy); if(norm<1e-9){ z.vecG.style.display="none"; z.alphaSigned=null; return; }
    const ux=dx/norm, uy=dy/norm;

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
      const prevSig = z.selSig || "";
      syncGeom(z);
      sel(z);
      maybeResetCustomOnSelChange(z, prevSig);
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

  // --- HUD (zone) ---
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

  // --- VEC TOOLS stable hover ---
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

  // ✅ FIX: when locking, freeze current vector coords as custom
  toolLock.onclick=(e)=>{
    e.preventDefault(); e.stopPropagation();
    if(!hoveredVecZone) return;

    const z = hoveredVecZone;
    const st = z.vecState || {};
    const nowLocked = !st.locked;

    let next;
    if(nowLocked){
      // freeze current geometry (even if it came from default)
      if(typeof z.vx0 === "number" && typeof z.vx1 === "number"){
        next = Object.assign({}, st, {
          locked: true,
          deleted: false,
          custom: true,
          x0: z.vx0, y0: z.vy0, x1: z.vx1, y1: z.vy1
        });
      } else {
        next = Object.assign({}, st, {locked: true});
      }
    } else {
      next = Object.assign({}, st, {locked: false});
    }

    z.vecState = next;
    saveVec(z.memVec, next);

    placeToolsFor(z);
    showHandlesFor(hoveredZone);
  };

  toolDel.onclick=(e)=>{
    e.preventDefault(); e.stopPropagation();
    if(!hoveredVecZone) return;
    const st=hoveredVecZone.vecState || {};
    const next=Object.assign({}, st, {deleted:true});
    hoveredVecZone.vecState=next;
    saveVec(hoveredVecZone.memVec, next);
    applyTf();
    hoveredVecZone=null;
    scheduleHide(0);
    if(hoveredZone) showHud(hoveredZone);
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

  const applyTf=()=>{
    vp.setAttribute("transform",`translate(${tx} ${ty}) scale(${sc})`);
    updateAll();
    if(hoveredVecZone) placeToolsFor(hoveredVecZone);
  };

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
    setV({type:"front_button", snapshot});
  };

  let pan=null;
  let dragVec=null;

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
    const tgt=e.target;
    if(tgt && (tgt===vecTools || vecTools.contains(tgt))) return;

    cancelHide();

    for(const z of zones){
      if(z.dragCorner!==null){
        const w=world(e.clientX,e.clientY);
        z.quad[z.dragCorner].x=w.x; z.quad[z.dragCorner].y=w.y;
        Mem.save(scope,z.memQuad,z.quad);
        applyTf();

        showHud(pickZone(e.clientX,e.clientY));

        const zVH=pickVectorHover(e.clientX,e.clientY);
        if(zVH){ hoveredVecZone=zVH; placeToolsFor(zVH); }
        else { hoveredVecZone=null; scheduleHide(220); }
        return;
      }
    }

    if(dragVec){
      const z=dragVec.z;
      const w=world(e.clientX,e.clientY);
      const dx=w.x-dragVec.start.x;
      const dy=w.y-dragVec.start.y;

      let x0=dragVec.v0.x, y0=dragVec.v0.y, x1=dragVec.v1.x, y1=dragVec.v1.y;
      if(dragVec.role==="v0"){ x0=x0+dx; y0=y0+dy; }
      else if(dragVec.role==="v1"){ x1=x1+dx; y1=y1+dy; }
      else { x0=x0+dx; y0=y0+dy; x1=x1+dx; y1=y1+dy; }

      const next=Object.assign({}, z.vecState||{}, {x0,y0,x1,y1, custom:true, deleted:false});
      z.vecState=next;
      saveVec(z.memVec, next);

      applyTf();
      hoveredVecZone=z;
      placeToolsFor(z);
      showHud(pickZone(e.clientX,e.clientY));
      return;
    }

    if(pan){
      const p=svgPt(e.clientX,e.clientY);
      tx=pan.tx+(p.x-pan.x);
      ty=pan.ty+(p.y-pan.y);
      STATE.tx=tx; STATE.ty=ty;
      applyTf();

      showHud(pickZone(e.clientX,e.clientY));
      const zVH=pickVectorHover(e.clientX,e.clientY);
      if(zVH){ hoveredVecZone=zVH; placeToolsFor(zVH); }
      else { hoveredVecZone=null; scheduleHide(220); }
      return;
    }

    showHud(pickZone(e.clientX,e.clientY));

    const zVH=pickVectorHover(e.clientX,e.clientY);
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
    pan=null;
    dragVec=null;
    for(const z of zones){z.dragCorner=null;}
  };

  svg.onwheel=(e)=>{
    e.preventDefault();
    const sp=svgPt(e.clientX,e.clientY);
    const zf=e.deltaY<0?1.08:1/1.08;
    const ns=clamp(sc*zf,0.15,25);
    const bx=(sp.x-tx)/sc, by=(sp.y-ty)/sc;
    sc=ns;
    tx=sp.x-bx*sc; ty=sp.y-by*sc;
    STATE.tx=tx; STATE.ty=ty; STATE.sc=sc;
    applyTf();

    showHud(pickZone(e.clientX,e.clientY));
    const zVH=pickVectorHover(e.clientX,e.clientY);
    if(zVH){ hoveredVecZone=zVH; placeToolsFor(zVH); }
    else { hoveredVecZone=null; scheduleHide(220); }
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
