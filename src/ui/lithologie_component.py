from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = None

# ✅ bump version pour forcer reload front
_BASE = Path(tempfile.gettempdir()) / "geodashboard_lithologie_component_v3"


def lithologie_component(data: dict, key: str):
    """
    Front returns:
      None OR
      { "type":"change", "rows":[{"sol":"SOL 1","nom":"...","couleur":"#RRGGBB"}, ...] }
    """
    _ensure_component()
    return _COMPONENT(data=data, default=None, key=key)  # type: ignore[name-defined]


def _ensure_component() -> None:
    global _COMPONENT
    if _COMPONENT is not None:
        return

    _BASE.mkdir(parents=True, exist_ok=True)
    (_BASE / "index.html").write_text(_index_html(), encoding="utf-8")
    _COMPONENT = components.declare_component("lithologie_component", path=str(_BASE))


def _index_html() -> str:
    return r"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Lithologie</title>
<style>
  :root{
    --bg:#ffffff;
    --text:rgba(0,0,0,0.88);
    --line:rgba(0,0,0,0.14);
    --shadow:0 6px 18px rgba(0,0,0,0.12);
    --radius:14px;
    --cell-h:38px;
    --font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  html,body{margin:0;padding:0;background:transparent;font-family:var(--font);color:var(--text);}
  .wrap{width:100%;}
  .grid{border:1px solid var(--line);border-radius:var(--radius);overflow:hidden;background:var(--bg);}
  table{border-collapse:separate;border-spacing:0;width:100%;table-layout:fixed;}
  thead th{
    background:rgba(0,0,0,0.03);
    font-weight:800;font-size:13px;
    padding:0 12px;height:var(--cell-h);
    border-bottom:1px solid var(--line);
    text-align:left;
  }
  tbody td{
    border-bottom:1px solid var(--line);
    height:var(--cell-h);
    padding:0 10px;font-size:13px;vertical-align:middle;
  }
  tbody tr:last-child td{border-bottom:none;}
  .solcell{
    width:110px;color:rgba(0,0,0,0.75);
    font-weight:800;background:rgba(0,0,0,0.02);
  }
  .nameInput{
    width:100%;height:30px;
    border:1px solid rgba(0,0,0,0.14);
    border-radius:10px;
    padding:0 10px;outline:none;
    font-size:13px;background:#fff;
  }
  .nameInput:focus{
    border-color:rgba(0,0,0,0.25);
    box-shadow:0 0 0 3px rgba(0,0,0,0.06);
  }
  .colorBtn{
    display:inline-flex;align-items:center;gap:10px;
    border:1px solid rgba(0,0,0,0.14);
    border-radius:12px;padding:6px 10px;background:#fff;
    cursor:pointer;user-select:none;
  }
  .swatch{
    width:18px;height:18px;border-radius:6px;
    border:1px solid rgba(0,0,0,0.20);
    background:#000;
  }
  .caret{color:rgba(0,0,0,0.55);font-size:12px;}

  .popover{
    position:fixed;z-index:999999;
    width:260px;border-radius:14px;
    background:#fff;box-shadow:var(--shadow);
    border:1px solid rgba(0,0,0,0.12);
    padding:12px;display:none;
  }
  .popTitle{font-size:12px;color:rgba(0,0,0,0.65);margin-bottom:10px;font-weight:800;}
  .palette{display:grid;grid-template-columns:repeat(10,1fr);gap:8px;}
  .pcol{
    width:20px;height:20px;border-radius:7px;
    border:1px solid rgba(0,0,0,0.18);
    cursor:pointer;
  }
  .pcol:hover{transform:translateY(-1px);}
  .custom{
    margin-top:10px;display:flex;align-items:center;justify-content:space-between;gap:10px;
    padding-top:10px;border-top:1px solid rgba(0,0,0,0.08);
  }
  .custom label{font-size:12px;color:rgba(0,0,0,0.65);font-weight:800;}
  input[type="color"]{width:46px;height:30px;border:none;background:transparent;padding:0;}
</style>
</head>
<body>
<div class="wrap">
  <div class="grid">
    <table>
      <thead>
        <tr>
          <th style="width:110px;"></th>
          <th>Nom</th>
          <th style="width:170px;">Couleur</th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<div id="popover" class="popover">
  <div class="popTitle">Choisir une couleur</div>
  <div id="palette" class="palette"></div>
  <div class="custom">
    <label>Custom</label>
    <input id="customPicker" type="color" value="#146EFF" />
  </div>
</div>

<script>
const API=1;
const post=(type,extra)=>window.parent.postMessage(Object.assign({isStreamlitMessage:true,type,apiVersion:API},extra||{}),"*");
const setH=(h)=>post("streamlit:setFrameHeight",{height:h});
const setV=(v)=>post("streamlit:setComponentValue",{value:v});

const clampHex=(h)=>{
  if(!h) return "#000000";
  h=String(h).trim();
  return /^#[0-9A-Fa-f]{6}$/.test(h) ? h.toUpperCase() : "#000000";
};

let state={palette:[],rows:[]};
const tbody=document.getElementById("tbody");
const pop=document.getElementById("popover");
const pal=document.getElementById("palette");
const customPicker=document.getElementById("customPicker");
let activeRowIndex=null;

// ✅ buffer anti-race: si render arrive tôt, on le garde
let pendingData=null;
let hasBooted=false;

const closePopover=()=>{pop.style.display="none"; activeRowIndex=null;};

const openPopover=(x,y,rowIndex)=>{
  activeRowIndex=rowIndex;
  const w=260, h=170;
  let left=x+10, top=y+10;
  const vw=window.innerWidth, vh=window.innerHeight;
  if(left+w>vw-10) left=vw-w-10;
  if(top+h>vh-10) top=vh-h-10;
  pop.style.left=left+"px";
  pop.style.top=top+"px";
  pop.style.display="block";
  customPicker.value=clampHex(state.rows[rowIndex]?.couleur || "#146EFF");
};

const buildPalette=()=>{
  pal.innerHTML="";
  (state.palette||[]).forEach((hx)=>{
    const d=document.createElement("div");
    d.className="pcol";
    d.style.background=clampHex(hx);
    d.addEventListener("click",()=>{
      if(activeRowIndex===null) return;
      state.rows[activeRowIndex].couleur=clampHex(hx);
      renderRows();
      closePopover();
      setV({type:"change", rows: state.rows});
    });
    pal.appendChild(d);
  });
};

customPicker.addEventListener("input",(e)=>{
  if(activeRowIndex===null) return;
  const hx=clampHex(e.target.value);
  state.rows[activeRowIndex].couleur=hx;
  renderRows();
  setV({type:"change", rows: state.rows});
});

document.addEventListener("click",(e)=>{
  const within = pop.contains(e.target);
  const isColorBtn = e.target.closest && e.target.closest(".colorBtn");
  if(!within && !isColorBtn) closePopover();
});

const renderRows=()=>{
  tbody.innerHTML="";
  state.rows.forEach((r,idx)=>{
    const tr=document.createElement("tr");

    const tdSol=document.createElement("td");
    tdSol.className="solcell";
    tdSol.textContent=r.sol||"";
    tr.appendChild(tdSol);

    const tdNom=document.createElement("td");
    const inp=document.createElement("input");
    inp.className="nameInput";
    inp.type="text";
    inp.value=r.nom||"";
    inp.addEventListener("input",()=>{
      state.rows[idx].nom=inp.value;
      setV({type:"change", rows: state.rows});
    });
    tdNom.appendChild(inp);
    tr.appendChild(tdNom);

    const tdCol=document.createElement("td");
    const btn=document.createElement("div");
    btn.className="colorBtn";

    const sw=document.createElement("div");
    sw.className="swatch";
    sw.style.background=clampHex(r.couleur||"#000000");
    btn.appendChild(sw);

    const car=document.createElement("div");
    car.className="caret";
    car.textContent="▾";
    btn.appendChild(car);

    btn.addEventListener("click",(ev)=>{
      ev.stopPropagation();
      openPopover(ev.clientX, ev.clientY, idx);
    });

    tdCol.appendChild(btn);
    tr.appendChild(tdCol);

    tbody.appendChild(tr);
  });

  setH(document.documentElement.scrollHeight);
};

function boot(D){
  if(!D) return;
  state.palette = Array.isArray(D.palette) ? D.palette : [];
  state.rows = Array.isArray(D.rows) ? D.rows : [];
  state.rows = state.rows.map((r,i)=>({
    sol: r.sol || `SOL ${i+1}`,
    nom: r.nom || "",
    couleur: clampHex(r.couleur || (state.palette[i % Math.max(state.palette.length,1)] || "#146EFF")),
  }));
  buildPalette();
  renderRows();
  hasBooted=true;
}

// ✅ listener AVANT componentReady => plus de tableau vide
addEventListener("message",(e)=>{
  const m=e.data;
  if(!m || m.type!=="streamlit:render") return;
  const D=m.args && m.args.data;
  // si jamais ça arrive avant que tout soit prêt, buffer
  if(!D){ return; }
  if(!document.body){ pendingData=D; return; }
  boot(D);
});

// si un render est arrivé trop tôt
window.addEventListener("load", ()=>{
  if(pendingData && !hasBooted) boot(pendingData);
  setH(document.documentElement.scrollHeight);
});

// maintenant seulement on dit "ready"
post("streamlit:componentReady");
setH(document.documentElement.scrollHeight);
</script>
</body>
</html>
"""
