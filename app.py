import streamlit as st

from data_utils import load_data, normalize_set_piece_df, ensure_columns, apply_flip_y
from ui_theme import (
    inject_styles,
    render_header,
    render_kpi_card,
    render_placeholder,
    THEMES,
    FONT_FAMILIES,
    HEATMAP_STYLES,
    build_chart_style,
)
from set_piece_charts import (
    CHART_REQUIREMENTS,
    CHART_BUILDERS,
    save_report_pdf,
    fig_to_png_bytes,
)

st.set_page_config(
    page_title="Set Piece Analysis App",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
render_header(
    title="⚽ Set Piece Analysis App",
    subtitle="Upload CSV / Excel → Set Piece Reports → Styled Visual Analysis",
)


import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE TACTICAL BOARD (mouse drag players + curved arrows)
# ─────────────────────────────────────────────────────────────────────────────
def render_interactive_tactical_board():
    st.markdown("## 🧠 Interactive Tactical Board")
    st.caption("Drag players with the mouse. Add arrows, then drag the arrow start/end/control handle to change length and curve.")
    html = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family: Inter, Arial, sans-serif; background:#0b1220; color:#e5e7eb; }
  .wrap { display:grid; grid-template-columns: 280px 1fr; gap:14px; padding:14px; }
  .panel { background:#111827; border:1px solid #263244; border-radius:16px; padding:14px; box-shadow:0 10px 28px rgba(0,0,0,.25); }
  .panel h3 { margin:0 0 12px; font-size:16px; }
  .row { display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-bottom:8px; }
  label { display:block; font-size:12px; color:#9ca3af; margin:6px 0 4px; }
  input, select, button { width:100%; border-radius:10px; border:1px solid #374151; background:#0f172a; color:#e5e7eb; padding:9px; }
  input[type=color] { padding:2px; height:38px; }
  button { cursor:pointer; font-weight:700; }
  button:hover { background:#1f2937; }
  button.active { background:#2563eb; border-color:#60a5fa; }
  .small { font-size:12px; color:#9ca3af; line-height:1.45; }
  .canvasBox { background:#111827; border:1px solid #263244; border-radius:16px; padding:12px; overflow:auto; }
  canvas { background:#f8fafc; border-radius:10px; display:block; margin:auto; cursor:default; }
  .toolbar { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
  .toolbar button { width:auto; min-width:95px; }
  .hint { margin-top:8px; color:#9ca3af; font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="panel">
    <h3>Controls</h3>
    <div class="toolbar">
      <button id="selectBtn" class="active">Move / Edit</button>
      <button id="addPlayerBtn">Add Player</button>
      <button id="addArrowBtn">Add Arrow</button>
      <button id="deleteBtn">Delete Selected</button>
    </div>
    <div class="row"><div><label>Pitch</label><select id="orientation"><option value="horizontal">Horizontal</option><option value="vertical">Vertical</option></select></div><div><label>Pitch bg</label><input id="pitchBg" type="color" value="#ffffff" /></div></div>
    <div class="row"><div><label>Line color</label><input id="lineColor" type="color" value="#111111" /></div><div><label>Player size</label><input id="playerSize" type="number" min="10" max="38" value="18" /></div></div>
    <hr style="border-color:#263244;margin:14px 0;">
    <h3>New / Selected Player</h3>
    <div class="row"><div><label>Number</label><input id="pNum" value="8" /></div><div><label>Name</label><input id="pName" value="Player" /></div></div>
    <div class="row"><div><label>Color</label><input id="pColor" type="color" value="#ff0000" /></div><div><label>Text</label><input id="pText" type="color" value="#ffffff" /></div></div>
    <h3>Arrow Style</h3>
    <div class="row"><div><label>Color</label><input id="aColor" type="color" value="#d60000" /></div><div><label>Width</label><input id="aWidth" type="number" min="1" max="12" value="4" /></div></div>
    <div class="row"><div><label>Style</label><select id="aStyle"><option value="solid">Solid</option><option value="dashed">Dashed</option><option value="dotted">Dotted</option></select></div><div><label>Arrow head</label><input id="aHead" type="number" min="6" max="30" value="16" /></div></div>
    <button id="downloadBtn">Download PNG</button><button id="clearBtn" style="margin-top:8px;">Clear Board</button>
    <p class="small"><b>How to use:</b><br>Add Player → click anywhere on the pitch.<br>Add Arrow → click start then click end.<br>Move/Edit → drag player or arrow handles.<br>Arrow middle handle controls the curve.</p>
  </div>
  <div class="canvasBox"><canvas id="board" width="1100" height="720"></canvas><div class="hint" id="hint">Mode: Move / Edit</div></div>
</div>
<script>
const canvas=document.getElementById('board'); const ctx=canvas.getContext('2d'); const $=(id)=>document.getElementById(id);
let mode='select', selected=null, drag=null, arrowStart=null;
let players=[{id:1,x:360,y:215,num:'6',name:'Murillo',color:'#ff0000',text:'#ffffff'},{id:2,x:585,y:310,num:'8',name:'E Anderson',color:'#ff0000',text:'#ffffff'},{id:3,x:705,y:420,num:'10',name:'Gibbs White',color:'#ff0000',text:'#ffffff'},{id:4,x:760,y:215,num:'16',name:'Dominguez',color:'#ff0000',text:'#ffffff'}];
let arrows=[]; let nextId=10;
function setMode(m){mode=m; selected=null; drag=null; arrowStart=null; $('selectBtn').classList.toggle('active',m==='select'); $('addPlayerBtn').classList.toggle('active',m==='player'); $('addArrowBtn').classList.toggle('active',m==='arrow'); $('hint').textContent=m==='select'?'Mode: Move / Edit':(m==='player'?'Mode: Add Player — click pitch':'Mode: Add Arrow — click start then end'); draw();}
$('selectBtn').onclick=()=>setMode('select'); $('addPlayerBtn').onclick=()=>setMode('player'); $('addArrowBtn').onclick=()=>setMode('arrow');
['orientation','pitchBg','lineColor','playerSize','pNum','pName','pColor','pText','aColor','aWidth','aStyle','aHead'].forEach(id=>{$(id).addEventListener('input',()=>{ if(selected&&selected.type==='player'){const p=players.find(x=>x.id===selected.id); if(p){p.num=$('pNum').value; p.name=$('pName').value; p.color=$('pColor').value; p.text=$('pText').value;}} if(selected&&selected.type==='arrow'){const a=arrows.find(x=>x.id===selected.id); if(a){a.color=$('aColor').value; a.width=Number($('aWidth').value); a.style=$('aStyle').value; a.head=Number($('aHead').value);}} resizeCanvas(); draw();});});
$('deleteBtn').onclick=()=>{if(!selected)return; if(selected.type==='player')players=players.filter(p=>p.id!==selected.id); if(selected.type==='arrow')arrows=arrows.filter(a=>a.id!==selected.id); selected=null; draw();};
$('clearBtn').onclick=()=>{if(confirm('Clear all players and arrows?')){players=[]; arrows=[]; selected=null; draw();}};
$('downloadBtn').onclick=()=>{draw(false); const link=document.createElement('a'); link.download='tactical_board.png'; link.href=canvas.toDataURL('image/png'); link.click(); draw(true);};
function resizeCanvas(){if($('orientation').value==='vertical'){canvas.width=760; canvas.height=1080;} else {canvas.width=1100; canvas.height=720;}}
function pitchRect(){return {x:30,y:30,w:canvas.width-60,h:canvas.height-60};}
function drawPitch(){const r=pitchRect(); ctx.fillStyle=$('pitchBg').value; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.strokeStyle=$('lineColor').value; ctx.lineWidth=3; const x=r.x,y=r.y,w=r.w,h=r.h; ctx.strokeRect(x,y,w,h); ctx.beginPath(); ctx.moveTo(x+w/2,y); ctx.lineTo(x+w/2,y+h); ctx.stroke(); ctx.beginPath(); ctx.arc(x+w/2,y+h/2,80,0,Math.PI*2); ctx.stroke(); ctx.beginPath(); ctx.arc(x+w/2,y+h/2,4,0,Math.PI*2); ctx.fillStyle=$('lineColor').value; ctx.fill(); const boxW=w*0.16,boxH=h*0.58,sixW=w*0.055,sixH=h*0.28; ctx.strokeRect(x,y+(h-boxH)/2,boxW,boxH); ctx.strokeRect(x,y+(h-sixH)/2,sixW,sixH); ctx.strokeRect(x+w-boxW,y+(h-boxH)/2,boxW,boxH); ctx.strokeRect(x+w-sixW,y+(h-sixH)/2,sixW,sixH); ctx.fillRect(x+w*0.11-3,y+h/2-3,6,6); ctx.fillRect(x+w-w*0.11-3,y+h/2-3,6,6); ctx.strokeRect(x-10,y+h/2-32,10,64); ctx.strokeRect(x+w,y+h/2-32,10,64);}
function handle(x,y,c){ctx.beginPath(); ctx.arc(x,y,8,0,Math.PI*2); ctx.fillStyle=c; ctx.fill(); ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();}
function drawArrow(a,showHandles=true){ctx.save(); ctx.strokeStyle=a.color; ctx.fillStyle=a.color; ctx.lineWidth=a.width; ctx.lineCap='round'; if(a.style==='dashed')ctx.setLineDash([16,12]); if(a.style==='dotted')ctx.setLineDash([3,12]); ctx.beginPath(); ctx.moveTo(a.x1,a.y1); ctx.quadraticCurveTo(a.cx,a.cy,a.x2,a.y2); ctx.stroke(); ctx.setLineDash([]); const ang=Math.atan2(a.y2-a.cy,a.x2-a.cx); const head=a.head||16; ctx.beginPath(); ctx.moveTo(a.x2,a.y2); ctx.lineTo(a.x2-head*Math.cos(ang-Math.PI/6),a.y2-head*Math.sin(ang-Math.PI/6)); ctx.lineTo(a.x2-head*Math.cos(ang+Math.PI/6),a.y2-head*Math.sin(ang+Math.PI/6)); ctx.closePath(); ctx.fill(); ctx.restore(); if(showHandles&&selected&&selected.type==='arrow'&&selected.id===a.id){handle(a.x1,a.y1,'#22c55e'); handle(a.x2,a.y2,'#ef4444'); handle(a.cx,a.cy,'#2563eb'); ctx.save(); ctx.setLineDash([5,6]); ctx.strokeStyle='#64748b'; ctx.lineWidth=1.5; ctx.beginPath(); ctx.moveTo(a.x1,a.y1); ctx.lineTo(a.cx,a.cy); ctx.lineTo(a.x2,a.y2); ctx.stroke(); ctx.restore();}}
function drawPlayer(p){const size=Number($('playerSize').value)||18; ctx.save(); ctx.beginPath(); ctx.arc(p.x,p.y,size,0,Math.PI*2); ctx.fillStyle=p.color; ctx.fill(); ctx.strokeStyle=(selected&&selected.type==='player'&&selected.id===p.id)?'#facc15':'#ffffff'; ctx.lineWidth=(selected&&selected.type==='player'&&selected.id===p.id)?4:2; ctx.stroke(); ctx.fillStyle=p.text; ctx.font=`bold ${Math.max(11,size-3)}px Arial`; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(p.num,p.x,p.y); if(p.name){ctx.fillStyle='#111827'; ctx.font='18px Arial'; ctx.fillText(p.name,p.x,p.y+size+22);} ctx.restore();}
function draw(showHandles=true){drawPitch(); arrows.forEach(a=>drawArrow(a,showHandles)); players.forEach(drawPlayer); if(arrowStart)handle(arrowStart.x,arrowStart.y,'#facc15');}
function pos(evt){const rect=canvas.getBoundingClientRect(); return {x:(evt.clientX-rect.left)*canvas.width/rect.width,y:(evt.clientY-rect.top)*canvas.height/rect.height};}
function dist(a,b,c,d){return Math.hypot(a-c,b-d);} function hitPlayer(x,y){const size=(Number($('playerSize').value)||18)+6; for(let i=players.length-1;i>=0;i--){if(dist(x,y,players[i].x,players[i].y)<=size)return players[i];} return null;}
function hitArrowHandle(x,y){for(let i=arrows.length-1;i>=0;i--){const a=arrows[i]; if(dist(x,y,a.x1,a.y1)<12)return {a,part:'start'}; if(dist(x,y,a.x2,a.y2)<12)return {a,part:'end'}; if(dist(x,y,a.cx,a.cy)<12)return {a,part:'control'};} return null;}
function hitArrowCurve(x,y){for(let i=arrows.length-1;i>=0;i--){const a=arrows[i]; for(let t=0;t<=1;t+=0.04){const qx=(1-t)*(1-t)*a.x1+2*(1-t)*t*a.cx+t*t*a.x2; const qy=(1-t)*(1-t)*a.y1+2*(1-t)*t*a.cy+t*t*a.y2; if(dist(x,y,qx,qy)<10)return a;}} return null;}
function loadSelectedToControls(){if(!selected)return; if(selected.type==='player'){const p=players.find(x=>x.id===selected.id); if(p){$('pNum').value=p.num; $('pName').value=p.name; $('pColor').value=p.color; $('pText').value=p.text;}} if(selected.type==='arrow'){const a=arrows.find(x=>x.id===selected.id); if(a){$('aColor').value=a.color; $('aWidth').value=a.width; $('aStyle').value=a.style; $('aHead').value=a.head;}}}
canvas.addEventListener('mousedown',(evt)=>{const p=pos(evt); if(mode==='player'){players.push({id:nextId++,x:p.x,y:p.y,num:$('pNum').value||String(players.length+1),name:$('pName').value,color:$('pColor').value,text:$('pText').value}); selected={type:'player',id:players[players.length-1].id}; setMode('select'); loadSelectedToControls(); draw(); return;} if(mode==='arrow'){if(!arrowStart){arrowStart=p; draw(); return;} const mx=(arrowStart.x+p.x)/2,my=(arrowStart.y+p.y)/2; const a={id:nextId++,x1:arrowStart.x,y1:arrowStart.y,x2:p.x,y2:p.y,cx:mx,cy:my-80,color:$('aColor').value,width:Number($('aWidth').value),style:$('aStyle').value,head:Number($('aHead').value)}; arrows.push(a); selected={type:'arrow',id:a.id}; arrowStart=null; setMode('select'); loadSelectedToControls(); draw(); return;} const ph=hitPlayer(p.x,p.y); if(ph){selected={type:'player',id:ph.id}; drag={type:'player',id:ph.id,dx:p.x-ph.x,dy:p.y-ph.y}; loadSelectedToControls(); draw(); return;} const ah=hitArrowHandle(p.x,p.y); if(ah){selected={type:'arrow',id:ah.a.id}; drag={type:'arrowHandle',id:ah.a.id,part:ah.part}; loadSelectedToControls(); draw(); return;} const ac=hitArrowCurve(p.x,p.y); if(ac){selected={type:'arrow',id:ac.id}; drag={type:'arrowMove',id:ac.id,last:p}; loadSelectedToControls(); draw(); return;} selected=null; draw();});
canvas.addEventListener('mousemove',(evt)=>{const p=pos(evt); if(!drag){canvas.style.cursor=hitPlayer(p.x,p.y)||hitArrowHandle(p.x,p.y)||hitArrowCurve(p.x,p.y)?'grab':'default'; return;} canvas.style.cursor='grabbing'; if(drag.type==='player'){const pl=players.find(x=>x.id===drag.id); if(pl){pl.x=p.x-drag.dx; pl.y=p.y-drag.dy;}} if(drag.type==='arrowHandle'){const a=arrows.find(x=>x.id===drag.id); if(a){if(drag.part==='start'){a.x1=p.x; a.y1=p.y;} if(drag.part==='end'){a.x2=p.x; a.y2=p.y;} if(drag.part==='control'){a.cx=p.x; a.cy=p.y;}}} if(drag.type==='arrowMove'){const a=arrows.find(x=>x.id===drag.id); if(a){const dx=p.x-drag.last.x,dy=p.y-drag.last.y; a.x1+=dx; a.y1+=dy; a.x2+=dx; a.y2+=dy; a.cx+=dx; a.cy+=dy; drag.last=p;}} draw();});
window.addEventListener('mouseup',()=>{drag=null; canvas.style.cursor='default';}); resizeCanvas(); draw();
</script>
</body>
</html>
'''
    components.html(html, height=1180, scrolling=True)

page = st.sidebar.selectbox("Page", ["Set Piece Charts", "Interactive Tactical Board"], index=0)
if page == "Interactive Tactical Board":
    render_interactive_tactical_board()
    st.stop()

with st.sidebar:
    st.markdown("## 🎛️ Global Controls")

    theme_name = st.selectbox("Theme preset", list(THEMES.keys()), index=0)
    flip_y = st.checkbox("Flip Y axis", value=False)
    uploaded = st.file_uploader("Upload your set piece file", type=["csv", "xlsx", "xls"])

    st.markdown("---")
    st.markdown("## 🏟️ Pitch Controls")

    pitch_orientation = st.radio(
        "Pitch orientation",
        options=["Horizontal", "Vertical"],
        index=0,
        horizontal=True,
        help="Switch all pitch-based charts between landscape (horizontal) and portrait (vertical) layout.",
    )
    pitch_vertical = pitch_orientation == "Vertical"

    show_thirds = st.checkbox(
        "Show thirds lines",
        value=False,
        help="Overlay dashed lines dividing the pitch into Defensive / Middle / Attacking thirds.",
    )

    st.markdown("---")
    st.markdown("## 🎨 Style Controls")

    base_theme = THEMES[theme_name]
    font_family = st.selectbox("Font family", FONT_FAMILIES, index=0)
    heatmap_cmap = st.selectbox("Heatmap theme", HEATMAP_STYLES, index=0)

    with st.expander("Colors", expanded=False):
        bg = st.color_picker("Figure background", base_theme["bg"])
        panel = st.color_picker("Panel / bar background", base_theme["panel"])
        pitch = st.color_picker("Pitch color", base_theme["pitch"])
        pitch_lines = st.color_picker("Pitch lines", base_theme["pitch_lines"])
        text = st.color_picker("Main text", base_theme["text"])
        muted = st.color_picker("Muted text", base_theme["muted"])
        lines = st.color_picker("Border / axis lines", base_theme["lines"])
        accent = st.color_picker("Primary accent", base_theme["accent"])
        accent_2 = st.color_picker("Secondary accent", base_theme["accent_2"])
        success = st.color_picker("Success", base_theme["success"])
        warning = st.color_picker("Warning", base_theme["warning"])
        danger = st.color_picker("Danger", base_theme["danger"])
        legend_bg = st.color_picker("Legend background", base_theme.get("legend_bg", base_theme["panel"]))
        legend_border = st.color_picker("Legend border", base_theme.get("legend_border", base_theme["lines"]))
        legend_text = st.color_picker("Legend text", base_theme.get("legend_text", base_theme["text"]))

    with st.expander("Typography", expanded=False):
        title_size = st.slider("Title size", 10, 28, 16)
        label_size = st.slider("Axis label size", 8, 22, 11)
        tick_size = st.slider("Tick size", 8, 20, 10)
        legend_size = st.slider("Legend size", 8, 20, 10)

    with st.expander("Markers / Lines / Heat", expanded=False):
        marker_size = st.slider("Marker size", 20, 220, 90)
        marker_edge_width = st.slider("Marker edge width", 0.0, 3.5, 1.2, 0.1)
        line_width = st.slider("Line width", 0.5, 4.0, 1.4, 0.1)
        trajectory_width = st.slider("Trajectory width", 0.5, 5.0, 1.8, 0.1)
        trajectory_alpha = st.slider("Trajectory alpha", 0.10, 1.00, 0.75, 0.05)
        trajectory_headwidth = st.slider("Trajectory arrow head", 2.0, 10.0, 4.5, 0.5)
        alpha = st.slider("Marker alpha", 0.10, 1.00, 0.90, 0.05)
        kde_alpha = st.slider("Heatmap alpha", 0.10, 1.00, 0.72, 0.05)
        grid_alpha = st.slider("Grid alpha", 0.00, 0.50, 0.18, 0.02)

    with st.expander("Layout", expanded=False):
        show_title = st.checkbox("Show chart title", value=True)
        show_legend = st.checkbox("Show legends", value=True)
        show_grid = st.checkbox("Show grid on bar charts", value=True)
        show_ticks = st.checkbox("Show pitch ticks", value=False)
        tight_layout = st.checkbox("Use tight layout", value=True)
        export_dpi = st.slider("Export PNG DPI", 120, 400, 260)

    with st.expander("⚽ Scatter Dot Color", expanded=False):
        st.caption("Color for delivery end scatter points")
        scatter_dot_color = st.color_picker("Scatter dot color", base_theme["accent"])

    with st.expander("🗺️ First Contact Map — Action Colors", expanded=False):
        st.caption("Color per action type on the First Contact Location Map")
        fc_shot_color      = st.color_picker("Shot",       "#FFD400")
        fc_clearance_color = st.color_picker("Clearance",  "#FF4D4D")
        fc_header_color    = st.color_picker("Header",     "#38BDF8")
        fc_threat_color    = st.color_picker("Threat",     "#A78BFA")
        fc_cross_color     = st.color_picker("Cross",      "#22C55E")
        fc_foul_color      = st.color_picker("Foul",       "#F97316")
        fc_other_color     = st.color_picker("Other",      "#94A3B8")
        st.caption("Show / hide action types")
        fc_show_shot      = st.checkbox("Show Shot",       value=True)
        fc_show_clearance = st.checkbox("Show Clearance",  value=True)
        fc_show_header    = st.checkbox("Show Header",     value=True)
        fc_show_threat    = st.checkbox("Show Threat",     value=True)
        fc_show_cross     = st.checkbox("Show Cross",      value=True)
        fc_show_foul      = st.checkbox("Show Foul",       value=True)
        fc_show_other     = st.checkbox("Show Other",      value=True)

    with st.expander("🎯 Arrow Colors (by delivery type)", expanded=False):
        st.caption("Override arrow colors per delivery type")
        arrow_inswing  = st.color_picker("Inswing",  base_theme["accent"])
        arrow_outswing = st.color_picker("Outswing", base_theme["warning"])
        arrow_straight = st.color_picker("Straight", base_theme["accent_2"])
        arrow_driven   = st.color_picker("Driven",   base_theme["success"])
        arrow_short    = st.color_picker("Short",    base_theme["danger"])

    with st.expander("📊 Bar Chart Colors", expanded=False):
        st.caption("Override bar/histogram fill color")
        bar_default = st.color_picker("Default bar color", base_theme["accent"])
        bar_success  = st.color_picker("Success bar color", base_theme["success"])
        bar_danger   = st.color_picker("Danger bar color",  base_theme["danger"])

    with st.expander("👕 Shirt Colors (Taker Stats Table)", expanded=False):
        shirt_body   = st.color_picker("Shirt body",   base_theme["accent"])
        shirt_sleeve = st.color_picker("Shirt sleeves", base_theme["panel"])
        shirt_number = st.color_picker("Number color",  base_theme["bg"])

    st.markdown("---")
    st.markdown("## 📊 Charts")

    all_charts = list(CHART_BUILDERS.keys())

    # ── attacking defaults ───────────────────────────────────────────────────
    attacking_defaults = [
        "Delivery Trajectories - Left Corners",
        "Delivery Trajectories - Right Corners",
        "Attack Free Kick Trajectories",
        "Delivery End Scatter - Left Corner",
        "Delivery End Scatter - Right Corner",
        "Zone Delivery Count Map - Left Corner",
        "Zone Delivery Count Map - Right Corner",
        "Avg Players Per Zone - Left Corner",
        "Avg Players Per Zone - Right Corner",
        "First Contact Location Map",
        "First Contact Players by Shirt Number",
        "Players Who Made First Contact",
        "Players That Lost First Contact",
        "Box Marking Scheme",
        "Set Piece Landing Heatmap",
        "Taker Stats Table",
    ]

    # ── defensive defaults ───────────────────────────────────────────────────
    defensive_defaults = [
        "Defensive Shape Map",
        "Defender vs Attacker Zone Matchup",
        "Clearance Outcome Map",
        "Set Piece Conceded Heatmap",
        "Defensive Success Rate By Zone",
        "First Contact Win Rate Trend",
        "Second Ball Recovery Map",
    ]

    st.markdown("#### ⚔️ Attacking Charts")
    attacking_charts = st.multiselect(
        "Attacking charts",
        [c for c in all_charts if c not in defensive_defaults],
        default=[c for c in attacking_defaults if c in all_charts],
        label_visibility="collapsed",
    )

    st.markdown("#### 🛡️ Defensive Charts")
    st.caption(
        "These charts use: **result** (clearance detection), **first_contact_win**, "
        "**second_ball_win**, **defenders_near_post / defenders_far_post**, and **opponent** columns."
    )
    defensive_charts = st.multiselect(
        "Defensive charts",
        defensive_defaults,
        default=[c for c in defensive_defaults if c in all_charts],
        label_visibility="collapsed",
    )

    selected_charts = attacking_charts + defensive_charts

    with st.expander("Chart requirements", expanded=False):
        for ch in selected_charts:
            st.write(f"**{ch}** → " + ", ".join(CHART_REQUIREMENTS.get(ch, [])))

    generate_clicked = st.button("Generate Set Piece Analysis", use_container_width=True)

style_overrides = {
    "bg": bg,
    "panel": panel,
    "pitch": pitch,
    "pitch_lines": pitch_lines,
    "text": text,
    "muted": muted,
    "lines": lines,
    "accent": accent,
    "accent_2": accent_2,
    "success": success,
    "warning": warning,
    "danger": danger,
    "legend_bg": legend_bg,
    "legend_border": legend_border,
    "legend_text": legend_text,
    "font_family": font_family,
    "title_size": title_size,
    "label_size": label_size,
    "tick_size": tick_size,
    "legend_size": legend_size,
    "marker_size": marker_size,
    "marker_edge_width": marker_edge_width,
    "line_width": line_width,
    "trajectory_width": trajectory_width,
    "trajectory_alpha": trajectory_alpha,
    "trajectory_headwidth": trajectory_headwidth,
    "alpha": alpha,
    "kde_alpha": kde_alpha,
    "grid_alpha": grid_alpha,
    "show_title": show_title,
    "show_legend": show_legend,
    "show_grid": show_grid,
    "show_ticks": show_ticks,
    "tight_layout": tight_layout,
    "export_dpi": export_dpi,
    "heatmap_cmap": heatmap_cmap,
    # ── pitch layout ──────────────────────────────────────────────────────────
    "pitch_vertical": pitch_vertical,
    "show_thirds":    show_thirds,
    # ── scatter dot color ─────────────────────────────────────────────────────
    "scatter_dot_color": scatter_dot_color,
    # ── arrow colours ─────────────────────────────────────────────────────────
    "arrow_colors": {
        "inswing":  arrow_inswing,
        "outswing": arrow_outswing,
        "straight": arrow_straight,
        "driven":   arrow_driven,
        "short":    arrow_short,
    },
    # ── bar colours ───────────────────────────────────────────────────────────
    "bar_colors": {
        "default": bar_default,
        "success": bar_success,
        "danger":  bar_danger,
    },
    # ── shirt colours ─────────────────────────────────────────────────────────
    "shirt_body_color":   shirt_body,
    "shirt_sleeve_color": shirt_sleeve,
    "shirt_number_color": shirt_number,
}
chart_style = build_chart_style(theme_name, style_overrides)

left_col, right_col = st.columns([1.0, 1.75], gap="large")

with left_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Style Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="panel-note">
            Theme: <b>{theme_name}</b><br>
            Orientation: <b>{pitch_orientation}</b><br>
            Thirds: <b>{"On" if show_thirds else "Off"}</b><br>
            Heatmap: <b>{heatmap_cmap}</b><br>
            Font: <b>{font_family}</b><br>
            Marker size: <b>{marker_size}</b><br>
            Line width: <b>{line_width}</b><br>
            Export DPI: <b>{export_dpi}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Defensive column guide ────────────────────────────────────────────────
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🛡️ Defensive Column Guide</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-note">
        The defensive charts use these optional columns:<br><br>
        <b>first_contact_win</b> — yes/no or 1/0<br>
        <b>second_ball_win</b> — yes/no or 1/0<br>
        <b>result</b> — clearance / shot / header…<br>
        <b>outcome</b> — Successful / Unsuccessful<br>
        <b>defenders_near_post</b> — integer count<br>
        <b>defenders_far_post</b> — integer count<br>
        <b>man_marking_in_box</b> — number of man-marking defenders in box<br>
        <b>zonal_marking_in_box</b> — number of zonal defenders in box<br>
        <b>lost_first_contact_player</b> — shirt number of player who lost first contact<br>
        <b>opponent</b> — match label for trend chart<br><br>
        Charts degrade gracefully when columns are missing.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="preview-shell">', unsafe_allow_html=True)
    st.markdown("### 📊 Preview & Downloads")

    if uploaded is None:
        render_placeholder(
            "No file uploaded yet",
            "Upload a CSV / Excel file from the sidebar, then generate the analysis.",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    df = load_data(uploaded)
    df = normalize_set_piece_df(df)
    df = apply_flip_y(df, flip_y=flip_y)

    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_card("Rows", len(df))
    with k2:
        render_kpi_card("Columns", len(df.columns))
    with k3:
        seq_n = int(df["sequence_id"].nunique()) if "sequence_id" in df.columns else 0
        render_kpi_card("Sequences", seq_n)

    with st.expander("Preview data (first 25 rows)", expanded=False):
        st.write("Columns:", list(df.columns))
        st.dataframe(df.head(25), use_container_width=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    filter_cols = st.columns(4)

    with filter_cols[0]:
        if "set_piece_type" in df.columns:
            sp_values = [x for x in df["set_piece_type"].dropna().astype(str).unique().tolist() if x and x != "nan"]
            if sp_values:
                selected_sp = st.multiselect("Set piece types", sp_values, default=sp_values)
                if selected_sp:
                    df = df[df["set_piece_type"].isin(selected_sp)].copy()

        # also support raw 'set_piece' column before normalisation
        elif "set_piece" in df.columns:
            sp_values = [x for x in df["set_piece"].dropna().astype(str).unique().tolist() if x and x != "nan"]
            if sp_values:
                selected_sp = st.multiselect("Set piece types", sp_values, default=sp_values)
                if selected_sp:
                    df = df[df["set_piece"].isin(selected_sp)].copy()

    with filter_cols[1]:
        if "team" in df.columns:
            teams = [x for x in df["team"].dropna().astype(str).unique().tolist() if x and x != "nan"]
            if teams:
                selected_team = st.selectbox("Team", ["all"] + teams, index=0)
                if selected_team != "all":
                    df = df[df["team"] == selected_team].copy()

    with filter_cols[2]:
        if "side" in df.columns:
            sides = [x for x in df["side"].dropna().astype(str).unique().tolist() if x and x != "nan"]
            if sides:
                selected_sides = st.multiselect("Side", sides, default=sides)
                if selected_sides:
                    df = df[df["side"].isin(selected_sides)].copy()

    with filter_cols[3]:
        if "delivery_type" in df.columns:
            dtypes = [x for x in df["delivery_type"].dropna().astype(str).unique().tolist() if x and x != "nan"]
            if dtypes:
                selected_delivery = st.multiselect("Delivery type", dtypes, default=dtypes)
                if selected_delivery:
                    df = df[df["delivery_type"].isin(selected_delivery)].copy()

    # ── Opponent filter (for trend chart) ────────────────────────────────────
    if "opponent" in df.columns:
        opps = [x for x in df["opponent"].dropna().astype(str).unique().tolist() if x and x != "nan"]
        if len(opps) > 1:
            selected_opps = st.multiselect("Opponent (for trend chart)", opps, default=opps)
            if selected_opps:
                df = df[df["opponent"].isin(selected_opps)].copy()

    if not generate_clicked:
        render_placeholder(
            "Ready to generate",
            "Adjust theme, orientation, thirds, heatmap style, and filters from the sidebar, then click Generate.",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # ── Render charts ─────────────────────────────────────────────────────────
    figures = []

    # Section headers
    att_set = set(attacking_charts)
    def_set = set(defensive_charts)
    shown_att_header = False
    shown_def_header = False

    for chart_name in selected_charts:
        # section header
        if chart_name in att_set and not shown_att_header:
            st.markdown("## ⚔️ Attacking Set Piece Charts")
            shown_att_header = True
        if chart_name in def_set and not shown_def_header:
            st.markdown("## 🛡️ Defensive Set Piece Charts")
            shown_def_header = True

        req = CHART_REQUIREMENTS.get(chart_name, [])
        missing = ensure_columns(df, req)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-title">{chart_name}</div>', unsafe_allow_html=True)
        if req:
            st.markdown('<div class="panel-note">Requirements: ' + ", ".join(req) + '</div>', unsafe_allow_html=True)

        if missing:
            st.warning("Missing columns: " + ", ".join(missing))
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        try:
            fig = CHART_BUILDERS[chart_name](
                df.copy(),
                theme_name=theme_name,
                flip_y=False,
                style_overrides=chart_style,
            )
            figures.append(fig)
            st.pyplot(fig, use_container_width=True)

            png_bytes = fig_to_png_bytes(fig, dpi=chart_style["export_dpi"])
            png_name = chart_name.lower().replace(" ", "_").replace("%", "pct") + ".png"

            st.download_button(
                f"⬇️ Download {chart_name} PNG",
                data=png_bytes,
                file_name=png_name,
                mime="image/png",
                key=f"png_{chart_name}",
            )
        except Exception as e:
            st.error(f"Could not render {chart_name}: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    if figures:
        pdf_bytes = save_report_pdf(figures)
        st.download_button(
            "⬇️ Download Full Set Piece Report PDF",
            data=pdf_bytes,
            file_name="set_piece_report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No charts were rendered. Check the required columns shown under each chart.")

    st.markdown("</div>", unsafe_allow_html=True)
