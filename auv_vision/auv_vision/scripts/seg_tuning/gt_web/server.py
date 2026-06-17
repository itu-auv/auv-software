#!/usr/bin/env python3
"""ROS-free ground-truth drawing web UI (stdlib http.server only).

A single-page canvas app: grab the live camera frame (or upload an image), paint
the yellow pipe with a brush / polygon, then "Save to dataset". Each save appends
a new ``sample_NNNN/{image.png,gt.png}`` under the dataset directory, so you can
build a multi-frame ground-truth set (near + far + tricky frames). The offline
tuner (seg_tuning/seg_optimize.py) optimises over all samples at once.

This module has no ROS dependency. ``seg_gt_ui_node.py`` wires a live ROS camera
frame into it via the ``frame_provider`` callback; it can also be run stand-alone
for testing with a static frame provider.
"""

import base64
import glob
import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Pipe GT Painter</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family: system-ui, sans-serif; background:#14161a; color:#e8e8e8; }
  header { padding:10px 14px; background:#1d2127; display:flex; gap:8px; flex-wrap:wrap;
           align-items:center; position:sticky; top:0; z-index:10;
           border-bottom:1px solid #2c323c; }
  button, label.btn { background:#2a3138; color:#e8e8e8; border:1px solid #3a424d;
           border-radius:6px; padding:7px 11px; cursor:pointer; font-size:13px; }
  button:hover, label.btn:hover { background:#343d47; }
  button.active { background:#3b6ea5; border-color:#4f86c6; }
  button.go { background:#2e7d4f; border-color:#3a9c63; }
  button.danger { background:#7d2e2e; border-color:#9c3a3a; }
  .sep { width:1px; height:24px; background:#3a424d; margin:0 4px; }
  #stage { padding:14px; }
  #wrap { position:relative; display:inline-block; line-height:0;
          box-shadow:0 0 0 1px #2c323c; }
  canvas { display:block; max-width:100%; height:auto; touch-action:none; }
  #mask { position:absolute; left:0; top:0; opacity:0.45; }
  #count { font-weight:600; color:#7fd6a0; }
  #status { font-size:13px; color:#9fb0c0; margin-left:auto; }
  input[type=range] { vertical-align:middle; }
  #hint { color:#7d8b99; font-size:12px; padding:0 14px 14px; }
</style>
</head>
<body>
<header>
  <button id="grab">Grab live frame</button>
  <label class="btn">Upload<input id="file" type="file" accept="image/*" hidden></label>
  <span class="sep"></span>
  <button id="brush" class="active">Brush</button>
  <button id="eraser">Eraser</button>
  <button id="poly">Polygon</button>
  <button id="polyDone" hidden>Finish polygon</button>
  <span class="sep"></span>
  <label>Size <input id="size" type="range" min="2" max="120" value="34"></label>
  <span class="sep"></span>
  <button id="clear">Clear paint</button>
  <button id="save" class="go">Save to dataset</button>
  <button id="reset" class="danger">Reset dataset</button>
  <span>dataset: <span id="count">0</span> samples</span>
  <span id="status">no image</span>
</header>
<div id="stage">
  <div id="wrap">
    <canvas id="img"></canvas>
    <canvas id="mask"></canvas>
  </div>
</div>
<div id="hint">Workflow: grab/upload a frame, paint the pipe (red), "Save to dataset"
(adds one sample). Repeat for several frames (near + far + tricky), then run the
optimizer. Polygon mode: click points, "Finish polygon" (or double-click) to fill.
Eraser removes paint.</div>
<script>
const imgC = document.getElementById('img'), maskC = document.getElementById('mask');
const ictx = imgC.getContext('2d'), mctx = maskC.getContext('2d');
const statusEl = document.getElementById('status'), countEl = document.getElementById('count');
let tool = 'brush', drawing = false, last = null, hasImage = false;
let polyMode = false, polyPts = [];

async function refreshCount(){
  try{ const r = await fetch('/list'); const j = await r.json(); countEl.textContent = j.count; }catch(e){}
}
refreshCount();

function setTool(t){
  tool = t;
  for (const id of ['brush','eraser','poly']) document.getElementById(id).classList.toggle('active', id===t);
  polyMode = (t === 'poly');
  document.getElementById('polyDone').hidden = !polyMode;
  if(!polyMode) polyPts = [];
}
document.getElementById('brush').onclick = ()=>setTool('brush');
document.getElementById('eraser').onclick = ()=>setTool('eraser');
document.getElementById('poly').onclick = ()=>setTool('poly');

function sizeCanvas(w,h){
  for(const c of [imgC,maskC]){ c.width=w; c.height=h; }
  mctx.clearRect(0,0,w,h);
}
function loadImage(src){
  const im = new Image();
  im.onload = ()=>{
    sizeCanvas(im.naturalWidth, im.naturalHeight);
    ictx.drawImage(im,0,0);
    hasImage = true; polyPts=[];
    statusEl.textContent = im.naturalWidth+'x'+im.naturalHeight;
  };
  im.onerror = ()=>{ statusEl.textContent = 'image load failed'; };
  im.src = src;
}
document.getElementById('grab').onclick = ()=>{
  statusEl.textContent = 'grabbing...';
  loadImage('/frame.jpg?t='+Date.now());
};
document.getElementById('file').onchange = (e)=>{
  const f = e.target.files[0]; if(!f) return;
  loadImage(URL.createObjectURL(f));
};

function pos(e){
  const r = maskC.getBoundingClientRect();
  const sx = maskC.width / r.width, sy = maskC.height / r.height;
  const t = e.touches ? e.touches[0] : e;
  return { x:(t.clientX-r.left)*sx, y:(t.clientY-r.top)*sy };
}
function brushSize(){ return parseInt(document.getElementById('size').value,10); }
function stroke(a,b){
  mctx.globalCompositeOperation = (tool==='eraser') ? 'destination-out' : 'source-over';
  mctx.strokeStyle = 'rgb(255,40,40)';
  mctx.fillStyle = 'rgb(255,40,40)';
  mctx.lineWidth = brushSize(); mctx.lineCap='round'; mctx.lineJoin='round';
  mctx.beginPath(); mctx.moveTo(a.x,a.y); mctx.lineTo(b.x,b.y); mctx.stroke();
  mctx.beginPath(); mctx.arc(b.x,b.y,brushSize()/2,0,7); mctx.fill();
}
function fillPoly(){
  if(polyPts.length < 3){ polyPts=[]; return; }
  mctx.globalCompositeOperation='source-over';
  mctx.fillStyle='rgb(255,40,40)';
  mctx.beginPath(); mctx.moveTo(polyPts[0].x,polyPts[0].y);
  for(let i=1;i<polyPts.length;i++) mctx.lineTo(polyPts[i].x,polyPts[i].y);
  mctx.closePath(); mctx.fill();
  polyPts=[];
}
document.getElementById('polyDone').onclick = fillPoly;

maskC.addEventListener('pointerdown',(e)=>{
  if(!hasImage) return; e.preventDefault();
  if(polyMode){ polyPts.push(pos(e)); statusEl.textContent='polygon pts: '+polyPts.length; return; }
  drawing=true; last=pos(e); stroke(last,last);
});
maskC.addEventListener('pointermove',(e)=>{
  if(!drawing||polyMode) return; e.preventDefault();
  const p=pos(e); stroke(last,p); last=p;
});
window.addEventListener('pointerup',()=>{ drawing=false; });
maskC.addEventListener('dblclick',()=>{ if(polyMode) fillPoly(); });

document.getElementById('clear').onclick = ()=>{
  mctx.clearRect(0,0,maskC.width,maskC.height); polyPts=[];
};

function binaryMaskURL(){
  const w=maskC.width,h=maskC.height;
  const src=mctx.getImageData(0,0,w,h).data;
  const out=document.createElement('canvas'); out.width=w; out.height=h;
  const octx=out.getContext('2d'); const od=octx.createImageData(w,h);
  for(let i=0;i<w*h;i++){
    const v = src[i*4+3] > 10 ? 255 : 0;
    od.data[i*4]=v; od.data[i*4+1]=v; od.data[i*4+2]=v; od.data[i*4+3]=255;
  }
  octx.putImageData(od,0,0);
  return out.toDataURL('image/png');
}
function maskPainted(){
  const d=mctx.getImageData(0,0,maskC.width,maskC.height).data;
  for(let i=3;i<d.length;i+=4) if(d[i]>10) return true;
  return false;
}
document.getElementById('save').onclick = async ()=>{
  if(!hasImage){ statusEl.textContent='grab/upload an image first'; return; }
  if(!maskPainted()){ statusEl.textContent='paint the pipe before saving'; return; }
  statusEl.textContent='saving...';
  const body = { image: imgC.toDataURL('image/png'), mask: binaryMaskURL() };
  try{
    const r = await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j = await r.json();
    if(r.ok){
      countEl.textContent = j.count;
      statusEl.textContent = 'saved '+j.sample+' (total '+j.count+'). Load the next frame.';
      mctx.clearRect(0,0,maskC.width,maskC.height); hasImage=false; ictx.clearRect(0,0,imgC.width,imgC.height);
    } else { statusEl.textContent = 'save failed: '+(j.error||r.status); }
  }catch(err){ statusEl.textContent='save error: '+err; }
};
document.getElementById('reset').onclick = async ()=>{
  if(!confirm('Delete ALL saved samples in the dataset?')) return;
  const r = await fetch('/clear',{method:'POST'}); const j = await r.json();
  countEl.textContent = j.count; statusEl.textContent='dataset cleared';
};
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    frame_provider = None      # callable() -> jpeg bytes or None
    save_dir = "/tmp/seg_session"
    on_save = None             # optional callback(sample_dir, image_path, mask_path)

    def log_message(self, *args):  # quiet by default
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE)
        elif path == "/frame.jpg":
            jpeg = self.frame_provider() if self.frame_provider else None
            if not jpeg:
                self._send(503, "no frame yet", "text/plain")
            else:
                self._send(200, jpeg, "image/jpeg")
        elif path == "/list":
            self._json(200, {"count": _sample_count(self.save_dir)})
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/save":
            self._handle_save()
        elif path == "/clear":
            _clear_dataset(self.save_dir)
            self._json(200, {"ok": True, "count": 0})
        else:
            self._send(404, "not found", "text/plain")

    def _handle_save(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            sample_dir, idx = _next_sample_dir(self.save_dir)
            os.makedirs(sample_dir, exist_ok=True)
            img_path = os.path.join(sample_dir, "image.png")
            mask_path = os.path.join(sample_dir, "gt.png")
            _write_data_url(data["image"], img_path)
            _write_data_url(data["mask"], mask_path)
            if self.on_save:
                try:
                    self.on_save(sample_dir, img_path, mask_path)
                except Exception:  # noqa: BLE001
                    pass
            self._json(200, {
                "ok": True, "sample": os.path.basename(sample_dir),
                "count": _sample_count(self.save_dir), "dir": sample_dir})
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": str(e)})


def _sample_dirs(root):
    return sorted(d for d in glob.glob(os.path.join(root, "sample_*"))
                  if os.path.isdir(d))


def _sample_count(root):
    return len(_sample_dirs(root))


def _next_sample_dir(root):
    existing = _sample_dirs(root)
    idx = 1
    if existing:
        last = os.path.basename(existing[-1]).split("_")[-1]
        try:
            idx = int(last) + 1
        except ValueError:
            idx = len(existing) + 1
    return os.path.join(root, "sample_%04d" % idx), idx


def _clear_dataset(root):
    for d in _sample_dirs(root):
        shutil.rmtree(d, ignore_errors=True)


def _write_data_url(data_url, path):
    b64 = data_url.split(",", 1)[1]
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def make_server(host, port, frame_provider, save_dir, on_save=None):
    """Build a ThreadingHTTPServer serving the GT painter.

    frame_provider: callable returning the latest camera frame as JPEG bytes
                    (or None if no frame yet).
    save_dir:       dataset root; each save adds sample_NNNN/{image,gt}.png.
    """
    os.makedirs(save_dir, exist_ok=True)
    handler = type("BoundHandler", (_Handler,), {
        "frame_provider": staticmethod(frame_provider) if frame_provider else None,
        "save_dir": save_dir,
        "on_save": staticmethod(on_save) if on_save else None,
    })
    return ThreadingHTTPServer((host, port), handler)


def serve_forever_in_thread(server):
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t
