// ══════════════════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════════════════
const state = {
  files: [],          // File objects
  imageUrls: [],      // object URLs for display
  cropData: [],       // [{tl,tr,br,bl}] per image (normalised 0-1)
  croppedCanvases: [], // Canvas elements after crop applied
  currentCropIdx: 0,
  ocrResults: [],
};

// ══════════════════════════════════════════════════════
//  PAGE NAVIGATION
// ══════════════════════════════════════════════════════

// Click logo → về trang Upload
document.querySelectorAll('.logo').forEach(logo => {
  logo.style.cursor = 'pointer';
  logo.addEventListener('click', () => goTo('page-upload'));
});

// Nút quay lại trang Crop → Upload
document.getElementById('btn-back-to-upload').addEventListener('click', () => goTo('page-upload'));

// Nút quay lại trang Process → Crop
document.getElementById('btn-back-to-crop').addEventListener('click', () => {
  goTo('page-crop');
  initCropPage();
});

/**
 * Đơn giản là nhảy đến page có id được chọn
 * @param {int} pageId - pageId của page muốn nhảy tới
 * @returns {void}
 */
function goTo(pageId) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active')); //? làm cho mọi page không active nữa
  document.getElementById(pageId).classList.add('active'); //? làm cho page mong muốn active
  document.querySelectorAll('.step').forEach(s => {
    s.classList.remove('active', 'done');
  });
  const order = ['page-upload','page-crop','page-process']; //? Hiện tại thì chỉ có 3 page, và đây là list ra id của 3 page
  const idx = order.indexOf(pageId); //? lấy ra idx của page muốn nhảy tới

  //? mỗi page sẽ có 1 navigation bar, với mỗi nav gồm 3 step, ví dụ như nhảy đến page 2 thì nó phải làm cho step-1 done và step-2 active
  order.forEach((id, i) => {
    const stepEl = document.getElementById('step-' + (i+1));
    if (!stepEl) return;
    if (i < idx)  stepEl.classList.add('done');
    if (i === idx) stepEl.classList.add('active');
  });
}

// ══════════════════════════════════════════════════════
//  PAGE 1 — UPLOAD
// ══════════════════════════════════════════════════════
const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const previewStrip = document.getElementById('preview-strip');
const btnToCrop   = document.getElementById('btn-to-crop');
const uploadCount = document.getElementById('upload-count');

//@ dropZone
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  addFiles([...e.dataTransfer.files]);
});

fileInput.addEventListener('change', () => addFiles([...fileInput.files]));

function addFiles(files) {
  files.filter(f => f.type.startsWith('image/')).forEach(f => {
    if (state.files.find(x => x.name === f.name && x.size === f.size)) return;
    state.files.push(f);
    state.imageUrls.push(URL.createObjectURL(f));
  });
  renderUploadPreviews();
}

function renderUploadPreviews() {
  previewStrip.innerHTML = '';
  state.files.forEach((f, i) => {
    const div = document.createElement('div');
    div.className = 'preview-thumb';
    div.innerHTML = `<img src="${state.imageUrls[i]}" /><button class="remove-btn" data-i="${i}">✕</button>`;
    previewStrip.appendChild(div);
  });
  previewStrip.querySelectorAll('.remove-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const i = +btn.dataset.i;
      URL.revokeObjectURL(state.imageUrls[i]);
      state.files.splice(i, 1);
      state.imageUrls.splice(i, 1);
      renderUploadPreviews();
    });
  });
  const n = state.files.length;
  uploadCount.textContent = n ? `${n} ảnh đã chọn` : '';
  btnToCrop.disabled = n === 0;
}

btnToCrop.addEventListener('click', async () =>  {
  try{
    const formData = new FormData();
    for(let i = 0; i < state.files.length; i++) {
      formData.append('images', state.files[i]);
    }
  
    const response = await fetch('http://localhost:8000/detect-corners-batch',
      {
        method: "POST",
        body: formData,
      }
    );
    const data = await response.json()
    state.cropData = data
  }
  catch (error){
    // init crop data with default full-image corners
    state.cropData = state.files.map(() => ({
      tl: {x:0.05, y:0.05},
      tr: {x:0.95, y:0.05},
      br: {x:0.95, y:0.95},
      bl: {x:0.05, y:0.95},
    }));
  }
  console.log(state.cropData)
  state.currentCropIdx = 0;
  state.croppedCanvases = new Array(state.files.length).fill(null);
  goTo('page-crop');
  initCropPage();
});

// ══════════════════════════════════════════════════════
//  PAGE 2 — CROP
// ══════════════════════════════════════════════════════
const filmstrip       = document.getElementById('filmstrip');
const cropCanvas      = document.getElementById('crop-canvas');
const ctx             = cropCanvas.getContext('2d');
const btnPrevCrop     = document.getElementById('btn-prev-crop');
const btnNextCrop     = document.getElementById('btn-next-crop');
const btnResetCrop    = document.getElementById('btn-reset-crop');
const btnDoneCrop     = document.getElementById('btn-done-crop');
const imgCounter      = document.getElementById('img-counter');
const coordEls        = ['coord-tl','coord-tr','coord-br','coord-bl'].map(id => document.getElementById(id));

let cropImg = new Image();
let dragging = null; // index of corner being dragged (0-3)
let corners  = [];   // [{x,y}] in canvas pixel space
const HANDLE_R = 14;

function initCropPage() {
  buildFilmstrip();
  loadCropImage(state.currentCropIdx);
}

function buildFilmstrip() {
  filmstrip.innerHTML = '';
  state.files.forEach((f, i) => {
    const div = document.createElement('div');
    div.className = 'film-item' + (i === state.currentCropIdx ? ' active' : '');
    div.innerHTML = `<img src="${state.imageUrls[i]}" /><span class="film-num">${i+1}</span><div class="film-done">✓</div>`;
    div.addEventListener('click', () => { saveCropData(); loadCropImage(i); });
    filmstrip.appendChild(div);
  });
}

function updateFilmstrip() {
  filmstrip.querySelectorAll('.film-item').forEach((el, i) => {
    el.classList.toggle('active', i === state.currentCropIdx);
    el.classList.toggle('done', !!state.croppedCanvases[i]);
  });
}

function loadCropImage(idx) {
  state.currentCropIdx = idx;
  cropImg = new Image();
  cropImg.onload = () => {
    // fit canvas to container
    const wrap = document.querySelector('.crop-canvas-wrap');
    const maxW = wrap.clientWidth  - 48;
    const maxH = wrap.clientHeight - 48;
    const scale = Math.min(maxW / cropImg.width, maxH / cropImg.height, 1);
    cropCanvas.width  = cropImg.width  * scale;
    cropCanvas.height = cropImg.height * scale;

    // restore corners (normalised → pixel)
    const d = state.cropData[idx];
    corners = [d.tl, d.tr, d.br, d.bl].map(p => ({
      x: p.x * cropCanvas.width,
      y: p.y * cropCanvas.height,
    }));

    drawCrop();
    updateImgCounter();
    updateFilmstrip();
    updateCoords();
  };
  cropImg.src = state.imageUrls[idx];
}

function drawCrop() {
  ctx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
  ctx.drawImage(cropImg, 0, 0, cropCanvas.width, cropCanvas.height);

  // dark overlay outside polygon
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, cropCanvas.width, cropCanvas.height);
  ctx.moveTo(corners[0].x, corners[0].y);
  corners.forEach(c => ctx.lineTo(c.x, c.y));
  ctx.closePath();
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fill('evenodd');
  ctx.restore();

  // polygon border
  ctx.beginPath();
  ctx.moveTo(corners[0].x, corners[0].y);
  corners.forEach(c => ctx.lineTo(c.x, c.y));
  ctx.closePath();
  ctx.strokeStyle = '#e8ff47';
  ctx.lineWidth = 2;
  ctx.stroke();

  // edge midpoint lines (grid helper)
  ctx.setLineDash([4,4]);
  ctx.strokeStyle = 'rgba(232,255,71,0.3)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const a = corners[i], b = corners[(i+1)%4];
    const mx = (a.x+b.x)/2, my = (a.y+b.y)/2;
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(mx,my); ctx.stroke();
  }
  ctx.setLineDash([]);

  // corner handles
  const labels = ['TL','TR','BR','BL'];
  corners.forEach((c, i) => {
    // outer ring
    ctx.beginPath();
    ctx.arc(c.x, c.y, HANDLE_R, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(232,255,71,0.15)';
    ctx.fill();
    ctx.strokeStyle = '#e8ff47';
    ctx.lineWidth = 2;
    ctx.stroke();

    // inner dot
    ctx.beginPath();
    ctx.arc(c.x, c.y, 5, 0, Math.PI*2);
    ctx.fillStyle = '#e8ff47';
    ctx.fill();

    // label
    ctx.fillStyle = '#000';
    ctx.font = 'bold 7px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(labels[i], c.x, c.y);
  });
}

// ── Canvas mouse / touch events ──
function getCanvasPos(e) {
  const rect = cropCanvas.getBoundingClientRect();
  const scaleX = cropCanvas.width  / rect.width;
  const scaleY = cropCanvas.height / rect.height;
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top)  * scaleY,
  };
}

function hitCorner(pos) {
  for (let i = 0; i < corners.length; i++) {
    const dx = pos.x - corners[i].x;
    const dy = pos.y - corners[i].y;
    if (Math.sqrt(dx*dx + dy*dy) <= HANDLE_R * 1.5) return i;
  }
  return -1;
}

cropCanvas.addEventListener('mousedown',  onDown);
cropCanvas.addEventListener('touchstart', onDown, {passive:false});

function onDown(e) {
  e.preventDefault();
  const pos = getCanvasPos(e);
  const hit = hitCorner(pos);
  if (hit >= 0) {
    dragging = hit;
    cropCanvas.style.cursor = 'grabbing';
  }
}

window.addEventListener('mousemove',  onMove);
window.addEventListener('touchmove',  onMove, {passive:false});

function onMove(e) {
  if (dragging === null) return;
  e.preventDefault();
  const pos = getCanvasPos(e);
  corners[dragging] = {
    x: Math.max(0, Math.min(cropCanvas.width,  pos.x)),
    y: Math.max(0, Math.min(cropCanvas.height, pos.y)),
  };
  drawCrop();
  updateCoords();
}

window.addEventListener('mouseup',  onUp);
window.addEventListener('touchend', onUp);

function onUp() {
  if (dragging !== null) {
    dragging = null;
    cropCanvas.style.cursor = 'default';
  }
}

// ── Coord display ──
function updateCoords() {
  const labels = ['TL','TR','BR','BL'];
  corners.forEach((c, i) => {
    coordEls[i].querySelector('span').textContent =
      `${Math.round(c.x / cropCanvas.width * 100)}%, ${Math.round(c.y / cropCanvas.height * 100)}%`;
  });
}

function updateImgCounter() {
  imgCounter.textContent = `${state.currentCropIdx + 1} / ${state.files.length}`;
  btnPrevCrop.disabled = state.currentCropIdx === 0;
  btnNextCrop.disabled = state.currentCropIdx === state.files.length - 1;
}

// ── Save current crop data (normalised) ──
function saveCropData() {
  if (!cropCanvas.width) return;
  state.cropData[state.currentCropIdx] = {
    tl: { x: corners[0].x / cropCanvas.width, y: corners[0].y / cropCanvas.height },
    tr: { x: corners[1].x / cropCanvas.width, y: corners[1].y / cropCanvas.height },
    br: { x: corners[2].x / cropCanvas.width, y: corners[2].y / cropCanvas.height },
    bl: { x: corners[3].x / cropCanvas.width, y: corners[3].y / cropCanvas.height },
  };
  // render cropped canvas for this image
  state.croppedCanvases[state.currentCropIdx] = renderCroppedCanvas(state.currentCropIdx);
}

// Perspective-correct crop via canvas transform
function renderCroppedCanvas(idx) {
  const img = cropImg; // already loaded for current idx
  const d = state.cropData[idx];

  // output size from bounding box
  const pts = [d.tl, d.tr, d.br, d.bl];
  const xs = pts.map(p => p.x * img.width);
  const ys = pts.map(p => p.y * img.height);
  const outW = Math.max(
    Math.hypot(xs[1]-xs[0], ys[1]-ys[0]),
    Math.hypot(xs[2]-xs[3], ys[2]-ys[3])
  );
  const outH = Math.max(
    Math.hypot(xs[3]-xs[0], ys[3]-ys[0]),
    Math.hypot(xs[2]-xs[1], ys[2]-ys[1])
  );

  const out = document.createElement('canvas');
  out.width  = Math.round(outW);
  out.height = Math.round(outH);
  const octx = out.getContext('2d');

  // Simple affine approximation (good enough for docs)
  octx.save();
  octx.beginPath();
  octx.moveTo(0, 0); octx.lineTo(outW, 0);
  octx.lineTo(outW, outH); octx.lineTo(0, outH);
  octx.closePath(); octx.clip();

  // We draw the full image scaled, then clip to polygon region
  // For a proper perspective warp we'd need WebGL; affine is sufficient here
  const sx = xs[0], sy = ys[0];
  const scaleX = outW / (xs[1] - xs[0]);
  const scaleY = outH / (ys[3] - ys[0]);
  octx.transform(scaleX, 0, 0, scaleY, -sx * scaleX, -sy * scaleY);
  octx.drawImage(img, 0, 0, img.width, img.height);
  octx.restore();

  return out;
}

btnPrevCrop.addEventListener('click', () => {
  if (state.currentCropIdx <= 0) return;
  saveCropData();
  loadCropImage(state.currentCropIdx - 1);
});

btnNextCrop.addEventListener('click', () => {
  if (state.currentCropIdx >= state.files.length - 1) return;
  saveCropData();
  loadCropImage(state.currentCropIdx + 1);
});

btnResetCrop.addEventListener('click', () => {
  corners = [
    {x: cropCanvas.width*0.05, y: cropCanvas.height*0.05},
    {x: cropCanvas.width*0.95, y: cropCanvas.height*0.05},
    {x: cropCanvas.width*0.95, y: cropCanvas.height*0.95},
    {x: cropCanvas.width*0.05, y: cropCanvas.height*0.95},
  ];
  drawCrop(); updateCoords();
});

btnDoneCrop.addEventListener('click', () => {
  saveCropData();
  // ensure all images have a cropped canvas
  // for images not visited, use full image
  state.files.forEach((_, i) => {
    if (!state.croppedCanvases[i]) {
      // load synchronously won't work; mark to use original
      state.croppedCanvases[i] = null;
    }
  });
  goTo('page-process');
  initProcessPage();
});

// ══════════════════════════════════════════════════════
//  PAGE 3 — PROCESS
// ══════════════════════════════════════════════════════
const croppedGrid  = document.getElementById('cropped-grid');
const btnRunOcr    = document.getElementById('btn-run-ocr');
const btnExport    = document.getElementById('btn-export');
const btnExportTxt = document.getElementById('btn-export-txt');
const btnCopy      = document.getElementById('btn-copy');
const btnClearTxt  = document.getElementById('btn-clear-text');
const resultText   = document.getElementById('result-text');
const logWrap      = document.getElementById('log-wrap');
const progressWrap = document.getElementById('progress-wrap');
const progressBar  = document.getElementById('progress-bar');
const resultBadge  = document.getElementById('result-badge');

function initProcessPage() {
  croppedGrid.innerHTML = '';
  state.ocrResults = [];
  resultText.value = '';
  logWrap.style.display = 'none';
  progressWrap.style.display = 'none';
  progressBar.style.width = '0%';
  resultBadge.textContent = 'CHƯA XỬ LÝ';
  resultBadge.className = 'panel-badge';
  [btnExport, btnExportTxt, btnCopy, btnClearTxt].forEach(b => b.disabled = true);

  state.files.forEach((f, i) => {
    const div = document.createElement('div');
    div.className = 'cropped-item';
    const canvas = state.croppedCanvases[i];
    if (canvas) {
      const img = document.createElement('img');
      img.src = canvas.toDataURL('image/jpeg', 0.9);
      div.appendChild(img);
    } else {
      const img = document.createElement('img');
      img.src = state.imageUrls[i];
      div.appendChild(img);
    }
    const statusDot = document.createElement('div');
    statusDot.className = 'item-status';
    statusDot.id = `item-status-${i}`;
    div.appendChild(statusDot);
    croppedGrid.appendChild(div);
  });
}

function log(msg, type = 'info') {
  logWrap.style.display = 'block';
  const line = document.createElement('div');
  line.className = `log-line ${type}`;
  const tag = type === 'ok' ? '[OK]' : type === 'err' ? '[ERR]' : '[INF]';
  line.innerHTML = `<span class="log-tag">${tag}</span><span>${msg}</span>`;
  logWrap.appendChild(line);
  logWrap.scrollTop = logWrap.scrollHeight;
}

btnRunOcr.addEventListener('click', async () => {
  if (state.files.length === 0) return;
  btnRunOcr.disabled = true;
  btnRunOcr.classList.add('loading');
  progressWrap.style.display = 'block';
  progressBar.style.width = '0%';
  logWrap.innerHTML = '';
  resultText.value = '';
  state.ocrResults = [];
  resultBadge.textContent = 'ĐANG XỬ LÝ...';
  resultBadge.className = 'panel-badge';

  log(`Bắt đầu xử lý ${state.files.length} ảnh...`);

  for (let i = 0; i < state.files.length; i++) {
    const statusEl = document.getElementById(`item-status-${i}`);
    if (statusEl) statusEl.className = 'item-status processing';
    log(`Đang xử lý ảnh ${i + 1}...`);

    try {
      // ── TODO: Thay bằng OCR engine thực tế ──
      // const canvas = state.croppedCanvases[i] || await imgToCanvas(state.imageUrls[i]);
      // const result = await yourOcrEngine(canvas, getOcrLang());
      const result = await simulateOcr(state.files[i], i);
      state.ocrResults.push(result);
      if (statusEl) statusEl.className = 'item-status done';
      log(`Ảnh ${i + 1} hoàn thành (${result.length} ký tự)`, 'ok');
    } catch (err) {
      if (statusEl) statusEl.className = 'item-status error';
      log(`Ảnh ${i + 1} lỗi: ${err.message}`, 'err');
      state.ocrResults.push('');
    }

    progressBar.style.width = `${Math.round(((i+1)/state.files.length)*100)}%`;
  }

  const combined = state.ocrResults.filter(Boolean).join('\n\n--- Trang tiếp theo ---\n\n');
  resultText.value = combined;
  resultBadge.textContent = 'CÓ KẾT QUẢ';
  resultBadge.className = 'panel-badge ok';
  log(`Hoàn tất! ${state.ocrResults.filter(Boolean).length}/${state.files.length} ảnh thành công.`, 'ok');

  btnRunOcr.disabled = false;
  btnRunOcr.classList.remove('loading');
  const hasText = !!combined;
  [btnExport, btnExportTxt, btnCopy, btnClearTxt].forEach(b => b.disabled = !hasText);
});

// ── Simulate OCR — replace with your engine ──
function simulateOcr(file, idx) {
  return new Promise(resolve => {
    setTimeout(() => {
      resolve(
        `[Kết quả OCR — ảnh ${idx + 1}: ${file.name}]\n\n` +
        `Văn bản được nhận dạng từ vùng đã crop. ` +
        `Hãy tích hợp engine OCR (Tesseract.js / API) vào hàm simulateOcr() trong main.js.`
      );
    }, 500 + Math.random() * 700);
  });
}

function getOcrLang() {
  const el = document.getElementById('ocr-lang');
  return el ? el.value : 'vie';
}

// ── Export ──
btnExport.addEventListener('click', () => {
  const text = resultText.value; if (!text) return;
  const font    = document.getElementById('word-font').value;
  const size    = document.getElementById('word-size').value;
  const ml      = document.getElementById('word-margin-left').value;
  const mr      = document.getElementById('word-margin-right').value;
  const keepLay = document.getElementById('keep-layout').checked;
  const filename = (document.getElementById('word-filename').value.trim() || 'ket_qua_ocr');

  const paragraphs = keepLay
    ? text.split('\n').map(l => `<p>${escHtml(l) || '&nbsp;'}</p>`).join('')
    : `<p>${escHtml(text).replace(/\n/g, '<br/>')}</p>`;

  const wordHtml = `<html xmlns:o='urn:schemas-microsoft-com:office:office'
    xmlns:w='urn:schemas-microsoft-com:office:word'
    xmlns='http://www.w3.org/TR/REC-html40'>
    <head><meta charset='utf-8'/>
    <style>body{font-family:'${font}',serif;font-size:${size}pt;
    margin:2cm ${mr}cm 2cm ${ml}cm;line-height:1.6}p{margin:0 0 6pt}</style>
    </head><body>${paragraphs}</body></html>`;

  download(new Blob(['\ufeff'+wordHtml],{type:'application/msword;charset=utf-8'}), filename+'.doc');
  log(`Đã xuất: ${filename}.doc`, 'ok');
});

btnExportTxt.addEventListener('click', () => {
  const text = resultText.value; if (!text) return;
  const filename = (document.getElementById('word-filename').value.trim() || 'ket_qua_ocr');
  download(new Blob([text],{type:'text/plain;charset=utf-8'}), filename+'.txt');
  log(`Đã xuất: ${filename}.txt`, 'ok');
});

btnCopy.addEventListener('click', () => {
  navigator.clipboard.writeText(resultText.value).then(() => {
    btnCopy.textContent = '✓ ĐÃ CHÉP';
    setTimeout(() => { btnCopy.textContent = '⎘ SAO CHÉP'; }, 1500);
  });
});

btnClearTxt.addEventListener('click', () => {
  resultText.value = '';
  [btnExport, btnExportTxt, btnCopy].forEach(b => b.disabled = true);
});

resultText.addEventListener('input', () => {
  const has = !!resultText.value;
  [btnExport, btnExportTxt, btnCopy, btnClearTxt].forEach(b => b.disabled = !has);
});

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name; a.click();
  URL.revokeObjectURL(a.href);
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}