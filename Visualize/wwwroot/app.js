import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- DOM refs ---
const canvas = document.getElementById('canvas');
const animList = document.getElementById('anim-list');
const searchInput = document.getElementById('search');
const captionBar = document.getElementById('caption-bar');
const playBtn = document.getElementById('play-btn');
const timeline = document.getElementById('timeline');
const frameInfo = document.getElementById('frame-info');
const noSelection = document.getElementById('no-selection');

// --- Three.js setup ---
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x1a1a2e);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
camera.position.set(0, 1.5, 4);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 1, 0);
controls.enableDamping = true;

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(5, 10, 5);
scene.add(dirLight);

// Ground grid
const grid = new THREE.GridHelper(10, 20, 0x0f3460, 0x0a1530);
scene.add(grid);

// --- Group colors ---
const groupColors = {
  spine:     0xffffff,
  left_leg:  0x4488ff,
  right_leg: 0xff4444,
  left_arm:  0x44ff88,
  right_arm: 0xffaa22,
};

// --- State ---
let animData = null;       // current loaded animation
let currentFrame = 0;
let playing = false;
let lastFrameTime = 0;
const FPS = 20;            // HumanML3D is 20fps

// Scene objects for the skeleton
let jointMeshes = [];
let boneLine = null;
let jointGroupData = [];
let edgeData = [];

// --- Animation list ---
let allAnimations = [];

async function loadAnimationList() {
  const res = await fetch('/api/animations');
  allAnimations = await res.json();
  renderList(filterAnimations(searchInput.value));
}

function filterAnimations(q) {
  q = (q || '').toLowerCase();
  if (!q) return allAnimations;
  return allAnimations.filter(a =>
    a.id.toLowerCase().includes(q) || (a.caption || '').toLowerCase().includes(q)
  );
}

function renderList(anims) {
  animList.innerHTML = '';
  for (const a of anims) {
    const li = document.createElement('li');
    li.dataset.split = a.split;
    li.dataset.id = a.id;
    li.innerHTML = `<span class="id">${a.id}</span> <span class="split">[${a.split}]</span> <span class="frames">${a.frameCount}f</span>
      <span class="caption" title="${esc(a.caption)}">${esc(a.caption)}</span>`;
    li.addEventListener('click', () => selectAnimation(a.split, a.id, li));
    animList.appendChild(li);
  }
}

async function loadModels() {
  const sel = document.getElementById('gen-model');
  const btn = document.getElementById('gen-btn');
  try {
    const res = await fetch('/api/models');
    const models = await res.json();
    sel.innerHTML = '';
    if (!models || models.length === 0) {
      const o = document.createElement('option');
      o.value = ''; o.textContent = '(no checkpoints)';
      sel.appendChild(o);
      btn.disabled = true;
      return;
    }
    for (const m of models) {
      const o = document.createElement('option');
      o.value = m.name; o.textContent = m.name;
      sel.appendChild(o);
    }
    btn.disabled = false;
  } catch (e) {
    sel.innerHTML = '<option value="">(error)</option>';
    btn.disabled = true;
  }
}

async function handleGenerate() {
  const btn = document.getElementById('gen-btn');
  const model = document.getElementById('gen-model').value;
  const prompt = document.getElementById('gen-prompt').value.trim();
  const frames = parseInt(document.getElementById('gen-frames').value) || 120;
  const guidanceScale = parseFloat(document.getElementById('gen-cfg').value) || 2.5;
  const status = document.getElementById('gen-status');

  status.classList.remove('error');
  if (!model) { status.textContent = 'No model selected'; status.classList.add('error'); return; }
  if (!prompt) { status.textContent = 'Enter a prompt'; status.classList.add('error'); return; }

  btn.disabled = true;
  status.textContent = 'Generating... (this may take a while)';

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, prompt, frames, guidanceScale })
    });
    if (!res.ok) {
      const text = await res.text();
      status.textContent = 'Error: ' + text;
      status.classList.add('error');
      return;
    }
    const info = await res.json();
    status.textContent = `Done: ${info.id}`;

    await loadAnimationList();
    // Auto-select the new item
    const li = [...document.querySelectorAll('#anim-list li')]
      .find(el => el.dataset.split === info.split && el.dataset.id === info.id);
    if (li) {
      li.click();
      li.scrollIntoView({ block: 'nearest' });
    }
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
    status.classList.add('error');
  } finally {
    btn.disabled = false;
  }
}

document.getElementById('gen-btn').addEventListener('click', handleGenerate);
document.getElementById('gen-prompt').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') handleGenerate();
});

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

searchInput.addEventListener('input', () => {
  const q = searchInput.value.toLowerCase();
  const filtered = allAnimations.filter(a =>
    a.id.toLowerCase().includes(q) || a.caption.toLowerCase().includes(q)
  );
  renderList(filtered);
});

// --- Load animation ---
async function selectAnimation(split, id, li) {
  // Highlight
  document.querySelectorAll('#anim-list li').forEach(el => el.classList.remove('active'));
  if (li) li.classList.add('active');

  const res = await fetch(`/api/animation/${split}/${encodeURIComponent(id)}`);
  animData = await res.json();
  edgeData = animData.edges;
  jointGroupData = animData.jointGroup;

  captionBar.textContent = animData.caption || '(no caption)';
  noSelection.style.display = 'none';

  // Reset playback
  currentFrame = 0;
  playing = false;
  playBtn.textContent = 'Play';
  playBtn.disabled = false;
  timeline.disabled = false;
  timeline.max = animData.frameCount - 1;
  timeline.value = 0;
  updateFrameInfo();

  buildSkeleton();
  updateSkeletonFrame(0);
}

// --- Build skeleton meshes ---
function buildSkeleton() {
  // Remove old
  for (const m of jointMeshes) scene.remove(m);
  if (boneLine) scene.remove(boneLine);
  jointMeshes = [];

  const jointGeo = new THREE.SphereGeometry(0.025, 8, 8);

  for (let j = 0; j < animData.joints; j++) {
    const color = groupColors[jointGroupData[j]] || 0xcccccc;
    const mat = new THREE.MeshStandardMaterial({ color });
    const mesh = new THREE.Mesh(jointGeo, mat);
    scene.add(mesh);
    jointMeshes.push(mesh);
  }

  // Bone lines
  const linePositions = new Float32Array(edgeData.length * 2 * 3);
  const lineColors = new Float32Array(edgeData.length * 2 * 3);
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
  lineGeo.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));

  const lineMat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
  boneLine = new THREE.LineSegments(lineGeo, lineMat);
  scene.add(boneLine);
}

// --- Update positions for a given frame ---
function updateSkeletonFrame(frame) {
  if (!animData) return;

  const pos = animData.positions[frame]; // 66 floats: 22 joints x 3

  for (let j = 0; j < animData.joints; j++) {
    const x = pos[j * 3];
    const y = pos[j * 3 + 1];
    const z = pos[j * 3 + 2];
    jointMeshes[j].position.set(x, y, z);
  }

  // Update bone lines
  const linePos = boneLine.geometry.attributes.position.array;
  const lineCol = boneLine.geometry.attributes.color.array;

  for (let i = 0; i < edgeData.length; i++) {
    const [src, dst] = edgeData[i];
    const si = i * 6;

    linePos[si]     = pos[src * 3];
    linePos[si + 1] = pos[src * 3 + 1];
    linePos[si + 2] = pos[src * 3 + 2];
    linePos[si + 3] = pos[dst * 3];
    linePos[si + 4] = pos[dst * 3 + 1];
    linePos[si + 5] = pos[dst * 3 + 2];

    // Color both ends with the child joint's group color
    const color = new THREE.Color(groupColors[jointGroupData[dst]] || 0xcccccc);
    lineCol[si]     = color.r; lineCol[si + 1] = color.g; lineCol[si + 2] = color.b;
    lineCol[si + 3] = color.r; lineCol[si + 4] = color.g; lineCol[si + 5] = color.b;
  }

  boneLine.geometry.attributes.position.needsUpdate = true;
  boneLine.geometry.attributes.color.needsUpdate = true;
}

// --- Playback controls ---
playBtn.addEventListener('click', () => {
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  if (playing) lastFrameTime = performance.now();
});

timeline.addEventListener('input', () => {
  currentFrame = parseInt(timeline.value);
  updateSkeletonFrame(currentFrame);
  updateFrameInfo();
});

function updateFrameInfo() {
  if (!animData) {
    frameInfo.textContent = '0 / 0';
    return;
  }
  frameInfo.textContent = `${currentFrame + 1} / ${animData.frameCount}`;
}

// --- Resize ---
function resize() {
  const vp = document.getElementById('viewport');
  const w = vp.clientWidth;
  const h = vp.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

// --- Render loop ---
function animate(time) {
  requestAnimationFrame(animate);

  if (playing && animData) {
    const elapsed = time - lastFrameTime;
    if (elapsed >= 1000 / FPS) {
      lastFrameTime = time - (elapsed % (1000 / FPS));
      currentFrame++;
      if (currentFrame >= animData.frameCount) {
        currentFrame = 0;
      }
      timeline.value = currentFrame;
      updateSkeletonFrame(currentFrame);
      updateFrameInfo();
    }
  }

  controls.update();
  renderer.render(scene, camera);
}

// --- Init ---
resize();
animate(0);
loadAnimationList();
loadModels();
