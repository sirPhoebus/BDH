import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- CONFIG ---
const N = 1000; // Match n_neurons in Rust
const SCALE = 20;

// --- THREE.JS SETUP ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
camera.position.z = 40;

// --- NEURON GEOMETRY (Anatomical Mapping) ---
const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(N * 3);
const colors = new Float32Array(N * 3);
const sizes = new Float32Array(N);

for (let i = 0; i < N; i++) {
    // Phyllotaxis-inspired spheroid for unified brain
    const phi = Math.acos(-1 + (2 * i) / N);
    const theta = Math.sqrt(N * Math.PI) * phi;

    const r = SCALE * (0.8 + 0.3 * Math.random());
    positions[i * 3] = r * Math.cos(theta) * Math.sin(phi);
    positions[i * 3 + 1] = r * Math.sin(theta) * Math.sin(phi) * 0.7; // Slightly flattened
    positions[i * 3 + 2] = r * Math.cos(phi);

    colors[i * 3] = 0.1;
    colors[i * 3 + 1] = 0.2;
    colors[i * 3 + 2] = 0.4;
    sizes[i] = 1.0;
}

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

// --- 2. TEXTURE (Sphere Particle) ---
function createSphereTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, 'rgba(255,255,255,1)');
    gradient.addColorStop(0.2, 'rgba(255,255,255,0.8)');
    gradient.addColorStop(0.5, 'rgba(255,255,255,0.2)');
    gradient.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 64, 64);

    const tex = new THREE.CanvasTexture(canvas);
    return tex;
}

const material = new THREE.PointsMaterial({
    size: 1.2,
    map: createSphereTexture(),
    vertexColors: true,
    transparent: true,
    opacity: 0.8,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true,
    depthWrite: false
});

const neuralCloud = new THREE.Points(geometry, material);
scene.add(neuralCloud);

// --- LIGHTING ---
const ambientLight = new THREE.AmbientLight(0x404040);
scene.add(ambientLight);

// --- WEBSOCKET CLIENT ---
const socket = new WebSocket('ws://' + window.location.host + '/ws');
const statusElem = document.getElementById('status');
const daBar = document.getElementById('da-bar');
const neBar = document.getElementById('ne-bar');
const achBar = document.getElementById('ach-bar');
const daVal = document.getElementById('da-val');
const neVal = document.getElementById('ne-val');
const achVal = document.getElementById('ach-val');
const narrativeElem = document.getElementById('narrative');

socket.onopen = () => {
    statusElem.innerText = "CONNECTED";
    statusElem.style.color = "#00ff00";
};

socket.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        updateVisualization(data);
    } catch (e) {
        // Handle input_word or other text messages if sent separately
    }
};

socket.onclose = () => {
    statusElem.innerText = "DISCONNECTED";
    statusElem.style.color = "#ff0000";
};

// --- 1. CONNECTOME (Synapses) ---
const lineNum = 3; // Connections per neuron
const connectionIndices = [];
for (let i = 0; i < N; i++) {
    // Find k-nearest neighbors (brute force initialization)
    const neighbors = [];
    for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const dx = positions[i * 3] - positions[j * 3];
        const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
        const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
        const distSq = dx * dx + dy * dy + dz * dz;
        neighbors.push({ index: j, d: distSq });
    }
    neighbors.sort((a, b) => a.d - b.d);
    for (let k = 0; k < lineNum; k++) {
        connectionIndices.push(i, neighbors[k].index);
    }
}

const lineGeometry = new THREE.BufferGeometry();
lineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(connectionIndices.length * 3), 3));
lineGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(connectionIndices.length * 3), 3));

const lineMaterial = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.15,
    blending: THREE.AdditiveBlending
});
const synapses = new THREE.LineSegments(lineGeometry, lineMaterial);
scene.add(synapses);


const originalPositions = positions.slice();
let persistenceFlashes = []; // Array of { indices, time }

// --- 3. REGION LABELS ---
// Removed for unified view (using colors to indicate regions)

function updateVisualization(data) {
    const energies = data.energies;
    const colorAttr = geometry.attributes.color;
    const posAttr = geometry.attributes.position;

    const linePosAttr = lineGeometry.attributes.position;
    const lineColorAttr = lineGeometry.attributes.color;

    const ne = data.norepinephrine;
    const ach = data.acetylcholine;
    const da = data.dopamine;

    // Handle Persistence Event (Memory Save)
    if (data.memory_event) {
        const memoryVec = data.memory_event;
        const activeIndices = [];
        // Find top 30 neurons involved in this memory
        const sorted = memoryVec.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);
        for (let k = 0; k < 30; k++) activeIndices.push(sorted[k].i); // Corrected from sorted[k].index to sorted[k].i
        persistenceFlashes.push({ indices: activeIndices, startTime: Date.now() });
    }

    // Find max energy for dynamic scaling
    let maxE = 0.00001;
    for (let i = 0; i < N; i++) {
        if (energies[i] > maxE) maxE = energies[i];
    }

    const now = Date.now() * 0.005;

    for (let i = 0; i < N; i++) {
        const rawE = energies[i];
        const t = i / N;

        // Boost visibility
        let e = Math.pow(rawE / maxE, 0.3);

        // Check for persistence flashes
        let flashIntensity = 0;
        for (let f = 0; f < persistenceFlashes.length; f++) {
            const flash = persistenceFlashes[f];
            if (flash.indices.includes(i)) {
                const age = Date.now() - flash.startTime;
                if (age < 2000) {
                    flashIntensity = Math.sin(age * 0.002) * (1.0 - age / 2000) * 0.8; // Adjusted sin frequency
                }
            }
        }
        persistenceFlashes = persistenceFlashes.filter(f => Date.now() - f.startTime < 2000);

        // --- 1. NEURON COLOR ---
        const finalE = Math.max(e, flashIntensity);
        if (finalE > 0.05) {
            const intensity = 0.4 + finalE * 0.6;
            if (flashIntensity > 0.05) { // Changed threshold for flash
                // Flash white-gold for persistence
                colorAttr.array[i * 3] = 1.0;
                colorAttr.array[i * 3 + 1] = 0.9;
                colorAttr.array[i * 3 + 2] = 0.5;
            } else if (t < 0.3) {
                colorAttr.array[i * 3] = intensity; colorAttr.array[i * 3 + 1] = intensity; colorAttr.array[i * 3 + 2] = 0.2;
            } else if (t < 0.6) {
                colorAttr.array[i * 3] = 0.2; colorAttr.array[i * 3 + 1] = intensity; colorAttr.array[i * 3 + 2] = intensity;
            } else if (t < 0.85) { // Added back parietal region
                colorAttr.array[i * 3] = intensity; colorAttr.array[i * 3 + 1] = 0.2; colorAttr.array[i * 3 + 2] = intensity;
            } else { // Occipital
                colorAttr.array[i * 3] = intensity; colorAttr.array[i * 3 + 1] = 0.5; colorAttr.array[i * 3 + 2] = 1.0;
            }
        } else {
            colorAttr.array[i * 3] = 0.01;
            colorAttr.array[i * 3 + 1] = 0.02 + ach * 0.1;
            colorAttr.array[i * 3 + 2] = 0.1 + ach * 0.3;
        }

        // --- 2. JITTER ---
        const jitter = ne * 2.5 * finalE;
        posAttr.array[i * 3] = originalPositions[i * 3] + (Math.random() - 0.5) * jitter;
        posAttr.array[i * 3 + 1] = originalPositions[i * 3 + 1] + (Math.random() - 0.5) * jitter;
        posAttr.array[i * 3 + 2] = originalPositions[i * 3 + 2] + (Math.random() - 0.5) * jitter;
    }

    // --- 3. SYNAPSE SYNC & FLOW ---
    for (let j = 0; j < connectionIndices.length; j++) {
        const neuronIdx = connectionIndices[j];

        linePosAttr.array[j * 3] = posAttr.array[neuronIdx * 3];
        linePosAttr.array[j * 3 + 1] = posAttr.array[neuronIdx * 3 + 1];
        linePosAttr.array[j * 3 + 2] = posAttr.array[neuronIdx * 3 + 2];

        // Information Flow Pulse: Use time to animate colors along the line segments
        // Each pair (j, j+1) is a line. j is start, j+1 is end? 
        // Index is flattening connectionIndices. 
        // The connectionIndices array stores pairs of neuron indices (start, end, start, end, ...).
        // So, j is the index in connectionIndices, which points to a neuron index.
        // The actual line segment is between connectionIndices[j] and connectionIndices[j+1] if j is even.
        // For a single point in the line geometry, we are setting its color based on the neuron it represents.
        // To show a pulse *along* the line, we need to consider the two endpoints of a line segment.
        // For simplicity, let's apply the pulse effect to each point based on its index in the flattened array.
        const pulse = Math.sin(now - (j * 0.1)) * 0.5 + 0.5; // Pulse based on time and position in the line array
        const activity = energies[neuronIdx] / maxE;

        lineColorAttr.array[j * 3] = colorAttr.array[neuronIdx * 3] * pulse * (0.2 + activity * 0.8);
        lineColorAttr.array[j * 3 + 1] = colorAttr.array[neuronIdx * 3 + 1] * pulse * (0.2 + activity * 0.8);
        lineColorAttr.array[j * 3 + 2] = colorAttr.array[neuronIdx * 3 + 2] * pulse * (0.2 + activity * 0.8);
    }

    colorAttr.needsUpdate = true;
    posAttr.needsUpdate = true;
    linePosAttr.needsUpdate = true;
    lineColorAttr.needsUpdate = true;

    material.size = 0.4 + (ach * 2.0);
    lineMaterial.opacity = 0.05 + (ach * 0.3);

    daBar.style.width = (da * 100) + '%';
    neBar.style.width = (ne * 100) + '%';
    achBar.style.width = (ach * 100) + '%';

    daVal.innerText = da.toFixed(2);
    neVal.innerText = ne.toFixed(2);
    achVal.innerText = ach.toFixed(2);

    statusElem.innerText = `[${data.input_word}] > ${data.prediction}`;
    statusElem.style.color = ne > 0.6 ? "#ff5555" : (da > 0.6 ? "#ffff00" : "#00ff00");

    narrativeElem.innerText = data.inner_voice || "...";
    if (data.inner_voice.includes("Danger")) narrativeElem.style.color = "#ff8888";
    else if (data.inner_voice.includes("Safety")) narrativeElem.style.color = "#88ff88";
    else if (data.inner_voice.includes("Life") || data.inner_voice.includes("Thought")) narrativeElem.style.color = "#88ccff";
    else narrativeElem.style.color = "#aaa";
}

// --- ANIMATION LOOP ---
function animate() {
    requestAnimationFrame(animate);
    // neuralCloud.rotation.y += 0.001; // STOPPED as requested
    controls.update();
    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();
