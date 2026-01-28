/**
 * Gotcha! AI ReCaptcha Solver - Main Application
 */

// ============================================
// Mobile Sidebar Toggle
// ============================================
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebar-overlay');

function toggleSidebar() {
    sidebar.classList.toggle('active');
    sidebarOverlay.classList.toggle('active');
    document.body.style.overflow = sidebar.classList.contains('active') ? 'hidden' : '';
}

// Close sidebar on resize to desktop
window.addEventListener('resize', () => {
    if (window.innerWidth >= 768) {
        sidebar.classList.remove('active');
        sidebarOverlay.classList.remove('active');
        document.body.style.overflow = '';
    }
});

// ============================================
// DOM Elements
// ============================================
const terminalContent = document.getElementById('terminal-content');
const clearTerminal = document.getElementById('clear-terminal');
const mainStatus = document.getElementById('main-status');

const v1Container = document.getElementById('v1-container');
const v2Container = document.getElementById('v2-container');
const challengeImg = document.getElementById('challenge-img');
const v1Scanner = document.getElementById('v1-scanner');
const v2Scanner = document.getElementById('v2-scanner');
const v2TargetName = document.getElementById('v2-target-name');
const v2Grid = document.getElementById('v2-grid');
const v1InputGroup = document.getElementById('v1-input-group');

const captchaInput = document.getElementById('captcha-solve-input');
const btnAutoSolve = document.getElementById('btn-auto-solve');
const btnVerify = document.getElementById('btn-verify-solve');
const refreshChallenge = document.getElementById('refresh-challenge');
const solverStatus = document.getElementById('solver-status');

const tabV1 = document.getElementById('tab-v1');
const tabV2 = document.getElementById('tab-v2');

// ============================================
// State Variables
// ============================================
let currentChallengeId = null;
let currentV2Challenge = null;
let selectedV2Indices = new Set();
let currentMode = 'v1';

// Metrics are injected from server-side template
// window.v1Metrics, window.v2Metrics, window.initialStatus

// ============================================
// Sidebar Metrics Update
// ============================================
function updateSidebarMetrics(mode) {
    const m = mode === 'v1' ? window.v1Metrics : window.v2Metrics;
    
    console.log(`Updating metrics for mode: ${mode}`, m);
    
    // Labels
    document.getElementById('metric-accuracy-label').innerText = mode === 'v1' ? 'Word Accuracy' : 'Object Accuracy';
    document.getElementById('metric-char-accuracy-label').innerText = mode === 'v1' ? 'Character Accuracy' : 'Class Accuracy';
    
    // Feed Description
    document.getElementById('feed-description').innerText = mode === 'v1' 
        ? 'Native CRNN+CTC Simulation Feed' 
        : 'Vision-based Object Classification Feed';

    // Values
    document.getElementById('metric-accuracy').innerText = m.accuracy || 'N/A';
    document.getElementById('metric-char-accuracy').innerText = m.char_accuracy || 'N/A';
    document.getElementById('metric-precision').innerText = m.precision || 'N/A';
    document.getElementById('metric-recall').innerText = m.recall || 'N/A';
    document.getElementById('metric-f1-score').innerText = m.f1_score || 'N/A';
    document.getElementById('metric-type').innerText = m.type || 'N/A';
    document.getElementById('metric-loss').innerText = m.loss || 'N/A';
    document.getElementById('metric-loss-value').innerText = m.loss_value || 'N/A';

    // Descriptions
    document.getElementById('metric-accuracy-desc').innerText = mode === 'v1' 
        ? 'Entire 5-character sequence must be perfect. One wrong letter = 0% success.'
        : 'Correct identification of the target object category within the grid.';
    document.getElementById('metric-char-accuracy-desc').innerText = mode === 'v1'
        ? 'Percentage of individual letters correctly recognized across all samples.'
        : 'Accuracy of the underlying classification model for all object classes.';
}

// ============================================
// Terminal Logging
// ============================================
function log(msg, type = 'info') {
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    const p = document.createElement('p');
    const typeColor = type === 'success' ? 'text-accent-green' : type === 'error' ? 'text-red-500' : 'text-primary';
    p.innerHTML = `<span class="text-[#9da6b9]">[${time}]</span> <span class="${typeColor} font-bold">${type.toUpperCase()}:</span> ${msg}`;
    terminalContent.appendChild(p);
    terminalContent.scrollTop = terminalContent.scrollHeight;
}

clearTerminal.onclick = () => {
    terminalContent.innerHTML = '<p class="text-accent-green font-bold">>> Console session cleared</p>';
};

// ============================================
// Challenge Loading
// ============================================
async function loadChallenge() {
    if (currentMode === 'v1') {
        await loadChallengeV1();
    } else {
        await loadChallengeV2();
    }
}

async function loadChallengeV1() {
    log('Requesting new v1 challenge...');
    v1Scanner.style.display = 'block';
    mainStatus.innerText = 'FETCHING';

    try {
        const response = await fetch('/api/challenge');
        const data = await response.json();
        challengeImg.src = data.image;
        currentChallengeId = data.challenge_id;
        captchaInput.value = "";
        solverStatus.innerText = "READY";
        log(`Challenge loaded: ${currentChallengeId}`, 'success');
    } catch (err) {
        log(`Failed to load v1 challenge: ${err.message}`, 'error');
    } finally {
        v1Scanner.style.display = 'none';
        mainStatus.innerText = 'READY';
    }
}

async function loadChallengeV2() {
    log('Requesting new v2 challenge...');
    v2Scanner.style.display = 'block';
    mainStatus.innerText = 'FETCHING';
    selectedV2Indices.clear();

    try {
        const response = await fetch('/api/v2/challenge');
        const data = await response.json();
        currentV2Challenge = data;
        v2TargetName.innerText = data.target;
        renderV2Grid(data.grid);
        solverStatus.innerText = "READY";
        log(`v2 Challenge loaded. Target: ${data.target}`, 'success');
    } catch (err) {
        log(`Failed to load v2 challenge: ${err.message}`, 'error');
    } finally {
        v2Scanner.style.display = 'none';
        mainStatus.innerText = 'READY';
    }
}

// ============================================
// V2 Grid Rendering
// ============================================
function renderV2Grid(grid) {
    const children = Array.from(v2Grid.children);
    children.forEach(c => { if (c.id !== 'v2-scanner') v2Grid.removeChild(c); });

    grid.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = "relative aspect-square bg-background-dark border border-border-dark cursor-pointer overflow-hidden group transition-all duration-300 hover:border-primary active:scale-95 touch-action-manipulation";
        div.innerHTML = `
            <img src="data:image/jpeg;base64,${item.image}" class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105 pointer-events-none" draggable="false">
            <div class="absolute inset-0 bg-primary/30 opacity-0 transition-opacity duration-200 select-overlay pointer-events-none"></div>
            <div class="absolute top-0.5 sm:top-1 right-0.5 sm:right-1 bg-primary text-white rounded-full w-4 h-4 sm:w-5 sm:h-5 flex items-center justify-center opacity-0 transition-all duration-200 transform scale-50 select-check pointer-events-none">
                <span class="material-symbols-outlined text-[10px] sm:text-xs">done</span>
            </div>
        `;
        
        div.addEventListener('click', (e) => {
            e.preventDefault();
            toggleV2Selection(div, index);
        });
        div.addEventListener('touchend', (e) => {
            e.preventDefault();
            toggleV2Selection(div, index);
        }, { passive: false });
        
        v2Grid.appendChild(div);
    });
}

function toggleV2Selection(el, index) {
    const overlay = el.querySelector('.select-overlay');
    const check = el.querySelector('.select-check');

    if (selectedV2Indices.has(index)) {
        selectedV2Indices.delete(index);
        overlay.style.opacity = '0';
        check.style.opacity = '0';
        check.style.transform = 'scale(0.5)';
        el.classList.remove('ring-4', 'ring-primary', 'ring-inset');
    } else {
        selectedV2Indices.add(index);
        overlay.style.opacity = '1';
        check.style.opacity = '1';
        check.style.transform = 'scale(1)';
        el.classList.add('ring-4', 'ring-primary', 'ring-inset');
    }
    log(`Index ${index} ${selectedV2Indices.has(index) ? 'selected' : 'deselected'}`);
}

// ============================================
// AI Auto-Solve
// ============================================
refreshChallenge.onclick = loadChallenge;

btnAutoSolve.onclick = async () => {
    if (currentMode === 'v1') {
        await autoSolveV1();
    } else {
        await autoSolveV2();
    }
};

async function autoSolveV1() {
    if (!currentChallengeId) return;
    btnAutoSolve.disabled = true;
    log('Initializing AI solving unit...', 'info');
    solverStatus.innerText = "ANALYZING...";
    mainStatus.innerText = 'SOLVING';

    try {
        const response = await fetch('/api/solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ challenge_id: currentChallengeId })
        });
        const data = await response.json();
        captchaInput.value = "";
        captchaInput.classList.add('typing-cursor');
        for (let char of data.prediction) {
            captchaInput.value += char;
            await new Promise(r => setTimeout(r, 100));
        }
        captchaInput.classList.remove('typing-cursor');
        log(`AI solve generated: "${data.prediction}"`, 'success');
        solverStatus.innerText = "SOLVE COMPLETE";
    } catch (error) {
        log('AI solving unit failed', 'error');
        solverStatus.innerText = "ERROR";
    } finally {
        btnAutoSolve.disabled = false;
        mainStatus.innerText = 'READY';
    }
}

async function autoSolveV2() {
    if (!currentV2Challenge) return;
    btnAutoSolve.disabled = true;
    log('Initializing Vision AI to scan grid...', 'info');
    solverStatus.innerText = "SCANNING...";
    mainStatus.innerText = 'SOLVING';

    try {
        const response = await fetch('/api/v2/solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target: currentV2Challenge.target,
                grid_ids: currentV2Challenge.grid.map(g => g.id)
            })
        });
        const data = await response.json();

        selectedV2Indices.clear();
        const gridItems = Array.from(v2Grid.children).filter(c => c.id !== 'v2-scanner');

        for (let idx of data.correct_indices) {
            toggleV2Selection(gridItems[idx], idx);
            await new Promise(r => setTimeout(r, 200));
        }

        log(`AI scan complete. Selected ${data.correct_indices.length} matches.`, 'success');
        solverStatus.innerText = "SCAN COMPLETE";
    } catch (error) {
        log('Vision AI failed', 'error');
        solverStatus.innerText = "ERROR";
    } finally {
        btnAutoSolve.disabled = false;
        mainStatus.innerText = 'READY';
    }
}

// ============================================
// Verification
// ============================================
btnVerify.onclick = async () => {
    if (currentMode === 'v1') {
        await verifyV1();
    } else {
        await verifyV2();
    }
};

async function verifyV1() {
    const userInput = captchaInput.value;
    if (!userInput || !currentChallengeId) return;
    log('Initiating validation check...');
    mainStatus.innerText = 'VERIFYING';

    try {
        const response = await fetch('/api/verify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ challenge_id: currentChallengeId, user_input: userInput })
        });
        const data = await response.json();
        handleVerifyResult(data.is_correct, data.true_label);
    } catch (err) {
        log(`Verification failed: ${err.message}`, 'error');
    }
}

async function verifyV2() {
    if (!currentV2Challenge) return;
    log('Verifying grid selection with server...');
    mainStatus.innerText = 'VERIFYING';

    const selectedIds = Array.from(selectedV2Indices).map(idx => currentV2Challenge.grid[idx].id);

    try {
        const response = await fetch('/api/v2/verify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target: currentV2Challenge.target,
                grid_ids: currentV2Challenge.grid.map(g => g.id),
                selected_ids: selectedIds
            })
        });
        const data = await response.json();
        handleVerifyResult(data.is_correct, `Matches: ${data.actual_ids.length}`);
    } catch (err) {
        log(`v2 Verification failed: ${err.message}`, 'error');
    }
}

function handleVerifyResult(isCorrect, infoText) {
    const targetBoxes = [v1Container, v2Grid];

    if (isCorrect) {
        log(`SUCCESS: ${infoText}`, 'success');
        solverStatus.innerText = "PASSED";
        mainStatus.innerText = 'SUCCESS';

        targetBoxes.forEach(box => {
            box.classList.add('border-accent-green', 'ring-2', 'ring-accent-green/20');
            box.classList.remove('border-border-dark');
        });

        setTimeout(() => {
            targetBoxes.forEach(box => {
                box.classList.remove('border-accent-green', 'ring-2', 'ring-accent-green/20');
                box.classList.add('border-border-dark');
            });
            loadChallenge();
        }, 2000);
    } else {
        log(`FAILURE: ${infoText}`, 'error');
        solverStatus.innerText = "FAILED";
        mainStatus.innerText = 'FAILED';

        targetBoxes.forEach(box => {
            box.classList.add('border-red-500', 'ring-2', 'ring-red-500/20');
            box.classList.remove('border-border-dark');
        });

        setTimeout(() => {
            mainStatus.innerText = 'READY';
            targetBoxes.forEach(box => {
                box.classList.remove('border-red-500', 'ring-2', 'ring-red-500/20');
                box.classList.add('border-border-dark');
            });
        }, 1000);
    }
}

// ============================================
// Tab Switching
// ============================================
tabV1.onclick = () => {
    currentMode = 'v1';
    tabV1.classList.add('bg-primary', 'text-white');
    tabV1.classList.remove('text-[#9da6b9]');
    tabV2.classList.remove('bg-primary', 'text-white');
    tabV2.classList.add('text-[#9da6b9]');
    v1Container.classList.remove('hidden');
    v2Container.classList.add('hidden');
    v1InputGroup.classList.remove('hidden');
    log('Switched to v1: Text Captcha');
    updateSidebarMetrics('v1');
    loadChallenge();
};

tabV2.onclick = () => {
    currentMode = 'v2';
    tabV2.classList.add('bg-primary', 'text-white');
    tabV2.classList.remove('text-[#9da6b9]');
    tabV1.classList.remove('bg-primary', 'text-white');
    tabV1.classList.add('text-[#9da6b9]');
    v2Container.classList.remove('hidden');
    v1Container.classList.add('hidden');
    v1InputGroup.classList.add('hidden');
    log('Switched to v2: Image Select');
    updateSidebarMetrics('v2');
    loadChallenge();
};

// ============================================
// Initialization
// ============================================
window.onload = () => {
    updateSidebarMetrics('v1');
    loadChallenge();
};
