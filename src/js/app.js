/**
 * MLLM — Main Application JavaScript
 *
 * Handles all frontend logic: view navigation, calling the Python
 * backend through pywebview's JS bridge (window.pywebview.api),
 * and rendering results.
 *
 * No external JS dependencies. Vanilla JS only.
 */

// ===== State =====
const state = {
    currentView: 'setup',
    selectedRuntime: null,
    runtimeConnected: false,
    modelLoaded: false,
    modelId: null,
    running: false,
};

// ===== Utilities =====

/** Wait for desktop bridge to be available (Tauri 2 or pywebview) */
function waitForBridge() {
    return new Promise((resolve) => {
        // Tauri 2 environment
        if (window.__TAURI_INTERNALS__ || (window.__TAURI__ && window.__TAURI__.core)) {
            const invoke = window.__TAURI__.core.invoke;
            const tauriApi = {
                detect_system: () => invoke('rpc', { method: 'detect_system', params: {} }),
                get_available_runtimes: () => invoke('rpc', { method: 'get_available_runtimes', params: {} }),
                get_strategies: () => invoke('rpc', { method: 'get_strategies', params: {} }),
                select_runtime: (name) => invoke('rpc', { method: 'select_runtime', params: { runtime_name: name } }),
                load_model: (modelId) => invoke('rpc', { method: 'load_model', params: { model_id: modelId } }),
                get_model_info: (modelId) => invoke('rpc', { method: 'get_model_info', params: { model_id: modelId } }),
                setup_agents: (count, strategy) => invoke('rpc', { method: 'setup_agents', params: { count, strategy } }),
                run_prompt: (prompt, strategy) => invoke('rpc', { method: 'run_prompt', params: { prompt, strategy } }),
                run_benchmark: (prompt, strategy) => invoke('rpc', { method: 'run_benchmark', params: { prompt, strategy } }),
                get_benchmark_history: () => invoke('rpc', { method: 'get_benchmark_history', params: {} }),
                shutdown: () => invoke('rpc', { method: 'shutdown', params: {} }),
            };
            resolve(tauriApi);
            return;
        }

        // Pywebview environment fallback
        if (window.pywebview && window.pywebview.api) {
            resolve(window.pywebview.api);
            return;
        }
        window.addEventListener('pywebviewready', () => {
            resolve(window.pywebview.api);
        });
    });
}

/** Show a status message in a container */
function showStatus(elementId, message, type = 'loading') {
    const el = document.getElementById(elementId);
    el.style.display = 'block';
    el.className = `status-message ${type}`;
    if (type === 'loading') {
        el.innerHTML = `<span class="spinner"></span>${escapeHtml(message)}`;
    } else {
        el.textContent = message;
    }
}

function hideStatus(elementId) {
    document.getElementById(elementId).style.display = 'none';
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ===== Navigation =====

function switchView(viewName) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

    document.getElementById(`view-${viewName}`).classList.add('active');
    document.querySelector(`.nav-btn[data-view="${viewName}"]`).classList.add('active');
    state.currentView = viewName;
}

document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
});

// ===== Setup View =====

async function initSetup(api) {
    // Detect hardware
    const sysInfo = await api.detect_system();
    renderSystemInfo(sysInfo);

    // Get available runtimes
    const runtimes = await api.get_available_runtimes();
    renderRuntimes(runtimes);
}

function renderSystemInfo(info) {
    const panel = document.getElementById('system-info');
    const rows = [
        ['OS', info.os, 'ok'],
        ['Architecture', info.architecture, null],
        ['CPU', info.cpu, null],
        ['Cores / Threads', `${info.cpu_cores} / ${info.cpu_threads}`, null],
        ['RAM', `${info.ram_available} free / ${info.ram_total}`, null],
        ['Disk Free', info.disk_free, null],
        ['GPU', info.gpu_name, info.gpu_metal || info.gpu_cuda ? 'ok' : 'warn'],
        ['VRAM', info.gpu_vram, null],
        ['Docker', info.docker_available ? info.docker_version : 'Not found', info.docker_available ? 'ok' : 'warn'],
        ['Ollama', info.ollama_installed ? info.ollama_path : 'Not installed', info.ollama_installed ? 'ok' : 'err'],
    ];

    panel.innerHTML = rows.map(([label, value, cls]) => `
        <div class="info-row">
            <span class="info-label">${escapeHtml(label)}</span>
            <span class="info-value${cls ? ` ${cls}` : ''}">${escapeHtml(String(value || 'Unknown'))}</span>
        </div>
    `).join('');
}

function renderRuntimes(runtimes) {
    const container = document.getElementById('runtime-list');
    container.innerHTML = runtimes.map(rt => `
        <label class="runtime-option${rt.available ? '' : ' disabled'}">
            <input type="radio" name="runtime" value="${escapeHtml(rt.name)}"
                   ${rt.available ? '' : 'disabled'}>
            <span class="runtime-name">${escapeHtml(rt.name.charAt(0).toUpperCase() + rt.name.slice(1))}</span>
            <span class="runtime-note">${escapeHtml(rt.note)}</span>
        </label>
    `).join('');

    // Show connect button
    const actionsEl = document.getElementById('setup-actions');
    actionsEl.style.display = 'block';

    // Enable connect button when a runtime is selected
    container.addEventListener('change', (e) => {
        state.selectedRuntime = e.target.value;
        document.getElementById('btn-connect').disabled = false;
        // Visual selection
        container.querySelectorAll('.runtime-option').forEach(opt => opt.classList.remove('selected'));
        e.target.closest('.runtime-option').classList.add('selected');
    });
}

// ===== Connect to Runtime =====

async function connectRuntime(api) {
    if (!state.selectedRuntime) return;

    const btn = document.getElementById('btn-connect');
    btn.disabled = true;
    showStatus('setup-status', `Connecting to ${state.selectedRuntime}...`);
    updateRuntimeIndicator('loading', `Connecting...`);

    try {
        const result = await api.select_runtime(state.selectedRuntime);

        if (result.success) {
            state.runtimeConnected = true;
            showStatus('setup-status', `Connected to ${state.selectedRuntime}`, 'success');
            updateRuntimeIndicator('online', `${state.selectedRuntime} ready`);

            // Show GPU info
            const caps = result.capabilities;
            if (caps.gpu) {
                showStatus('setup-status',
                    `Connected to ${state.selectedRuntime} — GPU: ${caps.gpu_vendor || 'detected'}, concurrent sessions: ${caps.concurrent_sessions ? 'yes' : 'no'}`,
                    'success'
                );
            }

            // Auto-navigate to model view
            setTimeout(() => switchView('model'), 800);
        } else {
            showStatus('setup-status', result.error, 'error');
            updateRuntimeIndicator('error', 'Connection failed');
            btn.disabled = false;
        }
    } catch (e) {
        showStatus('setup-status', `Error: ${e.message || e}`, 'error');
        updateRuntimeIndicator('error', 'Error');
        btn.disabled = false;
    }
}

function updateRuntimeIndicator(status, text) {
    const indicator = document.getElementById('runtime-indicator');
    indicator.className = `status-indicator ${status}`;
    document.getElementById('runtime-status-text').textContent = text;
}

// ===== Model View =====

async function loadModel(api) {
    const input = document.getElementById('model-input');
    const modelId = input.value.trim();
    if (!modelId) return;

    const btn = document.getElementById('btn-load-model');
    btn.disabled = true;
    showStatus('model-status', `Loading ${modelId}... This may take a while if the model needs to be downloaded.`);
    updateRuntimeIndicator('loading', `Loading ${modelId}...`);

    try {
        const result = await api.load_model(modelId);

        if (result.success) {
            state.modelLoaded = true;
            state.modelId = modelId;
            showStatus('model-status', result.message, 'success');
            updateRuntimeIndicator('online', `${state.selectedRuntime} · ${modelId}`);
            document.getElementById('model-badge').textContent = modelId;
            document.getElementById('btn-send').disabled = false;

            // Try to get model info
            try {
                const info = await api.get_model_info(modelId);
                renderModelInfo(info);
            } catch (e) {
                // Non-fatal
            }

            setTimeout(() => switchView('chat'), 800);
        } else {
            showStatus('model-status', result.error, 'error');
            updateRuntimeIndicator('error', 'Load failed');
            btn.disabled = false;
        }
    } catch (e) {
        showStatus('model-status', `Error: ${e.message || e}`, 'error');
        updateRuntimeIndicator('error', 'Error');
        btn.disabled = false;
    }
}

function renderModelInfo(info) {
    const panel = document.getElementById('model-info-panel');
    if (!info || info.error) return;

    const rows = [
        ['Model', info.model_id],
        ['Architecture', info.architecture],
        ['Parameters', info.parameter_count ? formatNumber(info.parameter_count) : 'Unknown'],
        ['Quantization', info.quantization],
        ['Format', info.format],
        ['Context Length', info.context_length || 'Unknown'],
        ['Supported', info.supported ? '✓ Yes' : '✗ No'],
    ].filter(([, v]) => v != null && v !== '' && v !== 'Unknown');

    if (rows.length === 0) return;

    panel.style.display = 'block';
    panel.innerHTML = rows.map(([label, value]) => `
        <div class="info-row">
            <span class="info-label">${escapeHtml(label)}</span>
            <span class="info-value">${escapeHtml(String(value))}</span>
        </div>
    `).join('');
}

function formatNumber(n) {
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    return n.toLocaleString();
}

// ===== Chat View =====

async function runPrompt(api) {
    const textarea = document.getElementById('prompt-input');
    const prompt = textarea.value.trim();
    if (!prompt || state.running) return;

    state.running = true;
    const sendBtn = document.getElementById('btn-send');
    sendBtn.disabled = true;
    sendBtn.textContent = 'Running...';

    const agentCount = parseInt(document.getElementById('agent-count').value);
    const strategy = document.getElementById('strategy-select').value;

    // Setup agents
    await api.setup_agents(agentCount, strategy);

    // Show loading state
    const outputArea = document.getElementById('output-area');
    outputArea.innerHTML = `<div class="output-placeholder"><span class="spinner"></span>Generating response...</div>`;

    // Hide metrics
    document.getElementById('metrics-bar').style.display = 'none';

    try {
        const result = await api.run_prompt(prompt, strategy);

        if (result.success) {
            renderResults(result);
        } else {
            outputArea.innerHTML = `<div class="status-message error">${escapeHtml(result.error)}</div>`;
        }
    } catch (e) {
        outputArea.innerHTML = `<div class="status-message error">Error: ${escapeHtml(e.message || String(e))}</div>`;
    }

    state.running = false;
    sendBtn.disabled = false;
    sendBtn.textContent = 'Run';
}

function renderResults(result) {
    const outputArea = document.getElementById('output-area');
    const agents = result.agents || [];

    const isSingle = agents.length === 1;
    const panelsHtml = agents.map(agent => {
        const metricsHtml = buildMetricsHtml(agent);
        return `
            <div class="agent-output">
                <div class="agent-output-header">
                    <span>${escapeHtml(agent.agent_name)}</span>
                    <span class="agent-output-metrics">${metricsHtml}</span>
                </div>
                <div class="agent-output-body">${escapeHtml(agent.text)}</div>
            </div>
        `;
    }).join('');

    outputArea.innerHTML = `<div class="agent-outputs${isSingle ? ' single' : ''}">${panelsHtml}</div>`;

    // Show total metrics bar
    const metricsBar = document.getElementById('metrics-bar');
    metricsBar.style.display = 'flex';
    metricsBar.innerHTML = `
        <div class="metric">
            <span class="metric-label">Strategy:</span>
            <span class="metric-value">${escapeHtml(result.strategy)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total time:</span>
            <span class="metric-value">${result.total_time_seconds}s</span>
        </div>
        <div class="metric">
            <span class="metric-label">Model calls:</span>
            <span class="metric-value">${result.total_model_calls}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Rounds:</span>
            <span class="metric-value">${result.rounds}</span>
        </div>
    `;
}

function buildMetricsHtml(agent) {
    const parts = [];
    const cls = agent.metrics_measured ? 'metric-measured' : 'metric-estimated';

    if (agent.tokens_per_second != null) {
        parts.push(`<span class="${cls}">${agent.tokens_per_second} tok/s</span>`);
    }
    if (agent.completion_tokens != null) {
        parts.push(`<span class="${cls}">${agent.completion_tokens} tokens</span>`);
    }
    if (agent.generation_time != null) {
        parts.push(`<span class="${cls}">${agent.generation_time}s</span>`);
    }

    return parts.join(' · ');
}

// ===== Strategy / Agent count sync =====
document.getElementById('strategy-select').addEventListener('change', (e) => {
    const strategy = e.target.value;
    const countSelect = document.getElementById('agent-count');
    if (strategy === 'single') {
        countSelect.value = '1';
    } else if (['independent', 'solver_critic', 'debate'].includes(strategy) && parseInt(countSelect.value) < 2) {
        countSelect.value = '2';
    }
});

document.getElementById('agent-count').addEventListener('change', (e) => {
    const count = parseInt(e.target.value);
    const stratSelect = document.getElementById('strategy-select');
    if (count === 1) {
        stratSelect.value = 'single';
    } else if (stratSelect.value === 'single') {
        stratSelect.value = 'independent';
    }
});

// ===== Keyboard shortcuts =====
document.getElementById('prompt-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        document.getElementById('btn-send').click();
    }
});

document.getElementById('model-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        document.getElementById('btn-load-model').click();
    }
});

// ===== Init =====
async function init() {
    const api = await waitForBridge();

    // Wire up buttons
    document.getElementById('btn-connect').addEventListener('click', () => connectRuntime(api));
    document.getElementById('btn-load-model').addEventListener('click', () => loadModel(api));
    document.getElementById('btn-send').addEventListener('click', () => runPrompt(api));

    // Run setup
    await initSetup(api);
}

init().catch(err => {
    console.error('Init failed:', err);
    document.getElementById('system-info').innerHTML =
        `<p class="info-loading" style="color: var(--error)">Initialization failed: ${escapeHtml(err.message || String(err))}</p>`;
});
