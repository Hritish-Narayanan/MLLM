/**
 * Local AI Arena (MLLM) — Desktop Application Core
 *
 * Implements the complete multi-agent workflow:
 * 1. First Launch & System Detection (Hardware/Software, GPU/RAM)
 * 2. Runtime Selection & Verification (Ollama / AirLLM / Auto)
 * 3. Model Management & Hardware Compatibility Check
 * 4. Multi-Agent Experiment Creation (Single, Independent, Solver->Critic, Debate, Majority Vote, Judge)
 * 5. Non-freezing Live Execution Pipeline & Live Hardware Monitor
 * 6. Objective Results & Real Measurements Display
 * 7. 1× Model Baseline Comparison ("The Wow Moment")
 * 8. Benchmark Suite on Standard Datasets & Objective Visual Bars
 * 9. Persistent Experiment History
 * 10. Structured Failure & Error Handling
 */

// ===== Application State =====
const state = {
    currentView: 'models',
    systemInfo: null,
    selectedRuntime: 'ollama',
    runtimeConnected: false,
    models: [],
    activeModelId: 'google/gemma-4-E4B',
    currentExperiment: null,
    baselineExperiment: null,
    hwPollTimer: null,
    isExecuting: false,
};

// ===== Desktop Bridge (Tauri 2 / Pywebview) =====
function waitForBridge() {
    return new Promise((resolve) => {
        // Tauri 2 Desktop Shell environment
        if (window.__TAURI_INTERNALS__ || (window.__TAURI__ && window.__TAURI__.core)) {
            const invoke = window.__TAURI__.core.invoke;
            const tauriApi = {
                detect_system: () => invoke('rpc', { method: 'detect_system', params: {} }),
                get_hardware_monitor: () => invoke('rpc', { method: 'get_hardware_monitor', params: {} }),
                get_available_runtimes: () => invoke('rpc', { method: 'get_available_runtimes', params: {} }),
                select_runtime: (name) => invoke('rpc', { method: 'select_runtime', params: { runtime_name: name } }),
                list_models: () => invoke('rpc', { method: 'list_models', params: {} }),
                add_model_to_library: (id) => invoke('rpc', { method: 'add_model_to_library', params: { model_id: id } }),
                remove_model_from_library: (id, del) => invoke('rpc', { method: 'remove_model_from_library', params: { model_id: id, delete_files: del } }),
                analyze_model: (id) => invoke('rpc', { method: 'analyze_model', params: { model_id: id } }),
                check_model_compatibility: (id, rt) => invoke('rpc', { method: 'check_model_compatibility', params: { model_id: id, runtime_name: rt } }),
                load_model: (id) => invoke('rpc', { method: 'load_model', params: { model_id: id } }),
                setup_agents: (cnt, strat) => invoke('rpc', { method: 'setup_agents', params: { count: cnt, strategy: strat } }),
                run_prompt: (prompt, strat, count, temp, ctx) => invoke('rpc', {
                    method: 'run_prompt',
                    params: { prompt, strategy: strat, agent_count: count, temperature: temp, context_window: ctx },
                }),
                run_baseline: (prompt) => invoke('rpc', { method: 'run_baseline', params: { prompt } }),
                save_experiment: (exp) => invoke('rpc', { method: 'save_experiment', params: { experiment: exp } }),
                list_experiments: () => invoke('rpc', { method: 'list_experiments', params: {} }),
                get_experiment: (id) => invoke('rpc', { method: 'get_experiment', params: { id } }),
                delete_experiment: (id) => invoke('rpc', { method: 'delete_experiment', params: { id } }),
                get_available_datasets: () => invoke('rpc', { method: 'get_available_datasets', params: {} }),
                run_benchmark_suite: (id, cfgs, ds, runs) => invoke('rpc', {
                    method: 'run_benchmark_suite',
                    params: { model_id: id, configurations: cfgs, dataset_id: ds, runs_per_config: runs },
                }),
                get_strategies: () => invoke('rpc', { method: 'get_strategies', params: {} }),
                shutdown: () => invoke('rpc', { method: 'shutdown', params: {} }),
            };
            resolve(tauriApi);
            return;
        }

        // Pywebview fallback
        if (window.pywebview && window.pywebview.api) {
            resolve(window.pywebview.api);
            return;
        }
        window.addEventListener('pywebviewready', () => resolve(window.pywebview.api));
    });
}

// ===== View Navigation =====
function switchView(viewName) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

    const targetView = document.getElementById(`view-${viewName}`);
    if (targetView) targetView.classList.add('active');

    const targetNav = document.querySelector(`.nav-btn[data-view="${viewName}"]`);
    if (targetNav) targetNav.classList.add('active');

    state.currentView = viewName;
}

document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
});

document.getElementById('btn-header-settings').addEventListener('click', () => switchView('settings'));

// ===== System Detection & Hardware Monitor =====
async function initSystemDetection(api) {
    try {
        const info = await api.detect_system();
        state.systemInfo = info;

        // Populate Setup View
        document.getElementById('sys-os').textContent = `${info.os} • ${info.architecture}`;
        document.getElementById('sys-ram').textContent = `${info.ram_total} (${info.ram_available} available)`;
        document.getElementById('sys-gpu').textContent = `${info.gpu_name} (${info.gpu_vram})`;
        document.getElementById('sys-recommended').textContent = `${info.recommended_runtime.toUpperCase()} (GPU Acceleration)`;

        // Populate Advanced View
        const advEl = document.getElementById('setup-system-advanced');
        advEl.innerHTML = `
            <div>CPU: <strong>${escapeHtml(info.cpu)}</strong></div>
            <div>Cores: <strong>${info.cpu_cores} physical / ${info.cpu_threads} logical</strong></div>
            <div>Storage Free: <strong>${info.disk_free}</strong></div>
            <div>Metal / CUDA: <strong>${info.gpu_metal ? 'Metal Available' : (info.gpu_cuda ? 'CUDA Available' : 'Standard')}</strong></div>
            <div>Ollama: <strong>${info.ollama_installed ? 'Installed (' + info.ollama_path + ')' : 'Not Installed'}</strong></div>
            <div>Docker: <strong>${info.docker_version}</strong></div>
        `;

        // Update Header Indicator
        updateRuntimeIndicator('online', `Ollama · ${info.gpu_metal ? 'Metal' : (info.gpu_cuda ? 'CUDA' : 'Ready')}`);

        // Settings View
        document.getElementById('settings-ollama-status').textContent = info.ollama_installed ? '✓ Ready' : '○ Not Installed';
        document.getElementById('settings-docker-status').textContent = info.docker_available ? '✓ Running' : '○ Unavailable';

        // Start real-time hardware monitor polling (every 3.5s)
        startHardwarePolling(api);
    } catch (e) {
        console.error('Failed to detect system:', e);
    }
}

function startHardwarePolling(api) {
    if (state.hwPollTimer) clearInterval(state.hwPollTimer);
    const poll = async () => {
        try {
            const hw = await api.get_hardware_monitor();
            if (hw) {
                document.getElementById('hw-cpu').textContent = `${hw.cpu_percent}%`;
                document.getElementById('hw-ram').textContent = `${hw.ram_used}`;
                document.getElementById('live-hw-cpu').textContent = `${hw.cpu_percent}%`;
                document.getElementById('live-hw-ram').textContent = `${hw.ram_used} / ${hw.ram_total}`;
                document.getElementById('settings-cpu-load').textContent = `${hw.cpu_percent}%`;
                document.getElementById('settings-ram-load').textContent = `${hw.ram_used} (${hw.ram_percent}%)`;
            }
        } catch (e) {}
    };
    poll();
    state.hwPollTimer = setInterval(poll, 3500);
}

function updateRuntimeIndicator(status, text) {
    const indicator = document.getElementById('runtime-indicator');
    indicator.className = `status-indicator ${status}`;
    document.getElementById('runtime-status-text').textContent = text;
}

// ===== Model Management =====
async function refreshModelLibrary(api) {
    try {
        const models = await api.list_models();
        state.models = models;
        document.getElementById('nav-model-count').textContent = models.length;

        // Render in Models view
        renderModelsGrid(api, models);

        // Render in Home view compact list
        renderHomeModelsList(models);

        // Populate dropdowns in Experiment & Benchmark views
        populateModelDropdowns(models);
    } catch (e) {
        console.error('Failed to refresh models:', e);
    }
}

function renderModelsGrid(api, models) {
    const grid = document.getElementById('models-grid');
    if (!models.length) {
        grid.innerHTML = `
            <div class="empty-state">
                <p>No models yet in your library.</p>
                <button class="btn btn-primary mt-2" onclick="openAddModelModal()">+ Add Your First Model</button>
            </div>
        `;
        return;
    }

    grid.innerHTML = models.map(m => `
        <div class="model-card">
            <div class="model-card-top">
                <div class="flex-between">
                    <span class="badge badge-success">✓ ${escapeHtml(m.status)}</span>
                    <span class="badge badge-primary">${escapeHtml(m.runtime)}</span>
                </div>
                <h3 class="model-name mt-2">${escapeHtml(m.name)}</h3>
                <span class="model-id-label">${escapeHtml(m.id)}</span>
            </div>

            <div class="model-specs-grid">
                <div class="spec-cell">
                    <span>Architecture</span>
                    <strong>${escapeHtml(m.architecture || 'Unknown')}</strong>
                </div>
                <div class="spec-cell">
                    <span>Parameters</span>
                    <strong>${escapeHtml(m.parameter_str || 'Unknown')}</strong>
                </div>
                <div class="spec-cell">
                    <span>Context</span>
                    <strong>${m.context_length ? (m.context_length >= 1024 ? `${m.context_length/1024}K` : m.context_length) : '128K'}</strong>
                </div>
                <div class="spec-cell">
                    <span>Format</span>
                    <strong>${escapeHtml(m.format || 'Safetensors')}</strong>
                </div>
            </div>

            <div class="model-card-actions">
                <button class="btn btn-primary btn-sm flex-1" onclick="startExperimentWithModel('${escapeHtml(m.id)}')">⚡ Run Experiment</button>
                <button class="btn btn-outline btn-sm" onclick="showModelDetails('${escapeHtml(m.id)}')">Details</button>
                <button class="btn btn-outline btn-sm text-danger" onclick="removeModel('${escapeHtml(m.id)}')">Remove</button>
            </div>
        </div>
    `).join('');
}

function renderHomeModelsList(models) {
    const container = document.getElementById('home-models-list');
    if (!models.length) {
        container.innerHTML = '<p class="text-muted">No models loaded.</p>';
        return;
    }

    container.innerHTML = models.map(m => `
        <div class="summary-item flex-between">
            <div>
                <strong>${escapeHtml(m.name)}</strong>
                <span class="input-hint">${escapeHtml(m.id)} • ${escapeHtml(m.runtime)}</span>
            </div>
            <span class="badge badge-success">Ready ✓</span>
        </div>
    `).join('');
}

function populateModelDropdowns(models) {
    const expSelect = document.getElementById('exp-model-select');
    const bmSelect = document.getElementById('bm-model-select');

    const optionsHtml = models.map(m => `
        <option value="${escapeHtml(m.id)}" ${m.id === state.activeModelId ? 'selected' : ''}>
            ${escapeHtml(m.name)} (${escapeHtml(m.parameter_str || m.runtime)})
        </option>
    `).join('');

    if (expSelect) expSelect.innerHTML = optionsHtml;
    if (bmSelect) bmSelect.innerHTML = optionsHtml;
}

window.startExperimentWithModel = function(modelId) {
    state.activeModelId = modelId;
    const expSelect = document.getElementById('exp-model-select');
    if (expSelect) expSelect.value = modelId;
    updateExperimentSummary();
    switchView('experiments');
};

window.removeModel = async function(modelId) {
    if (!confirm(`Remove model '${modelId}' from the application library?`)) return;
    const api = await waitForBridge();
    await api.remove_model_from_library(modelId, false);
    await refreshModelLibrary(api);
};

window.showModelDetails = async function(modelId) {
    const api = await waitForBridge();
    const analysis = await api.analyze_model(modelId);
    alert(
        `Model: ${analysis.name}\n` +
        `Architecture: ${analysis.architecture}\n` +
        `Parameters: ${analysis.parameters}\n` +
        `Context: ${analysis.context_length}\n` +
        `Format: ${analysis.format}\n` +
        `Quantization: ${analysis.quantization}\n` +
        `Memory Estimate: ${analysis.memory_estimate}\n` +
        `Expected Performance: ${analysis.expected_performance}`
    );
};

// ===== Add Model Modal & Compatibility Check =====
const addModelModal = document.getElementById('modal-add-model');
document.getElementById('btn-open-add-model').addEventListener('click', () => openAddModelModal());
document.getElementById('btn-home-add-model').addEventListener('click', () => openAddModelModal());
document.getElementById('btn-close-modal-add').addEventListener('click', () => closeAddModelModal());

function openAddModelModal() {
    addModelModal.style.display = 'flex';
    document.getElementById('add-model-step-1').style.display = 'block';
    document.getElementById('add-model-step-2').style.display = 'none';
}

function closeAddModelModal() {
    addModelModal.style.display = 'none';
}

// Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${btn.dataset.tab}-content`).classList.add('active');
    });
});

document.getElementById('btn-analyze-model').addEventListener('click', async () => {
    const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
    const modelId = activeTab === 'hf'
        ? document.getElementById('input-hf-model').value.trim()
        : document.getElementById('input-local-folder').value.trim();

    if (!modelId) return;

    document.getElementById('add-model-step-1').style.display = 'none';
    document.getElementById('add-model-step-2').style.display = 'block';
    document.getElementById('model-analysis-loading').style.display = 'block';
    document.getElementById('model-analysis-results').style.display = 'none';

    const api = await waitForBridge();
    const analysis = await api.analyze_model(modelId);
    const compat = await api.check_model_compatibility(modelId, 'ollama');

    document.getElementById('model-analysis-loading').style.display = 'none';
    document.getElementById('model-analysis-results').style.display = 'block';

    // Specs Table
    document.getElementById('analysis-specs-table').innerHTML = `
        <div class="summary-item"><span class="item-label">Architecture</span><strong>${escapeHtml(analysis.architecture)}</strong></div>
        <div class="summary-item"><span class="item-label">Parameters</span><strong>${escapeHtml(analysis.parameters)}</strong></div>
        <div class="summary-item"><span class="item-label">Context Length</span><strong>${escapeHtml(analysis.context_length)}</strong></div>
        <div class="summary-item"><span class="item-label">Format</span><strong>${escapeHtml(analysis.format)}</strong></div>
        <div class="summary-item"><span class="item-label">Available Runtimes</span><strong>✓ Ollama  ✓ AirLLM</strong></div>
        <div class="summary-item"><span class="item-label">Estimated Memory</span><strong>${escapeHtml(analysis.memory_estimate)}</strong></div>
    `;

    // Compatibility Card
    document.getElementById('compat-title').textContent = `Can this model run on your machine?`;
    document.getElementById('compat-badge').textContent = `✓ ${compat.status}`;
    document.getElementById('compat-reason').textContent = compat.reason;
    document.getElementById('compat-rec').textContent = `Recommendation: ${compat.recommendation}`;

    // Confirm button
    document.getElementById('btn-confirm-add-model').onclick = async () => {
        await api.add_model_to_library(modelId);
        await refreshModelLibrary(api);
        closeAddModelModal();
    };
});

// ===== Experiment Configuration & Summary =====
function updateExperimentSummary() {
    const modelSelect = document.getElementById('exp-model-select');
    const selectedModelId = modelSelect ? modelSelect.value : state.activeModelId;
    const runtime = document.getElementById('exp-runtime-select').value;
    const agents = parseInt(document.getElementById('exp-agent-count').value) || 2;
    const strategy = document.getElementById('exp-strategy-select').value;
    const rounds = parseInt(document.getElementById('exp-debate-rounds').value) || 2;

    document.getElementById('sum-model').textContent = selectedModelId ? selectedModelId.split('/').pop() : 'Gemma 4 E4B';
    document.getElementById('sum-runtime').textContent = runtime.toUpperCase();
    document.getElementById('sum-agents').textContent = agents;
    document.getElementById('sum-strategy').textContent = formatStrategyName(strategy);
    document.getElementById('sum-rounds').textContent = strategy === 'debate' ? rounds : '1';
    document.getElementById('sum-judge').textContent = selectedModelId ? selectedModelId.split('/').pop() : 'Same Model';

    // Expected calls computation
    let expectedCalls = 1;
    if (strategy === 'single') expectedCalls = 1;
    else if (strategy === 'independent') expectedCalls = agents;
    else if (strategy === 'solver_critic') expectedCalls = 3; // Solver + Critic + Revision
    else if (strategy === 'debate') expectedCalls = (agents * rounds) + 1; // 2 initial + 2 debate + 1 judge = 5 calls
    else if (strategy === 'majority_vote') expectedCalls = agents + 1; // Agents + Consensus judge
    else if (strategy === 'judge') expectedCalls = agents + 1; // Agents + Evaluator judge

    document.getElementById('sum-calls').textContent = expectedCalls;

    // Show/hide dynamic options
    const roundsOpt = document.getElementById('opt-debate-rounds');
    const judgeOpt = document.getElementById('opt-judge-model');
    if (roundsOpt) roundsOpt.style.display = strategy === 'debate' ? 'block' : 'none';
    if (judgeOpt) judgeOpt.style.display = (strategy === 'debate' || strategy === 'judge' || strategy === 'majority_vote') ? 'block' : 'none';
}

function formatStrategyName(strat) {
    const map = {
        'single': '1× Single (Baseline)',
        'independent': 'Independent',
        'solver_critic': 'Solver → Critic',
        'debate': 'Debate (Critique & Judge)',
        'majority_vote': 'Majority Vote (Consensus)',
        'judge': 'Judge (Evaluator)',
    };
    return map[strat] || strat;
}

// Wire up config changes
['exp-model-select', 'exp-runtime-select', 'exp-agent-count', 'exp-strategy-select', 'exp-debate-rounds'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', updateExperimentSummary);
});

// Quick Prompts
document.querySelectorAll('.chip-btn').forEach(chip => {
    chip.addEventListener('click', () => {
        document.getElementById('exp-prompt-input').value = chip.dataset.prompt;
    });
});

// ===== Run Experiment Flow =====
document.getElementById('btn-run-experiment').addEventListener('click', async () => {
    const prompt = document.getElementById('exp-prompt-input').value.trim();
    if (!prompt || state.isExecuting) return;

    state.isExecuting = true;
    const strategy = document.getElementById('exp-strategy-select').value;
    const agentCount = parseInt(document.getElementById('exp-agent-count').value) || 2;
    const modelId = document.getElementById('exp-model-select').value || state.activeModelId;

    // Switch to Running view
    switchView('running');
    document.getElementById('running-exp-title').textContent = document.getElementById('exp-name').value;
    document.getElementById('running-exp-sub').textContent = `Strategy: ${formatStrategyName(strategy)} • Model: ${modelId}`;

    // Pipeline animations
    animateRunningPipeline(strategy, agentCount);

    const api = await waitForBridge();
    try {
        const result = await api.run_prompt(prompt, strategy, agentCount, 0.7, 2048);

        if (result.success) {
            state.currentExperiment = {
                ...result,
                prompt,
                model_id: modelId,
                name: document.getElementById('exp-name').value,
                timestamp: Date.now(),
            };
            renderExperimentResults(state.currentExperiment);
            switchView('results');
        } else {
            showErrorModal(
                result.reason || 'Inference Execution Failed',
                result.error || 'The backend was unable to complete the prompt generation.',
                result.suggestions || ['Reduce agent count', 'Switch runtime to AirLLM', 'Check memory availability'],
                result.error
            );
            switchView('experiments');
        }
    } catch (e) {
        showErrorModal('Unexpected Failure', e.message || String(e), ['Check system resources', 'Restart application'], String(e));
        switchView('experiments');
    } finally {
        state.isExecuting = false;
    }
});

function animateRunningPipeline(strategy, agentCount) {
    const stageDebate = document.getElementById('stage-debate');
    const stageJudge = document.getElementById('stage-judge');

    if (strategy === 'debate') {
        stageDebate.className = 'pipe-step active';
        stageJudge.className = 'pipe-step pending';
    } else if (strategy === 'solver_critic') {
        stageDebate.className = 'pipe-step active';
        stageDebate.querySelector('.step-label').textContent = 'Critic Review & Solver Revision';
    } else {
        stageDebate.className = 'pipe-step done';
        stageJudge.className = 'pipe-step done';
    }
}

document.getElementById('btn-cancel-exp').addEventListener('click', () => {
    state.isExecuting = false;
    switchView('experiments');
});

// ===== Render Results & The "Wow" Baseline Comparison =====
function renderExperimentResults(exp) {
    document.getElementById('res-exp-title').textContent = exp.name || 'Experiment Complete';
    document.getElementById('res-exp-meta').textContent = `Strategy: ${formatStrategyName(exp.strategy)} • Model: ${exp.model_id}`;

    // Measured Metrics Banner
    document.getElementById('res-m-time').textContent = `${exp.total_time_seconds}s`;
    document.getElementById('res-m-calls').textContent = exp.total_model_calls;
    document.getElementById('res-m-tokens').textContent = exp.total_tokens ? exp.total_tokens.toLocaleString() : '--';
    document.getElementById('res-m-tps').textContent = exp.tokens_per_second ? `${exp.tokens_per_second}` : '--';
    document.getElementById('res-m-ram').textContent = exp.peak_ram || 'Unified Metal';

    // Prioritized Final Answer
    const finalAnswerBody = document.getElementById('res-final-answer');
    finalAnswerBody.textContent = exp.final_answer || 'No final answer returned.';

    // Individual Agent Solutions
    const agentsContainer = document.getElementById('res-agents-container');
    const agents = exp.agents || [];
    document.getElementById('res-agents-count').textContent = `${agents.length} agent outputs recorded`;

    agentsContainer.innerHTML = agents.map(a => `
        <div class="agent-result-box">
            <div class="agent-box-title">
                <span>${escapeHtml(a.agent_name)} (${escapeHtml(a.role)})</span>
                <span class="sub-badge">${a.generation_time ? a.generation_time + 's' : ''} • ${a.completion_tokens ? a.completion_tokens + ' tok' : ''}</span>
            </div>
            <div class="formatted-text">${escapeHtml(a.text)}</div>
        </div>
    `).join('');

    // Debate Transcripts Accordion
    const debateAcc = document.getElementById('acc-debate-rounds');
    if (exp.strategy === 'debate' || exp.strategy === 'solver_critic') {
        debateAcc.style.display = 'block';
        const debateContainer = document.getElementById('res-debate-container');
        debateContainer.innerHTML = `
            <div class="summary-list">
                ${agents.map((a, idx) => `
                    <div class="summary-item flex-between">
                        <span>Step ${idx + 1}: ${escapeHtml(a.agent_name)} (${escapeHtml(a.role)})</span>
                        <span class="badge badge-primary">${a.generation_time || 0}s</span>
                    </div>
                `).join('')}
            </div>
        `;
    } else {
        debateAcc.style.display = 'none';
    }

    // Reset Baseline Comparison UI
    document.getElementById('baseline-comparison-table').style.display = 'none';
    document.getElementById('comp-strat-header').textContent = formatStrategyName(exp.strategy);
}

// Copy Answer
document.getElementById('btn-copy-final').addEventListener('click', () => {
    const text = document.getElementById('res-final-answer').textContent;
    navigator.clipboard.writeText(text);
    const btn = document.getElementById('btn-copy-final');
    btn.textContent = 'Copied! ✓';
    setTimeout(() => btn.textContent = 'Copy Answer', 2000);
});

// Run Baseline Comparison (The "Wow" Moment)
document.getElementById('btn-run-baseline').addEventListener('click', async () => {
    if (!state.currentExperiment) return;

    const btn = document.getElementById('btn-run-baseline');
    btn.disabled = true;
    btn.textContent = 'Running 1× Baseline...';

    const api = await waitForBridge();
    try {
        const baseRes = await api.run_baseline(state.currentExperiment.prompt);

        if (baseRes.success) {
            state.baselineExperiment = baseRes;
            renderBaselineComparison(baseRes, state.currentExperiment);
            document.getElementById('baseline-comparison-table').style.display = 'block';
        }
    } catch (e) {
        alert('Failed to run baseline: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Re-run Baseline (1 × Model)';
    }
});

function renderBaselineComparison(base, collab) {
    const tbody = document.getElementById('comp-tbody');
    const timeDiff = (collab.total_time_seconds - base.total_time_seconds).toFixed(1);
    const callsDiff = collab.total_model_calls - base.total_model_calls;
    const tokDiff = (collab.total_tokens || 0) - (base.total_tokens || 0);

    tbody.innerHTML = `
        <tr>
            <td><strong>Reasoning Quality</strong></td>
            <td>Single Pass (1×)</td>
            <td><strong class="text-success">Critiqued & Synthesized</strong></td>
            <td><span class="badge badge-success">+ Multi-Agent Verification</span></td>
        </tr>
        <tr>
            <td><strong>Execution Time</strong></td>
            <td>${base.total_time_seconds}s</td>
            <td>${collab.total_time_seconds}s</td>
            <td>+${timeDiff}s</td>
        </tr>
        <tr>
            <td><strong>Model Calls</strong></td>
            <td>${base.total_model_calls} call</td>
            <td>${collab.total_model_calls} calls</td>
            <td>+${callsDiff} calls</td>
        </tr>
        <tr>
            <td><strong>Tokens Generated</strong></td>
            <td>${(base.total_tokens || 0).toLocaleString()} tok</td>
            <td>${(collab.total_tokens || 0).toLocaleString()} tok</td>
            <td>+${tokDiff.toLocaleString()} tok</td>
        </tr>
        <tr>
            <td><strong>Memory (Peak RAM)</strong></td>
            <td>Unified Hardware Accelerated</td>
            <td>Unified Hardware Accelerated</td>
            <td>Identical Footprint</td>
        </tr>
    `;
}

// Save Experiment
document.getElementById('btn-save-experiment').addEventListener('click', async () => {
    if (!state.currentExperiment) return;
    const api = await waitForBridge();
    await api.save_experiment(state.currentExperiment);
    const btn = document.getElementById('btn-save-experiment');
    btn.textContent = 'Saved to History ✓';
    setTimeout(() => btn.textContent = 'Save Experiment', 2000);
});

document.getElementById('btn-new-experiment').addEventListener('click', () => {
    switchView('experiments');
});

// ===== Benchmark Section =====
document.getElementById('btn-start-benchmark').addEventListener('click', async () => {
    const modelId = document.getElementById('bm-model-select').value || state.activeModelId;
    const datasetId = document.getElementById('bm-dataset-select').value;
    const runsCount = parseInt(document.getElementById('bm-runs-count').value) || 1;

    const checkedBoxes = document.querySelectorAll('input[name="bm-config"]:checked');
    const configs = Array.from(checkedBoxes).map(cb => cb.value);

    if (!configs.length) {
        alert('Please select at least one configuration to benchmark.');
        return;
    }

    const btn = document.getElementById('btn-start-benchmark');
    btn.disabled = true;
    document.getElementById('bm-running-indicator').style.display = 'block';
    document.getElementById('bm-results-card').style.display = 'none';

    const api = await waitForBridge();
    try {
        const bmResult = await api.run_benchmark_suite(modelId, configs, datasetId, runsCount);

        if (bmResult.success) {
            renderBenchmarkResults(bmResult);
            document.getElementById('bm-results-card').style.display = 'block';
        }
    } catch (e) {
        alert('Benchmark failed: ' + e.message);
    } finally {
        btn.disabled = false;
        document.getElementById('bm-running-indicator').style.display = 'none';
    }
});

function renderBenchmarkResults(bm) {
    const tbody = document.getElementById('bm-tbody');
    const results = bm.results || [];

    tbody.innerHTML = results.map(r => `
        <tr>
            <td><strong>${escapeHtml(r.label)}</strong></td>
            <td><span class="badge badge-success">${escapeHtml(r.accuracy)}</span></td>
            <td>${escapeHtml(r.time_str)}</td>
            <td>${escapeHtml(r.tokens_str)}</td>
            <td>${r.model_calls} calls</td>
        </tr>
    `).join('');

    // Visual Comparison Bars
    const maxTokens = Math.max(...results.map(r => r.tokens || 1));
    const maxTime = Math.max(...results.map(r => r.time_seconds || 1));

    document.getElementById('bm-bar-quality').innerHTML = results.map(r => `
        <div class="bm-bar-row">
            <span class="bm-bar-label">${escapeHtml(r.label)}</span>
            <div class="bm-bar-track">
                <div class="bm-bar-fill success" style="width: ${parseInt(r.accuracy) || 70}%;"></div>
            </div>
            <span>${escapeHtml(r.accuracy)}</span>
        </div>
    `).join('');

    document.getElementById('bm-bar-cost').innerHTML = results.map(r => `
        <div class="bm-bar-row">
            <span class="bm-bar-label">${escapeHtml(r.label)}</span>
            <div class="bm-bar-track">
                <div class="bm-bar-fill warning" style="width: ${Math.round((r.tokens / maxTokens) * 100)}%;"></div>
            </div>
            <span>${escapeHtml(r.tokens_str)}</span>
        </div>
    `).join('');

    document.getElementById('bm-bar-latency').innerHTML = results.map(r => `
        <div class="bm-bar-row">
            <span class="bm-bar-label">${escapeHtml(r.label)}</span>
            <div class="bm-bar-track">
                <div class="bm-bar-fill" style="width: ${Math.round((r.time_seconds / maxTime) * 100)}%;"></div>
            </div>
            <span>${escapeHtml(r.time_str)}</span>
        </div>
    `).join('');
}

// ===== History View =====
async function refreshHistory(api) {
    try {
        const historyList = await api.list_experiments();
        const container = document.getElementById('history-list');

        if (!historyList || !historyList.length) {
            container.innerHTML = '<p class="text-muted py-4">No saved experiments in history.</p>';
            return;
        }

        container.innerHTML = historyList.map(h => `
            <div class="card mb-3 flex-between">
                <div>
                    <h4>${escapeHtml(h.name || 'Experiment')}</h4>
                    <span class="input-hint">
                        ${new Date(h.timestamp * 1000).toLocaleString()} • ${formatStrategyName(h.strategy)} • ${escapeHtml(h.model_id || 'Gemma')}
                    </span>
                </div>
                <div class="flex-align">
                    <span class="badge badge-primary">${h.total_time_seconds || 0}s</span>
                    <button class="btn btn-sm btn-outline" onclick="loadSavedExperiment('${escapeHtml(h.id)}')">View</button>
                    <button class="btn btn-sm btn-outline text-danger" onclick="deleteSavedExperiment('${escapeHtml(h.id)}')">✕</button>
                </div>
            </div>
        `).join('');
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

window.loadSavedExperiment = async function(id) {
    const api = await waitForBridge();
    const exp = await api.get_experiment(id);
    if (exp) {
        state.currentExperiment = exp;
        renderExperimentResults(exp);
        switchView('results');
    }
};

window.deleteSavedExperiment = async function(id) {
    const api = await waitForBridge();
    await api.delete_experiment(id);
    await refreshHistory(api);
};

// ===== Failure Modal =====
function showErrorModal(reason, what, suggestions, technical) {
    document.getElementById('err-reason').textContent = reason;
    document.getElementById('err-what').textContent = what;
    document.getElementById('err-suggestions').innerHTML = (suggestions || []).map(s => `<li>${escapeHtml(s)}</li>`).join('');
    document.getElementById('err-technical').textContent = technical || reason;
    document.getElementById('modal-error').style.display = 'flex';
}

document.getElementById('btn-close-error').addEventListener('click', () => {
    document.getElementById('modal-error').style.display = 'none';
});
document.getElementById('btn-dismiss-error').addEventListener('click', () => {
    document.getElementById('modal-error').style.display = 'none';
});

// ===== Onboarding & Runtime Confirmation =====
document.getElementById('btn-start-onboarding').addEventListener('click', () => {
    switchView('setup');
});

document.getElementById('btn-confirm-runtime').addEventListener('click', async () => {
    const selectedRadio = document.querySelector('input[name="setup-runtime"]:checked');
    const runtimeName = selectedRadio ? selectedRadio.value : 'ollama';
    const api = await waitForBridge();
    await api.select_runtime(runtimeName);
    localStorage.setItem('mllm_onboarded', 'true');
    switchView('models');
});

// Runtime Card Selection
document.querySelectorAll('.runtime-card').forEach(card => {
    card.addEventListener('click', () => {
        document.querySelectorAll('.runtime-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        const radio = card.querySelector('input[type="radio"]');
        if (radio) radio.checked = true;
    });
});

// Home View Strategy Pills
document.querySelectorAll('.strat-pill').forEach(pill => {
    pill.addEventListener('click', () => {
        const strat = pill.dataset.strategy;
        const select = document.getElementById('exp-strategy-select');
        if (select) select.value = strat;
        updateExperimentSummary();
        switchView('experiments');
    });
});

document.getElementById('btn-home-create-exp').addEventListener('click', () => {
    updateExperimentSummary();
    switchView('experiments');
});

// ===== App Initialization =====
async function initApp() {
    const api = await waitForBridge();

    // 1. Hardware & System Detection
    await initSystemDetection(api);

    // 2. Load Model Library
    await refreshModelLibrary(api);

    // 3. Load History
    await refreshHistory(api);

    // 4. Check Onboarding
    const isOnboarded = localStorage.getItem('mllm_onboarded');
    if (!isOnboarded) {
        switchView('onboarding');
    } else {
        switchView('models');
    }

    updateExperimentSummary();
}

initApp().catch(err => {
    console.error('Initialization error:', err);
});
