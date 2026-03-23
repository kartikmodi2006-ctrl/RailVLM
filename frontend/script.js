const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('image-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const resetBtn = document.getElementById('reset-btn');
const loadingSection = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');

let currentFile = null;

// ── Drag & Drop ──────────────────────────────────────────────────────────────
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e =>
    dropZone.addEventListener(e, ev => { ev.preventDefault(); ev.stopPropagation(); }));

['dragenter', 'dragover'].forEach(e =>
    dropZone.addEventListener(e, () => dropZone.classList.add('dragover')));
['dragleave', 'drop'].forEach(e =>
    dropZone.addEventListener(e, () => dropZone.classList.remove('dragover')));

dropZone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
fileInput.addEventListener('change', function () { handleFiles(this.files); });

function handleFiles(files) {
    if (!files.length) return;
    currentFile = files[0];
    if (!currentFile.type.startsWith('image/')) { alert('Please upload an image file.'); return; }
    const reader = new FileReader();
    reader.onload = e => {
        imagePreview.src = e.target.result;
        dropZone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        resultsSection.classList.add('hidden');
    };
    reader.readAsDataURL(currentFile);
}

resetBtn.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    previewContainer.classList.add('hidden');
    dropZone.classList.remove('hidden');
    resultsSection.classList.add('hidden');
});

// ── Analyze ───────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    analyzeBtn.disabled = true;
    resetBtn.disabled = true;
    loadingSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('/detect', { method: 'POST', body: formData });
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        const data = await response.json();
        renderResults(data);
    } catch (err) {
        console.error(err);
        alert(`Analysis failed: ${err.message}. Check the server console.`);
    } finally {
        loadingSection.classList.add('hidden');
        analyzeBtn.disabled = false;
        resetBtn.disabled = false;
    }
});

// ── Render Results ────────────────────────────────────────────────────────────
function renderResults(data) {
    const v = data.verdict;

    // --- Verdict banner ---
    const banner = document.getElementById('verdict-banner');
    const icon = document.getElementById('verdict-icon');
    const label = document.getElementById('verdict-label');
    const sub = document.getElementById('verdict-sub');

    banner.classList.remove('verdict-safe', 'verdict-caution', 'verdict-defective');

    if (v.verdict === 'DEFECTIVE') {
        banner.classList.add('verdict-defective');
        icon.textContent = '🚨';
        label.textContent = 'DEFECTIVE — Immediate Inspection Required';
        sub.textContent = `${v.defect_count} defective zone(s) detected with avg confidence ${(v.dominant_conf * 100).toFixed(0)}%`;
    } else if (v.verdict === 'CAUTION') {
        banner.classList.add('verdict-caution');
        icon.textContent = '⚠️';
        label.textContent = 'CAUTION — Monitor Closely';
        sub.textContent = `${v.defect_count} potential fault(s) detected — follow-up inspection recommended`;
    } else {
        banner.classList.add('verdict-safe');
        icon.textContent = '✅';
        label.textContent = 'SAFE — No Significant Faults Detected';
        sub.textContent = 'Track section appears structurally sound based on visual analysis';
    }


    // --- Images ---
    document.getElementById('annotated-result').src = `data:image/jpeg;base64,${data.annotated_image}`;
    document.getElementById('heatmap-result').src = `data:image/jpeg;base64,${data.heatmap_image}`;

    // --- Explanations ---
    const list = document.getElementById('explanation-list');
    list.innerHTML = '';

    if (data.explanations.length === 0) {
        list.innerHTML = `
            <div class="no-faults-msg">
                <span>✅</span>
                <div>
                    <strong>No defective regions selected for explanation.</strong>
                    <p>Either no faults were detected, or all detections were classified as safe zones.</p>
                </div>
            </div>`;
    } else {
        data.explanations.forEach((exp, i) => {
            const confPct = Math.round(exp.confidence * 100);
            const item = document.createElement('div');
            item.className = 'explanation-item';
            item.innerHTML = `
                <div class="exp-header">
                    <div class="exp-title">
                        <span class="exp-num">#${i + 1}</span>
                        <span class="exp-class">${exp.class_name.replace(/_/g, ' ')}</span>
                    </div>
                    <span class="conf-badge ${confPct >= 50 ? 'conf-high' : 'conf-low'}">
                        Confidence: ${confPct}%
                    </span>
                </div>
                <div class="conf-mini-bar">
                    <div class="conf-mini-fill" style="width:${confPct}%"></div>
                </div>
                <p class="exp-text">${exp.explanation}</p>`;
            list.appendChild(item);
        });
    }

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
