/**
 * CellCount - Main application controller
 * Manages global state and coordinates all modules.
 */

const App = (() => {
    // ── Global State ────────────────────────────────────────────────
    const state = {
        cameraRunning: false,
        cameraWasRunning: false,  // track if camera was live before capture
        currentImageId: null,
        currentImageUrl: null,
        currentImageSize: null, // {width, height}
        currentDetections: [],
        currentSummary: null,
        currentGrid: null,        // grid detection result
        currentFiltered: null,    // filtered detection result
        annotateMode: false,
        annotateClass: 0, // 0=viable, 1=non_viable
        confThreshold: 0.25,
        session: null,
        activeSampleId: null,
        hasPreciseModel: false,
        gridSelectMode: false,    // manual grid selection active
        gridSelectStart: null,    // {x, y} in image coords
        manualGrid: null,         // user-set grid override (persists across re-analyze)
        gridSelectPoints: [],     // 3-point grid selection clicks
        liveTracking: false,      // live cell tracking on camera feed
    };

    // ── DOM Refs ────────────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const canvas = $('#mainCanvas');
    const ctx = canvas.getContext('2d');
    const placeholder = $('#canvasPlaceholder');
    const container = $('#canvasContainer');

    // ── Toast ───────────────────────────────────────────────────────
    let toastTimer = null;
    function toast(msg, durationMs = 2000) {
        const el = $('#toast');
        el.textContent = msg;
        el.classList.add('visible');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => el.classList.remove('visible'), durationMs);
    }

    // ── Canvas Drawing ──────────────────────────────────────────────
    let loadedImage = null;
    let displayScale = 1;
    let offsetX = 0, offsetY = 0;
    let displayW = 0, displayH = 0;

    function showCanvas() {
        canvas.classList.remove('hidden');
        placeholder.classList.add('hidden');
    }

    function hideCanvas() {
        canvas.classList.add('hidden');
        placeholder.classList.remove('hidden');
    }

    function fitCanvas() {
        const rect = container.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
    }

    function drawFrame(imageBitmap) {
        fitCanvas();
        const cw = canvas.width;
        const ch = canvas.height;
        const iw = imageBitmap.width;
        const ih = imageBitmap.height;

        // Fit image maintaining aspect ratio (letterbox)
        const scale = Math.min(cw / iw, ch / ih);
        displayW = Math.floor(iw * scale);
        displayH = Math.floor(ih * scale);
        offsetX = Math.floor((cw - displayW) / 2);
        offsetY = Math.floor((ch - displayH) / 2);
        displayScale = scale;

        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, cw, ch);
        ctx.drawImage(imageBitmap, offsetX, offsetY, displayW, displayH);

        state.currentImageSize = { width: iw, height: ih };
    }

    function drawDetections() {
        if (!state.currentDetections.length && !Annotator.getAdditions().length) return;

        const removals = new Set(Annotator.getRemovals());
        // Build set of excluded (outside-grid) detection IDs
        const excludedIds = new Set();
        if (state.currentFiltered && state.currentGrid && state.currentGrid.detected) {
            for (const det of (state.currentFiltered.excluded || [])) {
                excludedIds.add(det.id);
            }
        }

        for (const det of state.currentDetections) {
            const [x1, y1, x2, y2] = det.bbox;
            const sx = x1 * displayScale + offsetX;
            const sy = y1 * displayScale + offsetY;
            const sw = (x2 - x1) * displayScale;
            const sh = (y2 - y1) * displayScale;

            // Outside-grid detections: dim grey-yellow
            if (excludedIds.has(det.id)) {
                ctx.globalAlpha = 0.4;
                ctx.strokeStyle = '#c8b832';
                ctx.lineWidth = 1;
                ctx.strokeRect(sx, sy, sw, sh);
                ctx.globalAlpha = 1;
                continue;
            }

            if (removals.has(det.id)) {
                // Strikethrough removed detections
                ctx.globalAlpha = 0.3;
                ctx.strokeStyle = '#666';
                ctx.lineWidth = 1;
                ctx.strokeRect(sx, sy, sw, sh);
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.lineTo(sx + sw, sy + sh);
                ctx.stroke();
                ctx.globalAlpha = 1;
                continue;
            }

            const color = det.class === 0 ? '#48c78e' : '#ef4444';
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(sx, sy, sw, sh);
        }

        // Manual additions
        for (const ann of Annotator.getAdditions()) {
            const cx = ann.x * displayScale + offsetX;
            const cy = ann.y * displayScale + offsetY;
            const color = ann.class === 0 ? '#48c78e' : '#ef4444';

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(cx, cy, 12, 0, Math.PI * 2);
            ctx.stroke();

            // Cross marker
            ctx.beginPath();
            ctx.moveTo(cx - 6, cy);
            ctx.lineTo(cx + 6, cy);
            ctx.moveTo(cx, cy - 6);
            ctx.lineTo(cx, cy + 6);
            ctx.stroke();

            // + label
            ctx.font = '700 10px "JetBrains Mono", monospace';
            ctx.fillStyle = color;
            ctx.fillText('+', cx + 14, cy + 4);
        }
    }

    function drawGridOverlay() {
        const grid = state.currentGrid;
        if (!grid || !grid.detected || !grid.boundary) return;

        const angle = (grid.rotation_deg || 0) * Math.PI / 180;
        const cx = grid.grid_center ? grid.grid_center[0] : (grid.boundary[0] + grid.boundary[2]) / 2;
        const cy = grid.grid_center ? grid.grid_center[1] : (grid.boundary[1] + grid.boundary[3]) / 2;
        const halfW = grid.grid_size ? grid.grid_size[0] / 2 : (grid.boundary[2] - grid.boundary[0]) / 2;
        const halfH = grid.grid_size ? grid.grid_size[1] / 2 : (grid.boundary[3] - grid.boundary[1]) / 2;

        ctx.save();
        // Transform to canvas coords
        ctx.translate(offsetX, offsetY);
        ctx.scale(displayScale, displayScale);

        // Outer boundary (rotated rectangle)
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(angle);
        ctx.strokeStyle = 'rgba(0, 180, 255, 0.7)';
        ctx.lineWidth = 2 / displayScale;
        ctx.strokeRect(-halfW, -halfH, halfW * 2, halfH * 2);

        // Inner grid lines
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.4)';
        ctx.lineWidth = 1 / displayScale;

        // Horizontal lines (relative to center)
        for (const yVal of (grid.horizontal_lines || [])) {
            const dy = yVal - cy;
            ctx.beginPath();
            ctx.moveTo(-halfW, dy);
            ctx.lineTo(halfW, dy);
            ctx.stroke();
        }
        // Vertical lines (relative to center)
        for (const xVal of (grid.vertical_lines || [])) {
            const dx = xVal - cx;
            ctx.beginPath();
            ctx.moveTo(dx, -halfH);
            ctx.lineTo(dx, halfH);
            ctx.stroke();
        }
        ctx.restore();
        ctx.restore();
    }

    function drawGridSelectPoints() {
        if (!state.gridSelectMode || !state.gridSelectPoints.length) return;
        const pts = state.gridSelectPoints;
        const toCanvas = (p) => ({
            x: p.x * displayScale + offsetX,
            y: p.y * displayScale + offsetY,
        });

        ctx.strokeStyle = '#ff9800';
        ctx.fillStyle = '#ff9800';
        ctx.lineWidth = 2;

        // Draw placed points as dots
        for (const p of pts) {
            const cp = toCanvas(p);
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, 5, 0, Math.PI * 2);
            ctx.fill();
        }

        // Draw lines between placed points
        if (pts.length >= 2) {
            const c0 = toCanvas(pts[0]);
            const c1 = toCanvas(pts[1]);
            ctx.beginPath();
            ctx.moveTo(c0.x, c0.y);
            ctx.lineTo(c1.x, c1.y);
            ctx.stroke();
        }

        // If we have a hover preview (mouse position), show projected shape
        if (state._gridSelectHover && pts.length >= 1 && pts.length < 3) {
            const hover = toCanvas(state._gridSelectHover);
            ctx.setLineDash([4, 4]);
            if (pts.length === 1) {
                // Line from P1 to hover
                const c0 = toCanvas(pts[0]);
                ctx.beginPath();
                ctx.moveTo(c0.x, c0.y);
                ctx.lineTo(hover.x, hover.y);
                ctx.stroke();
            } else if (pts.length === 2) {
                // Show projected quadrilateral: P2->hover, hover->P4preview, P4preview->P1
                const c1 = toCanvas(pts[1]);
                const p4 = {
                    x: pts[0].x + state._gridSelectHover.x - pts[1].x,
                    y: pts[0].y + state._gridSelectHover.y - pts[1].y,
                };
                const c4 = toCanvas(p4);
                ctx.beginPath();
                ctx.moveTo(c1.x, c1.y);
                ctx.lineTo(hover.x, hover.y);
                ctx.lineTo(c4.x, c4.y);
                ctx.lineTo(toCanvas(pts[0]).x, toCanvas(pts[0]).y);
                ctx.stroke();
            }
            ctx.setLineDash([]);
            // Hover dot
            ctx.globalAlpha = 0.5;
            ctx.beginPath();
            ctx.arc(hover.x, hover.y, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 1;
        }

        // If all 3 points placed, draw the complete quadrilateral
        if (pts.length === 3) {
            const p4 = {
                x: pts[0].x + pts[2].x - pts[1].x,
                y: pts[0].y + pts[2].y - pts[1].y,
            };
            const corners = [pts[0], pts[1], pts[2], p4].map(toCanvas);
            ctx.beginPath();
            ctx.moveTo(corners[0].x, corners[0].y);
            for (let i = 1; i < 4; i++) ctx.lineTo(corners[i].x, corners[i].y);
            ctx.closePath();
            ctx.stroke();
        }

        // Label
        const labelPt = pts.length > 0 ? toCanvas(pts[0]) : null;
        if (labelPt) {
            const remaining = 3 - pts.length;
            const msg = remaining > 0 ? `Click ${remaining} more corner${remaining > 1 ? 's' : ''}` : 'Processing...';
            ctx.font = '600 12px "JetBrains Mono", monospace';
            ctx.fillStyle = '#ff9800';
            ctx.fillText(msg, labelPt.x, labelPt.y - 10);
        }
    }

    function redraw() {
        if (!loadedImage) return;
        drawFrame(loadedImage);
        drawGridOverlay();
        drawDetections();
        drawGridSelectPoints();
    }

    // ── Image Loading ───────────────────────────────────────────────
    async function loadImageFromUrl(url) {
        const resp = await fetch(url);
        const blob = await resp.blob();
        loadedImage = await createImageBitmap(blob);
        showCanvas();
        redraw();
    }

    // ── API Helpers ─────────────────────────────────────────────────
    async function api(method, path, body = null) {
        const opts = { method };
        if (body instanceof FormData) {
            opts.body = body;
        } else if (body) {
            opts.headers = { 'Content-Type': 'application/json' };
            opts.body = JSON.stringify(body);
        }
        const resp = await fetch(path, opts);
        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(err);
        }
        const ct = resp.headers.get('content-type') || '';
        if (ct.includes('json')) return resp.json();
        return resp;
    }

    // ── Settings Dialog ─────────────────────────────────────────────
    function openSettings() {
        // Populate calibration fields from session state
        const cal = state.session?.calibration;
        if (cal) {
            $('#calPixPerMm').value = cal.pixels_per_mm ?? 0;
            $('#calDilution').value = cal.dilution_factor ?? 1;
            $('#calGridSide').value = cal.grid_square_side_mm ?? 1.0;
            $('#calTrypanBlue').checked = cal.trypan_blue_dilution !== false;
        }
        updateEffectiveDilution();
        $('#settingsOverlay').classList.add('visible');
    }

    function updateEffectiveDilution() {
        const factor = parseInt($('#calDilution').value) || 1;
        const trypan = $('#calTrypanBlue').checked;
        const effective = factor * (trypan ? 2 : 1);
        $('#calEffectiveDilution').textContent = `Effective dilution: ${effective}x` +
            (trypan ? ` (${factor}x sample \u00D7 2x trypan blue)` : '');
    }

    function closeSettings() {
        $('#settingsOverlay').classList.remove('visible');
    }

    function switchSettingsTab(tabName) {
        document.querySelectorAll('.modal-tabs .tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tabName);
        });
        document.querySelectorAll('.tab-content').forEach(tc => {
            tc.classList.toggle('active', tc.id === `tab-${tabName}`);
        });
    }

    // ── Export Dropdown ─────────────────────────────────────────────
    function toggleExportMenu() {
        $('#exportMenu').classList.toggle('visible');
    }

    function closeExportMenu() {
        $('#exportMenu').classList.remove('visible');
    }

    // ── Actions (called by shortcuts, buttons, etc.) ────────────────
    const actions = {
        async startCamera() {
            try {
                const deviceId = parseInt($('#cameraDeviceSelect').value) || 0;
                const result = await api('POST', `/api/camera/start?device_id=${deviceId}`);
                state.cameraRunning = true;
                Camera.connect();
                // Sync live tracking state with camera
                if (state.liveTracking) {
                    Camera.sendCommand({ live_tracking: true });
                }
                updateCameraUI(true, result.backend);
                toast('Camera started');
            } catch (e) {
                toast('Failed to start camera');
            }
        },

        async stopCamera() {
            Camera.disconnect();
            await api('POST', '/api/camera/stop').catch(() => {});
            state.cameraRunning = false;
            updateCameraUI(false);
            // Show black screen (clear canvas to black)
            showCanvas();
            fitCanvas();
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            loadedImage = null;
            toast('Camera stopped');
        },

        toggleCamera() {
            if (state.cameraRunning) actions.stopCamera();
            else actions.startCamera();
        },

        async capture() {
            if (!state.cameraRunning) return;
            try {
                state.cameraWasRunning = true;
                Camera.pause();  // pause feed while we capture
                state.manualGrid = null;  // fresh image = fresh grid detection

                // Capture frame while camera is still running on backend
                const result = await api('POST', '/api/capture');
                state.currentImageId = result.image_id;
                state.currentImageUrl = result.image_url;

                // Now stop the camera fully
                Camera.disconnect();
                await api('POST', '/api/camera/stop').catch(() => {});
                state.cameraRunning = false;
                updateCameraUI(false);

                // Display captured image (persists on screen)
                await loadImageFromUrl(result.image_url);
                // Quick count with nano first
                await actions.analyze(false);
                $('#captureBtn').classList.add('pulse');
                setTimeout(() => $('#captureBtn').classList.remove('pulse'), 300);
                // Auto-refine with medium model if available
                if (state.hasPreciseModel) {
                    await actions.analyze(true);
                }
            } catch (e) {
                toast('Capture failed');
                Camera.resume();
            }
        },

        async analyze(usePrecise = false) {
            if (!state.currentImageId) return;
            try {
                const params = `image_id=${state.currentImageId}&conf=${state.confThreshold}&use_precise=${usePrecise}`;
                const body = state.manualGrid ? { override_grid: state.manualGrid } : undefined;
                const result = await api('POST', `/api/analyze?${params}`, body);
                state.currentDetections = result.detections;
                state.currentSummary = result.summary;
                state.currentGrid = result.grid || null;
                state.currentFiltered = result.filtered || null;
                Annotator.reset();

                // Use filtered summary if grid was detected, otherwise raw summary
                const displaySummary = (result.filtered && result.grid && result.grid.detected)
                    ? result.filtered.summary : result.summary;
                updateResults(displaySummary, result.filtered);
                Charts.updateViability(displaySummary.viable, displaySummary.non_viable);
                const modelTag = result.model_used === 'precise' ? ' (precise)' : '';
                $('#inferenceTime').textContent = `${result.inference_ms}ms${modelTag}`;
                $('#saveSampleBtn').disabled = false;
                redraw();

                // Image quality warnings
                showImageQualityWarnings(result.image_quality);

                // Grid detection notification
                if (result.grid && !result.grid.detected) {
                    showGridFailureNotification();
                    toast(`Detected ${result.summary.total} cells (grid not found)`);
                } else if (result.grid && result.grid.confidence < 0.5) {
                    showGridFailureNotification();
                    toast(`Detected ${displaySummary.total} cells (low grid confidence)`);
                } else {
                    hideGridFailureNotification();
                    toast(`Detected ${displaySummary.total} cells`);
                }
            } catch (e) {
                toast('Analysis failed');
            }
        },

        async analyzeUpload(file) {
            state.manualGrid = null;  // fresh image = fresh grid detection
            const fd = new FormData();
            fd.append('file', file);
            fd.append('conf', state.confThreshold);
            try {
                const result = await api('POST', '/api/analyze/upload', fd);
                state.currentImageId = result.image_id;
                state.currentImageUrl = result.image_url;
                state.currentDetections = result.detections;
                state.currentSummary = result.summary;
                state.currentGrid = result.grid || null;
                state.currentFiltered = result.filtered || null;
                Annotator.reset();
                await loadImageFromUrl(result.image_url);

                const displaySummary = (result.filtered && result.grid && result.grid.detected)
                    ? result.filtered.summary : result.summary;
                updateResults(displaySummary, result.filtered);
                Charts.updateViability(displaySummary.viable, displaySummary.non_viable);
                $('#inferenceTime').textContent = `${result.inference_ms}ms`;
                $('#saveSampleBtn').disabled = false;
                redraw();

                showImageQualityWarnings(result.image_quality);

                if (result.grid && !result.grid.detected) {
                    showGridFailureNotification();
                    toast(`Detected ${result.summary.total} cells (grid not found)`);
                } else if (result.grid && result.grid.confidence < 0.5) {
                    showGridFailureNotification();
                    toast(`Detected ${displaySummary.total} cells (low grid confidence)`);
                } else {
                    hideGridFailureNotification();
                    toast(`Detected ${displaySummary.total} cells`);
                }

                // Auto-refine with precise model if available (match capture behavior)
                if (state.hasPreciseModel) {
                    await actions.analyze(true);
                }
            } catch (e) {
                toast('Upload analysis failed');
            }
        },

        async reAnalyze() {
            if (!state.currentImageId) return;
            // R key always uses the best available model
            await actions.analyze(state.hasPreciseModel);
        },

        toggleAnnotate() {
            state.annotateMode = !state.annotateMode;
            $('#annotateToggle').checked = state.annotateMode;
            container.classList.toggle('annotating', state.annotateMode);
            const hint = $('#annotateHint');
            if (hint) hint.classList.toggle('hidden', !state.annotateMode);
        },

        setAnnotateOff() {
            state.annotateMode = false;
            $('#annotateToggle').checked = false;
            container.classList.remove('annotating');
            const hint = $('#annotateHint');
            if (hint) hint.classList.add('hidden');
        },

        async saveSample() {
            if (!state.currentImageId || !state.currentSummary) return;
            const baseSummary = getDisplayBaseSummary();
            const excludedIds = getExcludedDetectionIds();
            const eff = Annotator.getEffectiveSummary(baseSummary, excludedIds);
            const gridInfo = state.currentGrid || {};
            const filteredSummary = state.currentFiltered?.summary || {};

            // Check if this is an existing sample (re-save) or a new one
            const isExisting = state.activeSampleId === state.currentImageId;

            try {
                if (isExisting) {
                    // Update existing sample via PATCH
                    await api('PATCH', `/api/session/sample/${state.currentImageId}`, {
                        detections: state.currentDetections,
                        summary: eff,
                        conf_threshold: state.confThreshold,
                        grid_info: gridInfo,
                        filtered_summary: filteredSummary,
                        additions: Annotator.getAdditions(),
                        removals: Annotator.getRemovals(),
                    });
                } else {
                    // Create new sample via POST
                    await api('POST', '/api/session/sample', {
                        image_id: state.currentImageId,
                        detections: state.currentDetections,
                        summary: eff,
                        conf_threshold: state.confThreshold,
                        grid_info: gridInfo,
                        filtered_summary: filteredSummary,
                    });
                    // Also persist annotations if any
                    if (Annotator.getAdditions().length || Annotator.getRemovals().length) {
                        await api('PATCH', `/api/annotations/${state.currentImageId}`, {
                            additions: Annotator.getAdditions(),
                            removals: Annotator.getRemovals(),
                        });
                    }
                }
                await Session.refresh();
                toast(isExisting ? 'Sample updated' : `Sample #${Session.getSampleCount()} saved`);
                $('#saveSampleBtn').disabled = true;

                // Resume camera if it was running before capture
                if (state.cameraWasRunning) {
                    state.cameraWasRunning = false;
                    await actions.startCamera();
                }
            } catch (e) {
                console.error('Save sample failed:', e);
                toast('Failed to save sample');
            }
        },

        async exportCSV() {
            closeExportMenu();
            try {
                const resp = await fetch('/api/session/export');
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${$('#experimentName').value}.csv`;
                a.click();
                URL.revokeObjectURL(url);
                toast('CSV exported');
            } catch (e) {
                Export.exportClientSide();
            }
        },

        async exportZip() {
            closeExportMenu();
            try {
                toast('Preparing ZIP...');
                const resp = await fetch('/api/session/export/zip');
                if (!resp.ok) throw new Error('ZIP export failed');
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${$('#experimentName').value}.zip`;
                a.click();
                URL.revokeObjectURL(url);
                toast('ZIP downloaded');
            } catch (e) {
                toast('ZIP export failed');
            }
        },

        exportImage() {
            closeExportMenu();
            Export.downloadAnnotatedImage();
        },

        async newSampleGroup() {
            try {
                const result = await api('POST', '/api/session/group/new');
                await Session.refresh();
                toast(`${result.name} started`);
            } catch (e) {
                toast('Failed to create sample group');
            }
        },

        async newSession() {
            if (!confirm('Start a new session?\n\nAll current samples and images will be cleared. Export your data first if you want to keep it.')) return;
            try {
                await api('POST', '/api/session/new');
                await Session.refresh();
                state.currentImageId = null;
                state.currentDetections = [];
                state.currentSummary = null;
                state.currentGrid = null;
                state.currentFiltered = null;
                Annotator.reset();
                hideCanvas();
                updateResults(null);
                Charts.updateViability(0, 0);
                $('#saveSampleBtn').disabled = true;
                toast('New session started');
            } catch (e) {
                toast('Failed to create session');
            }
        },

        async uploadModel(file) {
            const fd = new FormData();
            fd.append('file', file);
            try {
                const result = await api('POST', '/api/model/upload', fd);
                // Switch to uploaded model
                await api('POST', `/api/model/select?model_path=${encodeURIComponent(result.path)}`);
                $('#modelName').textContent = result.model;
                toast(`Model switched to ${result.model}`);
            } catch (e) {
                toast('Failed to upload model');
            }
        },

        escape() {
            // Cancel grid selection first
            if (state.gridSelectMode) {
                endGridSelection();
                redraw();
                return;
            }
            // Close export menu first
            if ($('#exportMenu').classList.contains('visible')) {
                closeExportMenu();
                return;
            }
            // Close settings dialog
            if ($('#settingsOverlay').classList.contains('visible')) {
                closeSettings();
                return;
            }
            // Close help overlay
            if ($('#helpOverlay').classList.contains('visible')) {
                $('#helpOverlay').classList.remove('visible');
                return;
            }
            // Exit annotate mode
            if (state.annotateMode) {
                actions.setAnnotateOff();
            }
        },

        toggleHelp() {
            $('#helpOverlay').classList.toggle('visible');
        },

        adjustConf(delta) {
            let val = state.confThreshold + delta;
            val = Math.max(0.05, Math.min(0.95, Math.round(val * 100) / 100));
            state.confThreshold = val;
            $('#confValue').textContent = `${Math.round(val * 100)}%`;
            // Debounced re-analyze after adjustment
            clearTimeout(state._confDebounce);
            state._confDebounce = setTimeout(() => {
                if (state.currentImageId) actions.reAnalyze();
            }, 400);
        },
    };

    // ── UI Updates ──────────────────────────────────────────────────
    function updateCameraUI(connected, backend) {
        const badge = $('#cameraStatusBadge');
        if (connected) {
            badge.className = 'camera-status-badge connected';
            badge.innerHTML = `<span class="dot"></span> ${backend || 'Connected'}`;
            $('#cameraStartBtn').disabled = true;
            $('#cameraStopBtn').disabled = false;
            $('#captureBtn').disabled = false;
        } else {
            badge.className = 'camera-status-badge disconnected';
            badge.innerHTML = '<span class="dot"></span> Off';
            $('#cameraStartBtn').disabled = false;
            $('#cameraStopBtn').disabled = true;
            $('#captureBtn').disabled = true;
        }
    }

    function updateResults(summary, filtered) {
        if (!summary) {
            $('#statTotal').textContent = '--';
            $('#statViable').textContent = '--';
            $('#statDead').textContent = '--';
            $('#donutCenterText').textContent = '--';
            $('#statConc').textContent = 'N/A';
            $('#inferenceTime').textContent = '';
            return;
        }
        $('#statTotal').textContent = summary.total;
        $('#statViable').textContent = summary.viable;
        $('#statDead').textContent = summary.non_viable;

        // Viability percentage shown in donut center
        const pct = summary.viability_pct;
        const donutCenter = $('#donutCenterText');
        donutCenter.textContent = `${pct}%`;
        if (pct >= 80) donutCenter.style.color = 'var(--color-viable)';
        else if (pct >= 60) donutCenter.style.color = 'var(--color-warning)';
        else donutCenter.style.color = 'var(--color-dead)';

        // Concentration for current image: cells/mL = total × dilution × 10,000
        // Reads directly from UI controls (per-observation, not persisted)
        const dilution = parseInt($('#mainDilution').value) || 1;
        const trypan = $('#mainTrypanBlue').checked;
        const effectiveDilution = dilution * (trypan ? 2 : 1);
        const conc = summary.total * effectiveDilution * 10000;
        if (conc >= 1e6) {
            $('#statConc').textContent = `${(conc / 1e6).toFixed(2)} \u00D7 10\u2076 /mL`;
        } else {
            $('#statConc').textContent = `${Math.round(conc).toLocaleString()} /mL`;
        }
    }

    /** Get the base summary to use for annotation adjustments.
     *  When a grid is detected, use the filtered (inside-grid) summary;
     *  otherwise fall back to the raw (all detections) summary.
     */
    function getDisplayBaseSummary() {
        if (state.currentFiltered && state.currentGrid && state.currentGrid.detected) {
            return state.currentFiltered.summary;
        }
        return state.currentSummary;
    }

    /** Get the set of detection IDs that are excluded (outside grid).
     *  Used by getEffectiveSummary to ignore removals of outside-grid detections.
     */
    function getExcludedDetectionIds() {
        if (state.currentFiltered && state.currentGrid && state.currentGrid.detected) {
            return new Set((state.currentFiltered.excluded || []).map(d => d.id));
        }
        return new Set();
    }

    function onAnnotationChanged() {
        if (!state.currentSummary) return;
        const baseSummary = getDisplayBaseSummary();
        const excludedIds = getExcludedDetectionIds();
        const eff = Annotator.getEffectiveSummary(baseSummary, excludedIds);
        updateResults(eff, state.currentFiltered);
        Charts.updateViability(eff.viable, eff.non_viable);
        redraw();
    }

    // ── Grid Detection Notification ─────────────────────────────────
    function showGridFailureNotification() {
        let banner = $('#gridFailureBanner');
        if (!banner) {
            banner = document.createElement('div');
            banner.id = 'gridFailureBanner';
            banner.className = 'grid-failure-banner';
            banner.innerHTML = `
                <span class="grid-failure-icon">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                </span>
                <span>Grid detection failed. </span>
                <button class="grid-select-btn" id="gridSelectBtn">Select grid manually</button>
                <button class="grid-dismiss-btn" id="gridDismissBtn">&times;</button>
            `;
            container.appendChild(banner);
            $('#gridSelectBtn').addEventListener('click', startGridSelection);
            $('#gridDismissBtn').addEventListener('click', hideGridFailureNotification);
        }
        banner.classList.add('visible');
    }

    function hideGridFailureNotification() {
        const banner = $('#gridFailureBanner');
        if (banner) banner.classList.remove('visible');
    }

    // ── Image Quality Warnings ──────────────────────────────────────
    function showImageQualityWarnings(quality) {
        // Remove any existing quality banner
        const existing = $('#qualityBanner');
        if (existing) existing.remove();

        if (!quality || !quality.warnings || quality.warnings.length === 0) return;

        const banner = document.createElement('div');
        banner.id = 'qualityBanner';
        banner.className = 'quality-banner';

        const hasError = quality.warnings.some(w => w.severity === 'error');
        banner.classList.add(hasError ? 'severity-error' : 'severity-warning');

        const msgs = quality.warnings.map(w => w.message);
        // Build banner using DOM methods to avoid innerHTML XSS
        const icon = document.createElement('span');
        icon.className = 'quality-icon';
        icon.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
        const msgSpan = document.createElement('span');
        msgSpan.className = 'quality-msgs';
        msgSpan.textContent = msgs.join(' ');
        const dismissBtn = document.createElement('button');
        dismissBtn.className = 'grid-dismiss-btn';
        dismissBtn.textContent = '\u00D7';
        dismissBtn.addEventListener('click', () => banner.remove());
        banner.appendChild(icon);
        banner.appendChild(msgSpan);
        banner.appendChild(dismissBtn);
        container.appendChild(banner);

        // Auto-dismiss after 8 seconds for warnings, persist for errors
        if (!hasError) {
            setTimeout(() => { if (banner.parentElement) banner.remove(); }, 8000);
        }
    }

    /** Client-side detection filtering against grid boundary.
     *  Mirrors the server-side CellFilter logic so loaded samples
     *  can show inside/excluded detections without a server round-trip. */
    function recomputeFiltered(detections, grid) {
        if (!grid || !grid.detected || !grid.boundary) return null;
        const cx = grid.grid_center ? grid.grid_center[0] : (grid.boundary[0] + grid.boundary[2]) / 2;
        const cy = grid.grid_center ? grid.grid_center[1] : (grid.boundary[1] + grid.boundary[3]) / 2;
        const halfW = grid.grid_size ? grid.grid_size[0] / 2 : (grid.boundary[2] - grid.boundary[0]) / 2;
        const halfH = grid.grid_size ? grid.grid_size[1] / 2 : (grid.boundary[3] - grid.boundary[1]) / 2;
        const angle = (grid.rotation_deg || 0) * Math.PI / 180;
        const cosA = Math.cos(-angle);
        const sinA = Math.sin(-angle);

        const inside = [], excluded = [];
        for (const det of detections) {
            const [bx1, by1, bx2, by2] = det.bbox;
            const dx = (bx1 + bx2) / 2 - cx;
            const dy = (by1 + by2) / 2 - cy;
            const lx = dx * cosA - dy * sinA;
            const ly = dx * sinA + dy * cosA;
            if (Math.abs(lx) > halfW || Math.abs(ly) > halfH) {
                excluded.push(det);
            } else {
                inside.push(det);
            }
        }
        const v = inside.filter(d => d.class === 0).length;
        const nv = inside.filter(d => d.class === 1).length;
        const total = v + nv;
        return {
            detections: inside,
            excluded,
            summary: {
                total,
                viable: v,
                non_viable: nv,
                viability_pct: total > 0 ? Math.round(v / total * 1000) / 10 : 0,
            },
        };
    }

    function startGridSelection() {
        hideGridFailureNotification();
        state.gridSelectMode = true;
        state.gridSelectPoints = [];
        state._gridSelectHover = null;
        container.classList.add('grid-selecting');
        toast('Click 3 corners of the grid square (in order)', 5000);
    }

    function endGridSelection() {
        state.gridSelectMode = false;
        state.gridSelectPoints = [];
        state._gridSelectHover = null;
        container.classList.remove('grid-selecting');
    }

    function handleGridSelectMouseMove(e) {
        if (!state.gridSelectMode || state.gridSelectPoints.length >= 3) return;
        const rect = canvas.getBoundingClientRect();
        state._gridSelectHover = {
            x: (e.clientX - rect.left - offsetX) / displayScale,
            y: (e.clientY - rect.top - offsetY) / displayScale,
        };
        redraw();
    }

    async function handleGridSelectClick(e) {
        if (!state.gridSelectMode) return;
        // Ignore right-click
        if (e.button !== 0) return;

        const rect = canvas.getBoundingClientRect();
        const imgX = (e.clientX - rect.left - offsetX) / displayScale;
        const imgY = (e.clientY - rect.top - offsetY) / displayScale;

        // Bounds check
        if (imgX < 0 || imgY < 0 || displayScale <= 0) return;

        state.gridSelectPoints.push({ x: Math.round(imgX), y: Math.round(imgY) });
        redraw();

        if (state.gridSelectPoints.length < 3) return;

        // All 3 points collected — send to backend
        const pts = state.gridSelectPoints;
        const points = [[pts[0].x, pts[0].y], [pts[1].x, pts[1].y], [pts[2].x, pts[2].y]];

        try {
            const result = await api('POST', '/api/grid/manual', {
                image_id: state.currentImageId,
                points,
            });
            state.currentGrid = result.grid;
            state.manualGrid = result.grid;  // persist across re-analyze
            if (state.session) state.session.calibration = result.calibration;

            // Re-filter detections with the new grid
            if (state.currentDetections.length) {
                state.currentFiltered = recomputeFiltered(state.currentDetections, result.grid);
                const baseSummary = state.currentFiltered ? state.currentFiltered.summary : state.currentSummary;
                const excludedIds = getExcludedDetectionIds();
                const eff = Annotator.getEffectiveSummary(baseSummary, excludedIds);
                updateResults(eff, state.currentFiltered);
                Charts.updateViability(eff.viable, eff.non_viable);
            }

            endGridSelection();
            redraw();
            toast(`Grid set manually: ${result.grid.pixels_per_mm} px/mm`);
            // Enable save for re-saving
            $('#saveSampleBtn').disabled = false;
        } catch (err) {
            console.error('Manual grid failed:', err);
            endGridSelection();
            toast('Failed to set manual grid');
        }
    }

    // ── Initialization ──────────────────────────────────────────────
    function init() {
        // Card collapse toggles
        document.querySelectorAll('[data-collapsible]').forEach(header => {
            header.addEventListener('click', () => {
                header.closest('.card').classList.toggle('collapsed');
            });
        });

        // Confidence +/- buttons
        $('#confDown').addEventListener('click', () => actions.adjustConf(-0.05));
        $('#confUp').addEventListener('click', () => actions.adjustConf(0.05));

        // Camera buttons
        $('#cameraStartBtn').addEventListener('click', actions.startCamera);
        $('#cameraStopBtn').addEventListener('click', actions.stopCamera);
        $('#captureBtn').addEventListener('click', actions.capture);

        // Grid override button
        $('#gridOverrideBtn').addEventListener('click', startGridSelection);

        // Image upload
        $('#imageUpload').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) actions.analyzeUpload(file);
            e.target.value = '';
        });

        // Model upload (in settings dialog)
        $('#modelUpload').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) actions.uploadModel(file);
            e.target.value = '';
        });

        // Annotate toggle
        $('#annotateToggle').addEventListener('change', () => {
            state.annotateMode = $('#annotateToggle').checked;
            container.classList.toggle('annotating', state.annotateMode);
            const hint = $('#annotateHint');
            if (hint) hint.classList.toggle('hidden', !state.annotateMode);
        });

        // Live tracking toggle
        $('#liveTrackToggle').addEventListener('change', () => {
            state.liveTracking = $('#liveTrackToggle').checked;
            Camera.sendCommand({ live_tracking: state.liveTracking });
        });

        // Calibration save (in settings dialog)
        $('#calSaveBtn').addEventListener('click', async () => {
            try {
                const params = new URLSearchParams({
                    pixels_per_mm: $('#calPixPerMm').value,
                    dilution_factor: $('#calDilution').value,
                    squares_counted: 1,
                    grid_square_side_mm: $('#calGridSide').value,
                    trypan_blue_dilution: $('#calTrypanBlue').checked,
                });
                const result = await api('POST', `/api/calibration?${params}`);
                if (state.session) state.session.calibration = result;
                // Sync main UI controls from settings
                $('#mainDilution').value = $('#calDilution').value;
                $('#mainTrypanBlue').checked = $('#calTrypanBlue').checked;
                if (state.currentSummary) {
                    const baseSummary = getDisplayBaseSummary();
                    const excludedIds = getExcludedDetectionIds();
                    const eff = Annotator.getEffectiveSummary(baseSummary, excludedIds);
                    updateResults(eff, state.currentFiltered);
                }
                toast('Calibration saved');
            } catch (e) {
                toast('Failed to save calibration');
            }
        });

        // Update effective dilution display when inputs change
        $('#calDilution').addEventListener('input', updateEffectiveDilution);
        $('#calTrypanBlue').addEventListener('change', updateEffectiveDilution);

        // Main UI dilution/trypan blue controls — local only, recalculate display
        const mainDilution = $('#mainDilution');
        const mainTrypan = $('#mainTrypanBlue');
        function onMainCalChange() {
            // Recalculate displayed results with current params
            if (state.currentSummary) {
                const baseSummary = getDisplayBaseSummary();
                const excludedIds = getExcludedDetectionIds();
                const eff = Annotator.getEffectiveSummary(baseSummary, excludedIds);
                updateResults(eff, state.currentFiltered);
            }
        }
        mainDilution.addEventListener('change', onMainCalChange);
        mainTrypan.addEventListener('change', onMainCalChange);

        // Sample group controls
        $('#newSampleGroupBtn').addEventListener('click', actions.newSampleGroup);
        $('#activeSampleName').addEventListener('change', async (e) => {
            const activeGroupId = state.session?.active_group_id;
            if (activeGroupId) {
                try {
                    await api('POST', '/api/session/group/rename', {
                        group_id: activeGroupId,
                        name: e.target.value,
                    });
                    await Session.refresh();
                } catch (err) { /* ignore */ }
            }
        });

        // Settings dialog (button removed but dialog kept for future use)
        const settingsBtn = $('#settingsBtn');
        if (settingsBtn) settingsBtn.addEventListener('click', openSettings);
        $('#settingsClose').addEventListener('click', closeSettings);
        $('#settingsOverlay').addEventListener('click', (e) => {
            if (e.target === e.currentTarget) closeSettings();
        });
        document.querySelectorAll('.modal-tabs .tab').forEach(tab => {
            tab.addEventListener('click', () => switchSettingsTab(tab.dataset.tab));
        });

        // Export dropdown
        $('#exportBtn').addEventListener('click', (e) => {
            e.stopPropagation();
            toggleExportMenu();
        });
        $('#exportCsvBtn').addEventListener('click', actions.exportCSV);
        $('#exportZipBtn').addEventListener('click', actions.exportZip);
        $('#exportImageBtn').addEventListener('click', actions.exportImage);

        // Close export menu on outside click
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.export-dropdown')) {
                closeExportMenu();
            }
        });

        // Action buttons
        $('#saveSampleBtn').addEventListener('click', actions.saveSample);
        $('#newSessionBtn').addEventListener('click', actions.newSession);

        // Experiment name
        $('#experimentName').addEventListener('change', async (e) => {
            try {
                await api('POST', `/api/session/rename?name=${encodeURIComponent(e.target.value)}`);
            } catch (err) { /* ignore */ }
        });

        // Help button
        $('#helpBtn').addEventListener('click', actions.toggleHelp);

        // Drag and drop
        const cc = container;
        cc.addEventListener('dragover', (e) => { e.preventDefault(); $('#dropOverlay').classList.add('active'); });
        cc.addEventListener('dragleave', () => { $('#dropOverlay').classList.remove('active'); });
        cc.addEventListener('drop', (e) => {
            e.preventDefault();
            $('#dropOverlay').classList.remove('active');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) actions.analyzeUpload(file);
        });

        // Placeholder click
        placeholder.addEventListener('click', () => {
            $('#imageUpload').click();
        });

        // Manual grid selection on canvas (3-point click)
        canvas.addEventListener('click', handleGridSelectClick);
        canvas.addEventListener('mousemove', handleGridSelectMouseMove);

        // Window resize
        window.addEventListener('resize', () => {
            if (loadedImage) redraw();
        });

        // Load session
        Session.refresh();

        // Load model info
        api('GET', '/api/model/info').then(info => {
            state.hasPreciseModel = info.has_precise_model || false;
            const label = info.has_precise_model
                ? `${info.model} (live) / ${info.precise_model} (precise)`
                : info.model;
            $('#modelName').textContent = label;
        }).catch(() => {});

        // Enumerate camera devices
        api('GET', '/api/camera/devices').then(devices => {
            const select = $('#cameraDeviceSelect');
            if (devices && devices.length > 0) {
                select.innerHTML = '';
                devices.forEach(d => {
                    const opt = document.createElement('option');
                    opt.value = d.device_id;
                    opt.textContent = `${d.name} (${d.resolution[0]}x${d.resolution[1]})`;
                    select.appendChild(opt);
                });
            }
        }).catch(() => {});
    }

    document.addEventListener('DOMContentLoaded', init);

    return { state, actions, redraw, toast, loadImageFromUrl, onAnnotationChanged, updateResults, recomputeFiltered, $, canvas, ctx, showCanvas, displayScale: () => displayScale, offsetX: () => offsetX, offsetY: () => offsetY };
})();
