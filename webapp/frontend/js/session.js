/**
 * Session module - Session/experiment UI management
 * Supports sample groups (multiple images per biological sample).
 */
const Session = (() => {
    let currentSession = null;
    let activeSampleIndex = -1;  // flat index across all groups

    async function refresh() {
        try {
            const data = await (await fetch('/api/session/current')).json();
            currentSession = data;
            App.state.session = data;

            // Update experiment name
            document.getElementById('experimentName').value = data.experiment_name;

            // Update sample badge in top bar
            const groupsWithImages = (data.sample_groups || []).filter(g => (g.images || []).length > 0);
            const imgCount = (data.samples || []).length;
            document.getElementById('sampleBadge').textContent =
                `${groupsWithImages.length} sample${groupsWithImages.length !== 1 ? 's' : ''} \u00B7 ${imgCount} img`;

            // Update active sample name field
            const activeGroup = (data.sample_groups || []).find(g => g.group_id === data.active_group_id);
            if (activeGroup) {
                document.getElementById('activeSampleName').value = activeGroup.name;
            }

            // Render grouped samples list
            renderGroups(data.sample_groups || []);

            // Update comparison chart with group-level concentration
            Charts.updateComparison(groupsWithImages, data.calibration);

            // Load calibration into settings form and main UI controls
            if (data.calibration) {
                const calPix = document.getElementById('calPixPerMm');
                if (calPix) calPix.value = data.calibration.pixels_per_mm || 0;
                const calDil = document.getElementById('calDilution');
                if (calDil) calDil.value = data.calibration.dilution_factor || 1;
                const calGrid = document.getElementById('calGridSide');
                if (calGrid) calGrid.value = data.calibration.grid_square_side_mm || 1.0;
                // Sync main UI controls
                document.getElementById('mainDilution').value = data.calibration.dilution_factor || 1;
                const trypan = data.calibration.trypan_blue_dilution !== false;
                document.getElementById('mainTrypanBlue').checked = trypan;
            }
        } catch (e) {
            // Session not available yet
        }
    }

    /** Calculate concentration for a group using calibration settings */
    function groupConcentration(group, cal) {
        const agg = group.aggregate_summary || {};
        const images = group.images || [];
        if (!images.length || !agg.total) return 0;
        const dilution = cal?.dilution_factor || 1;
        const trypan = cal?.trypan_blue_dilution !== false;
        const effectiveDilution = dilution * (trypan ? 2 : 1);
        return (agg.total / images.length) * effectiveDilution * 10000;
    }

    /** Format concentration for display */
    function formatConc(conc) {
        if (!conc) return '--';
        if (conc >= 1e6) return `${(conc / 1e6).toFixed(1)}M`;
        if (conc >= 1e3) return `${(conc / 1e3).toFixed(0)}K`;
        return Math.round(conc).toLocaleString();
    }

    function renderGroups(groups) {
        const container = document.getElementById('samplesContainer');
        const badge = document.getElementById('imageCountBadge');
        const cal = currentSession?.calibration;

        const totalImages = groups.reduce((sum, g) => sum + (g.images || []).length, 0);
        badge.textContent = `${totalImages} image${totalImages !== 1 ? 's' : ''} saved`;

        if (!groups.length || totalImages === 0) {
            container.innerHTML = '<div class="no-samples">No samples yet. Capture or upload an image to begin.</div>';
            return;
        }

        let flatIndex = 0;

        container.innerHTML = groups.map(g => {
            const images = g.images || [];
            if (!images.length) return '';

            const agg = g.aggregate_summary || {};
            const viab = agg.viability_pct != null ? agg.viability_pct : '--';
            let viabColor = 'var(--text-primary)';
            if (viab >= 80) viabColor = 'var(--color-viable)';
            else if (viab >= 60) viabColor = 'var(--color-warning)';
            else if (viab !== '--') viabColor = 'var(--color-dead)';

            const conc = groupConcentration(g, cal);
            const concStr = conc ? `${formatConc(conc)} /mL` : '';

            const isActiveGroup = g.group_id === currentSession?.active_group_id;

            const imagesHtml = images.map((s, imgIdx) => {
                const eff = s.effective_summary || s.summary || {};
                const time = s.timestamp ? s.timestamp.split(' ')[1] || s.timestamp : '';
                const total = eff.total != null ? eff.total : '--';
                const viable = eff.viable != null ? eff.viable : '--';
                const dead = eff.non_viable != null ? eff.non_viable : '--';
                const thumbUrl = s.image_id ? `/api/images/${s.image_id}.jpg` : '';
                const isActive = flatIndex === activeSampleIndex;
                const fi = flatIndex;
                flatIndex++;

                return `<div class="sample-row ${isActive ? 'active' : ''}" data-flat-index="${fi}" data-image-id="${s.image_id}" data-group-id="${g.group_id}">
                    ${thumbUrl ? `<img class="sample-thumb" src="${thumbUrl}" alt="" loading="lazy">` : '<div class="sample-thumb"></div>'}
                    <div class="sample-info">
                        <div class="sample-info-top">
                            <span class="sample-num">#${imgIdx + 1}</span>
                            <span class="sample-time">${time}</span>
                        </div>
                        <div class="sample-info-bottom">
                            <span class="sample-count">${total} cells</span>
                            <span class="sample-viable">${viable}V</span>
                            <span class="sample-dead">${dead}D</span>
                        </div>
                    </div>
                </div>`;
            }).join('');

            return `<div class="sample-group ${isActiveGroup ? 'active-group' : ''}">
                <div class="sample-group-header" data-group-id="${g.group_id}">
                    <div class="group-title">
                        <span class="group-name">${g.name}</span>
                        <span class="group-badges">
                            <span class="group-count">${images.length} img</span>
                            <span class="group-viability" style="color:${viabColor}">${viab}%</span>
                        </span>
                    </div>
                    ${concStr ? `<span class="group-conc">${concStr}</span>` : ''}
                </div>
                <div class="sample-group-images">${imagesHtml}</div>
            </div>`;
        }).join('');

        // Click handler for image rows
        container.querySelectorAll('.sample-row').forEach(row => {
            row.addEventListener('click', () => {
                const fi = parseInt(row.dataset.flatIndex);
                const imageId = row.dataset.imageId;
                selectSample(fi, imageId);
            });
        });
    }

    async function selectSample(flatIndex, imageId) {
        activeSampleIndex = flatIndex;
        App.state.activeSampleId = imageId;

        // Find sample in groups
        let sample = null;
        if (currentSession && currentSession.sample_groups) {
            for (const g of currentSession.sample_groups) {
                for (const img of (g.images || [])) {
                    if (img.image_id === imageId) {
                        sample = img;
                        break;
                    }
                }
                if (sample) break;
            }
        }
        // Fallback to flat samples
        if (!sample && currentSession && currentSession.samples) {
            sample = currentSession.samples.find(s => s.image_id === imageId);
        }
        if (!sample) return;

        // Load the sample image
        try {
            const imageUrl = `/api/images/${imageId}.jpg`;
            await App.loadImageFromUrl(imageUrl);

            // Restore detections
            App.state.currentImageId = imageId;
            App.state.currentDetections = sample.detections || [];
            App.state.currentSummary = sample.summary || sample.effective_summary;
            App.state.currentGrid = sample.grid_info || null;

            // Recompute filtered detections client-side so inside/excluded
            // arrays are available for rendering (grey-yellow outside-grid cells)
            const grid = App.state.currentGrid;
            if (grid && grid.detected && App.state.currentDetections.length) {
                App.state.currentFiltered = App.recomputeFiltered(App.state.currentDetections, grid);
            } else {
                App.state.currentFiltered = null;
            }

            // Restore manual grid if it was user-set (confidence === 1.0 signals manual)
            if (grid && grid.detected && grid.confidence === 1.0) {
                App.state.manualGrid = grid;
            } else {
                App.state.manualGrid = null;
            }

            // Restore annotations
            Annotator.reset();
            if (sample.manual_additions) {
                sample.manual_additions.forEach(a => Annotator.getAdditions().push(a));
            }
            if (sample.manual_removals) {
                sample.manual_removals.forEach(r => Annotator.getRemovals().push(r));
            }

            // Use filtered summary when grid was detected
            const baseSummary = (App.state.currentFiltered && grid && grid.detected)
                ? (App.state.currentFiltered.summary || App.state.currentSummary)
                : App.state.currentSummary;
            const excludedIds = App.state.currentFiltered
                ? new Set((App.state.currentFiltered.excluded || []).map(d => d.id))
                : new Set();
            const eff = Annotator.getEffectiveSummary(baseSummary, excludedIds);
            App.updateResults(eff, App.state.currentFiltered);
            Charts.updateViability(eff.viable, eff.non_viable);

            // Enable save button for re-editing
            document.getElementById('saveSampleBtn').disabled = false;

            App.redraw();

            // Highlight active row
            renderGroups(currentSession.sample_groups || []);
        } catch (e) {
            App.toast('Failed to load sample');
        }
    }

    function cycleSample(direction) {
        if (!currentSession) return;
        const allSamples = currentSession.samples || [];
        if (!allSamples.length) return;
        const count = allSamples.length;
        activeSampleIndex = ((activeSampleIndex + direction) % count + count) % count;
        const sample = allSamples[activeSampleIndex];
        selectSample(activeSampleIndex, sample.image_id);
    }

    function getSampleCount() {
        if (!currentSession) return 0;
        return (currentSession.samples || []).length;
    }

    function getGroupCount() {
        if (!currentSession) return 0;
        return (currentSession.sample_groups || []).length;
    }

    return { refresh, cycleSample, getSampleCount, getGroupCount };
})();
