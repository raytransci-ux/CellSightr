/**
 * Annotator module - Canvas-based manual cell annotation
 *
 * L-click: add viable cell (green)
 * R-click: add non-viable cell (red)
 * Shift+click: remove detection at cursor
 * Z: undo last action
 * X: reset all annotations
 */
const Annotator = (() => {
    const additions = [];  // {x, y, class, class_name}
    const removals = [];   // detection IDs
    const history = [];    // for undo: {type: 'add'|'remove', ...}

    function tryRemoveDetection(pos) {
        const det = findDetectionAt(pos.x, pos.y);
        if (!det) return false;
        if (removals.includes(det.id)) return false;  // already removed
        removals.push(det.id);
        history.push({ type: 'remove', detId: det.id });
        App.onAnnotationChanged();
        return true;
    }

    function init() {
        const canvas = App.canvas;

        canvas.addEventListener('click', (e) => {
            if (!App.state.annotateMode) return;
            const pos = canvasToImage(e.offsetX, e.offsetY);
            if (!pos) return;

            // Shift+click: remove detection
            if (e.shiftKey) {
                tryRemoveDetection(pos);
                return;
            }

            const ann = {
                x: pos.x,
                y: pos.y,
                class: 0,
                class_name: 'viable',
            };
            additions.push(ann);
            history.push({ type: 'add', index: additions.length - 1 });
            App.onAnnotationChanged();
        });

        canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            if (!App.state.annotateMode) return;
            const pos = canvasToImage(e.offsetX, e.offsetY);
            if (!pos) return;

            // Shift+right-click: also remove detection
            if (e.shiftKey) {
                tryRemoveDetection(pos);
                return;
            }

            const ann = {
                x: pos.x,
                y: pos.y,
                class: 1,
                class_name: 'non_viable',
            };
            additions.push(ann);
            history.push({ type: 'add', index: additions.length - 1 });
            App.onAnnotationChanged();
        });
    }

    function canvasToImage(cx, cy) {
        const scale = App.displayScale();
        const ox = App.offsetX();
        const oy = App.offsetY();
        if (scale <= 0) return null;

        const x = (cx - ox) / scale;
        const y = (cy - oy) / scale;

        // Bounds check
        const size = App.state.currentImageSize;
        if (!size || x < 0 || y < 0 || x > size.width || y > size.height) return null;

        return { x: Math.round(x), y: Math.round(y) };
    }

    function findDetectionAt(imgX, imgY) {
        // Build set of excluded (outside-grid) detection IDs
        const excluded = new Set();
        const filtered = App.state.currentFiltered;
        const grid = App.state.currentGrid;
        if (filtered && grid && grid.detected) {
            for (const det of (filtered.excluded || [])) {
                excluded.add(det.id);
            }
        }

        // Check if point falls inside any active detection bbox
        for (const det of App.state.currentDetections) {
            if (excluded.has(det.id)) continue;  // skip outside-grid detections
            const [x1, y1, x2, y2] = det.bbox;
            if (imgX >= x1 && imgX <= x2 && imgY >= y1 && imgY <= y2) {
                return det;
            }
        }
        return null;
    }

    function undo() {
        if (!history.length) return;
        const action = history.pop();
        if (action.type === 'add') {
            additions.splice(action.index, 1);
        } else if (action.type === 'remove') {
            const idx = removals.indexOf(action.detId);
            if (idx >= 0) removals.splice(idx, 1);
        }
        App.onAnnotationChanged();
    }

    function reset() {
        additions.length = 0;
        removals.length = 0;
        history.length = 0;
    }

    function getAdditions() {
        return additions;
    }

    function getRemovals() {
        return removals;
    }

    function getEffectiveSummary(originalSummary, excludedIds) {
        let viable = originalSummary.viable;
        let nonViable = originalSummary.non_viable;
        const excluded = excludedIds || new Set();

        // Subtract removals (only for detections that are in the active set,
        // i.e. not excluded by grid filtering)
        for (const detId of removals) {
            if (excluded.has(detId)) continue;  // outside grid — not in counts
            const det = App.state.currentDetections.find(d => d.id === detId);
            if (det) {
                if (det.class === 0) viable--;
                else nonViable--;
            }
        }

        // Add additions
        for (const ann of additions) {
            if (ann.class === 0) viable++;
            else nonViable++;
        }

        viable = Math.max(0, viable);
        nonViable = Math.max(0, nonViable);
        const total = viable + nonViable;
        return {
            total,
            viable,
            non_viable: nonViable,
            viability_pct: total > 0 ? Math.round((viable / total) * 1000) / 10 : 0,
        };
    }

    document.addEventListener('DOMContentLoaded', init);

    return { getAdditions, getRemovals, getEffectiveSummary, undo, reset };
})();
