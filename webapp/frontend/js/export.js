/**
 * Export module - CSV generation, image download, ZIP export
 */
const Export = (() => {

    function exportClientSide() {
        const session = App.state.session;
        const groups = session?.sample_groups || [];
        if (!groups.length) {
            App.toast('No samples to export');
            return;
        }

        const cal = session.calibration || {};
        const dilution = cal.dilution_factor || 1;
        const trypan = cal.trypan_blue_dilution !== false;
        const effectiveDilution = dilution * (trypan ? 2 : 1);

        const headers = [
            'experiment', 'sample_name', 'image_num', 'image_id', 'timestamp',
            'total_cells', 'viable', 'non_viable', 'viability_pct',
            'group_total', 'group_viable', 'group_non_viable', 'group_viability_pct',
            'group_concentration_cells_per_ml', 'group_image_count',
            'dilution_factor', 'trypan_blue_dilution', 'effective_dilution',
            'manual_added', 'manual_removed', 'notes',
        ];

        const rows = [];
        for (const g of groups) {
            const images = g.images || [];
            if (!images.length) continue;
            const agg = g.aggregate_summary || {};
            const groupConc = ((agg.total || 0) / images.length) * effectiveDilution * 10000;

            images.forEach((s, imgIdx) => {
                const eff = s.effective_summary || s.summary || {};
                rows.push([
                    session.experiment_name,
                    g.name,
                    imgIdx + 1,
                    s.image_id,
                    s.timestamp,
                    eff.total || 0,
                    eff.viable || 0,
                    eff.non_viable || 0,
                    eff.viability_pct || 0,
                    agg.total || 0, agg.viable || 0, agg.non_viable || 0, agg.viability_pct || 0,
                    Math.round(groupConc), images.length,
                    dilution, trypan, effectiveDilution,
                    (s.manual_additions || []).length,
                    (s.manual_removals || []).length,
                    (s.notes || '').replace(/,/g, ';'),
                ]);
            });
        }

        const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
        downloadBlob(csv, `${session.experiment_name}.csv`, 'text/csv');
        App.toast('CSV exported (client-side)');
    }

    function downloadAnnotatedImage() {
        const canvas = App.canvas;
        if (!canvas) return;

        canvas.toBlob((blob) => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `cellcount_${App.state.currentImageId || 'image'}.png`;
            a.click();
            URL.revokeObjectURL(url);
            App.toast('Image downloaded');
        });
    }

    function downloadBlob(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    return { exportClientSide, downloadAnnotatedImage };
})();
