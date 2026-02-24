/**
 * Charts module - Viability donut and sample comparison bar
 */
const Charts = (() => {
    let donutChart = null;
    let barChart = null;

    function init() {
        // Viability donut
        const donutCtx = document.getElementById('viabilityDonut');
        donutChart = new Chart(donutCtx, {
            type: 'doughnut',
            data: {
                labels: ['Viable', 'Non-viable'],
                datasets: [{
                    data: [0, 0],
                    backgroundColor: ['#48c78e', '#ef4444'],
                    borderWidth: 0,
                    hoverBorderWidth: 0,
                }],
            },
            options: {
                cutout: '72%',
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#232733',
                        titleColor: '#e8eaed',
                        bodyColor: '#9aa0a6',
                        borderColor: '#2a2e3b',
                        borderWidth: 1,
                        cornerRadius: 6,
                    },
                },
                animation: { duration: 300 },
            },
        });

        // Comparison bar chart — concentration per sample group
        const barCtx = document.getElementById('comparisonBar');
        barChart = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'cells/mL',
                    data: [],
                    backgroundColor: '#818cf8',
                    borderRadius: 3,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: { color: '#5f6368', font: { size: 10, family: "'JetBrains Mono', monospace" } },
                        grid: { display: false },
                        border: { color: '#2a2e3b' },
                    },
                    y: {
                        ticks: {
                            color: '#5f6368',
                            font: { size: 9 },
                            callback: function(value) {
                                if (value >= 1e6) return (value / 1e6).toFixed(1) + 'M';
                                if (value >= 1e3) return (value / 1e3).toFixed(0) + 'K';
                                return value;
                            },
                        },
                        grid: { color: '#1a1d27' },
                        border: { color: '#2a2e3b' },
                    },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#232733',
                        titleColor: '#e8eaed',
                        bodyColor: '#9aa0a6',
                        borderColor: '#2a2e3b',
                        borderWidth: 1,
                        cornerRadius: 6,
                        callbacks: {
                            label: function(ctx) {
                                const val = ctx.parsed.y;
                                if (val >= 1e6) return `${(val / 1e6).toFixed(2)} x 10\u2076 /mL`;
                                return `${Math.round(val).toLocaleString()} /mL`;
                            },
                        },
                    },
                },
                animation: { duration: 300 },
            },
        });
    }

    function updateViability(viable, nonViable) {
        if (!donutChart) return;

        donutChart.data.datasets[0].data = [viable, nonViable];
        donutChart.update();

        const total = viable + nonViable;
        const pct = total > 0 ? Math.round((viable / total) * 1000) / 10 : 0;
        document.getElementById('donutCenterText').textContent = `${pct}%`;
    }

    /** Update comparison chart with group-level concentration */
    function updateComparison(groups, cal) {
        if (!barChart) return;

        const dilution = cal?.dilution_factor || 1;
        const trypan = cal?.trypan_blue_dilution !== false;
        const effectiveDilution = dilution * (trypan ? 2 : 1);

        const labels = [];
        const data = [];

        for (const g of groups) {
            const images = g.images || [];
            if (!images.length) continue;
            const agg = g.aggregate_summary || {};
            const conc = (agg.total / images.length) * effectiveDilution * 10000;
            labels.push(g.name);
            data.push(Math.round(conc));
        }

        barChart.data.labels = labels;
        barChart.data.datasets[0].data = data;
        barChart.update();
    }

    document.addEventListener('DOMContentLoaded', init);

    return { updateViability, updateComparison };
})();
