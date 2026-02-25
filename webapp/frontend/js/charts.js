/**
 * Charts module - Viability donut
 */
const Charts = (() => {
    let donutChart = null;

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
    }

    function updateViability(viable, nonViable) {
        if (!donutChart) return;

        donutChart.data.datasets[0].data = [viable, nonViable];
        donutChart.update();

        const total = viable + nonViable;
        const pct = total > 0 ? Math.round((viable / total) * 1000) / 10 : 0;
        document.getElementById('donutCenterText').textContent = `${pct}%`;
    }

    /** No-op stub for backward compat with session.js calls */
    function updateComparison() {}

    document.addEventListener('DOMContentLoaded', init);

    return { updateViability, updateComparison };
})();
