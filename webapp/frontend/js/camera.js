/**
 * Camera module - WebSocket feed consumer
 */
const Camera = (() => {
    let ws = null;
    let paused = false;
    let reconnectTimer = null;

    function connect() {
        if (ws && ws.readyState <= 1) return; // already connected/connecting

        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws/camera`);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            clearTimeout(reconnectTimer);
            App.showCanvas();
            // Sync live tracking state on connect
            if (App.state.liveTracking) {
                sendCommand({ live_tracking: true });
            }
        };

        ws.onmessage = async (event) => {
            if (paused) return;
            try {
                const blob = new Blob([event.data], { type: 'image/jpeg' });
                const bitmap = await createImageBitmap(blob);
                const canvas = App.canvas;
                const ctx = App.ctx;
                const container = canvas.parentElement;
                const rect = container.getBoundingClientRect();

                canvas.width = rect.width;
                canvas.height = rect.height;

                const cw = canvas.width;
                const ch = canvas.height;
                const iw = bitmap.width;
                const ih = bitmap.height;
                const scale = Math.min(cw / iw, ch / ih);
                const dw = Math.floor(iw * scale);
                const dh = Math.floor(ih * scale);
                const ox = Math.floor((cw - dw) / 2);
                const oy = Math.floor((ch - dh) / 2);

                ctx.fillStyle = '#000';
                ctx.fillRect(0, 0, cw, ch);
                ctx.drawImage(bitmap, ox, oy, dw, dh);
                bitmap.close();
            } catch (e) {
                // frame decode error, skip
            }
        };

        ws.onclose = () => {
            if (App.state.cameraRunning) {
                // Auto-reconnect after 2s
                reconnectTimer = setTimeout(connect, 2000);
            }
        };

        ws.onerror = () => {
            ws.close();
        };
    }

    function disconnect() {
        clearTimeout(reconnectTimer);
        if (ws) {
            ws.close();
            ws = null;
        }
        paused = false;
    }

    function pause() {
        paused = true;
    }

    function resume() {
        paused = false;
    }

    function sendCommand(cmd) {
        if (ws && ws.readyState === 1) {
            ws.send(JSON.stringify(cmd));
        }
    }

    return { connect, disconnect, pause, resume, sendCommand };
})();
