/**
 * Shortcuts module - Keyboard shortcut handler
 */
const Shortcuts = (() => {
    function init() {
        document.addEventListener('keydown', (e) => {
            // Don't capture when typing in input fields
            const tag = e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

            const key = e.key;
            const actions = App.actions;

            switch (key) {
                case ' ':
                    e.preventDefault();
                    actions.capture();
                    break;
                case 'a':
                case 'A':
                    actions.toggleAnnotate();
                    break;
                case 's':
                    e.preventDefault();
                    actions.saveSample();
                    break;
                case 'e':
                    actions.exportCSV();
                    break;
                case 'c':
                    actions.toggleCamera();
                    break;
                case 'r':
                    actions.reAnalyze();
                    break;
                case 'n':
                    actions.newSampleGroup();
                    break;
                case 'z':
                case 'Z':
                    if (App.state.annotateMode) {
                        Annotator.undo();
                        App.toast('Undo');
                    }
                    break;
                case 'x':
                case 'X':
                    if (App.state.annotateMode) {
                        Annotator.reset();
                        App.onAnnotationChanged();
                        App.toast('Annotations reset');
                    }
                    break;
                case 'Escape':
                    actions.escape();
                    break;
                case '?':
                case 'F1':
                    e.preventDefault();
                    actions.toggleHelp();
                    break;
                case '+':
                case '=':
                    actions.adjustConf(0.05);
                    break;
                case '-':
                    actions.adjustConf(-0.05);
                    break;
                case 'Tab':
                    e.preventDefault();
                    Session.cycleSample(e.shiftKey ? -1 : 1);
                    break;
            }
        });
    }

    document.addEventListener('DOMContentLoaded', init);

    return {};
})();
