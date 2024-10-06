document.addEventListener('DOMContentLoaded', function() {
    const preBlocks = document.querySelectorAll('pre');
    const minLines = 32; // Minimum number of lines to make <pre> collapsible

    preBlocks.forEach(pre => {
        const codeLines = pre.textContent.trim().split('\n').length;

        if (codeLines > minLines) {
            const details = document.createElement('details');
            const summary = document.createElement('summary');
            summary.textContent = 'View code';

            // Toggle text between 'View code' and 'Hide code'
            details.addEventListener('toggle', function() {
                summary.textContent = details.open ? 'Hide code' : 'View code';
            });

            pre.parentNode.insertBefore(details, pre);
            details.appendChild(summary);
            details.appendChild(pre);
        }
    });
});
