window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Copy BibTeX functionality
    $('#copy-bibtex').click(function() {
        const bibtexText = document.getElementById('bibtex-code').textContent;
        const button = $(this);

        navigator.clipboard.writeText(bibtexText).then(function() {
            // Success feedback
            const originalText = button.find('span:last').text();
            const originalIcon = button.find('i').attr('class');

            button.find('span:last').text('Copied!');
            button.find('i').attr('class', 'fas fa-check');
            button.addClass('is-success').removeClass('is-dark');

            // Reset after 2 seconds
            setTimeout(function() {
                button.find('span:last').text(originalText);
                button.find('i').attr('class', originalIcon);
                button.removeClass('is-success').addClass('is-dark');
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy text: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexText;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);

            // Success feedback for fallback
            const originalText = button.find('span:last').text();
            button.find('span:last').text('Copied!');
            setTimeout(function() {
                button.find('span:last').text(originalText);
            }, 2000);
        });
    });

})
