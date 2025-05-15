window.onload = function () {
    // Check if theme was previously set, if not, set light theme as default
    if (!localStorage.getItem('theme')) {
        document.body.classList.remove('dark-theme');
    } else if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-theme');
    }
};
