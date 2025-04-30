document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard Loaded');

    // Theme toggle
    const themeSwitch = document.getElementById('themeSwitch');
    if (themeSwitch) {
        themeSwitch.addEventListener('click', () => {
            document.body.classList.toggle('dark-theme');
            localStorage.setItem(
                'darkTheme',
                document.body.classList.contains('dark-theme')
            );
        });

        // Apply saved theme
        if (localStorage.getItem('darkTheme') === 'true') {
            document.body.classList.add('dark-theme');
        }
    }
});