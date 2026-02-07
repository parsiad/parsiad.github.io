import { codeToHtml } from 'https://esm.sh/shiki@3.0.0'

document.querySelectorAll('code[data-lang]').forEach(async el => {
    const lang = el.dataset.lang
    const highlighted = await codeToHtml(el.textContent, {
        lang,
        theme: 'catppuccin-latte'
    })
    el.parentElement.innerHTML = highlighted
});