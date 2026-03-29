/**
 * PrismaLab Mini App V2 — Express + Photosets
 */

// === Helpers ===
function pluralCredits(n) {
    const abs = Math.abs(n) % 100;
    const last = abs % 10;
    if (abs >= 11 && abs <= 19) return 'кредитов';
    if (last === 1) return 'кредит';
    if (last >= 2 && last <= 4) return 'кредита';
    return 'кредитов';
}

function uuid() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}

// === Analytics ===
function trackEvent(event, data) {
    fetch('/app/api/track', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ init_data: state.initData, event, data: data || {} }),
    }).catch(() => {});
}

// === State ===
const state = {
    initData: '',
    userId: null,
    gender: null,
    // Express credits
    expressCredits: { fast: 0, free_used: false },
    // Photosets (persona) credits
    photosetsCredits: 0,
    // Express flow
    expressThemes: [],
    selectedTheme: null,
    expressStyles: [],
    selectedExpressStyle: null,
    expressFile: null,
    expressTaskId: null,
    // Photosets flow
    packs: [],
    personaStyles: [],
    selectedPack: null,
    selectedPersonaStyle: null,
    photosetFile: null,
    photosetResult: null,
    photosetFilter: 'all',
    // General
    hasPersona: false,
};

// === Telegram SDK ===
const tg = window.Telegram?.WebApp;

// === Init ===
document.addEventListener('DOMContentLoaded', () => {
    if (tg) {
        tg.ready();
        tg.expand();
        tg.setHeaderColor('#0a0a0f');
        tg.setBackgroundColor('#0a0a0f');
        state.initData = tg.initData || '';
    }
    setupDragDrop();
    authenticate();
});

// === Auth ===
async function authenticate() {
    try {
        const resp = await fetch('/app/api/auth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ init_data: state.initData }),
        });

        if (resp.ok) {
            const data = await resp.json();
            state.userId = data.user_id;
            state.gender = data.gender;
            state.expressCredits = data.credits || { fast: 0, free_used: false };
            state.hasPersona = !!data.has_persona;
            state.photosetsCredits = data.persona_credits || 0;
            updateBalanceDisplays();

            // Check pack payment return
            const params = new URLSearchParams(window.location.search);
            if (params.get('pack_paid')) {
                showScreen('pack-paid');
                return;
            }

            trackEvent('miniapp_open');

            if (!data.gender) {
                showScreen('gender');
                return;
            }
            showScreen('main');
        } else {
            if (resp.status === 403) {
                showBlocked('Mini App сейчас доступна только тестовому аккаунту.');
                return;
            }
            showBlocked('Не удалось авторизовать Mini App.');
        }
    } catch (e) {
        console.error('Auth error:', e);
        showBlocked('Ошибка подключения. Попробуйте позже.');
    }
}

function showBlocked(message) {
    document.body.innerHTML = `
        <div style="min-height:100vh;display:flex;align-items:center;justify-content:center;background:#0a0a0f;color:#fff;padding:24px;text-align:center;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
            <div>
                <h2 style="margin:0 0 12px;font-size:22px;">Mini App недоступна</h2>
                <p style="margin:0;color:#b5b7c0;font-size:15px;line-height:1.5;">${message}</p>
            </div>
        </div>
    `;
}

function updateBalanceDisplays() {
    const expressBalance = state.expressCredits.free_used ? state.expressCredits.fast : (state.expressCredits.fast + 1);

    // Main screen
    const mainExpress = document.getElementById('main-express-balance');
    if (mainExpress) mainExpress.textContent = expressBalance;
    const mainPhotosets = document.getElementById('main-photosets-balance');
    if (mainPhotosets) mainPhotosets.textContent = state.photosetsCredits;
    const mainPhotosetsWord = document.getElementById('main-photosets-balance-word');
    if (mainPhotosetsWord) mainPhotosetsWord.textContent = pluralCredits(state.photosetsCredits);

    // Headers
    ['express-credits-count', 'express-styles-credits'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = expressBalance;
    });
    const pCredits = document.getElementById('photosets-credits-count');
    if (pCredits) pCredits.textContent = state.photosetsCredits;

    // Profile
    const pe = document.getElementById('profile-express-credits');
    if (pe) pe.textContent = expressBalance;
    const pp = document.getElementById('profile-photosets-credits');
    if (pp) pp.textContent = state.photosetsCredits;
    const pg = document.getElementById('profile-gender');
    if (pg) pg.textContent = state.gender === 'male' ? 'Мужской' : state.gender === 'female' ? 'Женский' : '—';
}

// === Navigation ===

function showScreen(name) {
    const screens = document.querySelectorAll('.screen');
    screens.forEach(s => {
        if (s.classList.contains('active')) {
            s.classList.add('slide-out');
            setTimeout(() => s.classList.remove('active', 'slide-out'), 300);
        }
    });
    setTimeout(() => {
        const target = document.getElementById(`screen-${name}`);
        if (target) target.classList.add('active');
    }, 50);
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
}

function goBack(screen) {
    showScreen(screen);
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
}

// === Gender ===

async function selectGender(gender) {
    state.gender = gender;
    trackEvent('miniapp_gender_select', { gender });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
    try {
        await fetch('/app/api/profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ init_data: state.initData, gender }),
        });
    } catch (e) { console.error('Gender save error:', e); }
    showScreen('main');
}

// === Main Screen Navigation ===

function goToExpress() {
    trackEvent('v2_nav_express');
    loadExpressThemes();
}

function goToPhotosets() {
    trackEvent('v2_nav_photosets');
    loadPhotosets();
}

function goToProfile() {
    trackEvent('v2_nav_profile');
    updateBalanceDisplays();
    showScreen('profile');
}

function closeMiniApp() {
    if (tg) { tg.close(); } else { window.close(); }
}

function buyCredits() {
    // Redirect to bot for credit purchase
    if (tg) {
        tg.sendData(JSON.stringify({ action: 'buy_credits' }));
        tg.close();
    }
}

// === EXPRESS FLOW ===

const THEME_ICONS = {
    lifestyle: '🌟', glamour: '✨', mood: '🌙', beauty: '💄',
    creative: '🎨', outdoor: '🌿', business: '💼', general: '🎯',
};

async function loadExpressThemes() {
    showScreen('express-themes');
    const grid = document.getElementById('themes-grid');
    grid.innerHTML = Array(6).fill('<div class="theme-card skeleton"><div style="height:100%"></div></div>').join('');

    try {
        const resp = await fetch(`/app/api/v2/express-themes?gender=${state.gender || 'female'}`, {
            headers: { 'X-Telegram-Init-Data': state.initData },
        });
        const data = await resp.json();
        state.expressThemes = data.themes || [];
        renderExpressThemes();
    } catch (e) {
        console.error('Load themes error:', e);
        grid.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-secondary)">Ошибка загрузки</div>';
    }
}

function renderExpressThemes() {
    const grid = document.getElementById('themes-grid');
    if (!state.expressThemes.length) {
        grid.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-secondary);grid-column:1/-1">Темы скоро появятся</div>';
        return;
    }
    grid.innerHTML = state.expressThemes.map((theme, i) => `
        <div class="theme-card fade-in" style="animation-delay:${i * 0.05}s" onclick="selectExpressTheme('${theme}')">
            <div class="theme-card-icon">${THEME_ICONS[theme] || '🎯'}</div>
            <div class="theme-card-name">${theme.charAt(0).toUpperCase() + theme.slice(1)}</div>
        </div>
    `).join('');
}

async function selectExpressTheme(theme) {
    state.selectedTheme = theme;
    trackEvent('v2_express_theme_select', { theme });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');

    document.getElementById('express-theme-title').textContent = theme.charAt(0).toUpperCase() + theme.slice(1);
    showScreen('express-styles');

    const grid = document.getElementById('express-styles-grid');
    grid.innerHTML = Array(6).fill('<div class="style-card skeleton"><div style="height:100%"></div></div>').join('');

    try {
        const resp = await fetch(`/app/api/v2/express-styles?gender=${state.gender || 'female'}&theme=${theme}`, {
            headers: { 'X-Telegram-Init-Data': state.initData },
        });
        const data = await resp.json();
        state.expressStyles = data.styles || [];
        renderExpressStyles();
    } catch (e) {
        console.error('Load styles error:', e);
        grid.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-secondary)">Ошибка загрузки</div>';
    }
}

function renderExpressStyles() {
    const grid = document.getElementById('express-styles-grid');
    if (!state.expressStyles.length) {
        grid.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-secondary);grid-column:1/-1">Стили скоро появятся</div>';
        return;
    }

    const gradients = [
        'linear-gradient(135deg, #667eea, #764ba2)',
        'linear-gradient(135deg, #f093fb, #f5576c)',
        'linear-gradient(135deg, #4facfe, #00f2fe)',
        'linear-gradient(135deg, #43e97b, #38f9d7)',
        'linear-gradient(135deg, #fa709a, #fee140)',
        'linear-gradient(135deg, #a18cd1, #fbc2eb)',
    ];

    grid.innerHTML = state.expressStyles.map((style, i) => {
        const hasImage = style.image_url && style.image_url.startsWith('http');
        return `
        <div class="style-card fade-in" style="animation-delay:${i * 0.05}s" onclick="selectExpressStyle(${JSON.stringify(style).replace(/"/g, '&quot;')})">
            ${hasImage
                ? `<img class="style-card-image" src="${style.image_url}" alt="" loading="lazy">`
                : `<div class="style-card-bg" style="background:${gradients[i % gradients.length]}"></div>
                   <div class="style-card-emoji">${style.emoji || '🎨'}</div>`
            }
            <div class="style-card-info">
                <div class="style-card-name">${style.emoji || ''} ${style.label}</div>
            </div>
        </div>`;
    }).join('');
}

function selectExpressStyle(style) {
    state.selectedExpressStyle = style;
    state.expressFile = null;
    trackEvent('v2_express_style_select', { style_id: style.id });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');

    document.getElementById('express-selected-style').textContent = style.label;
    resetExpressUpload();
    showScreen('express-upload');
}

// Express Upload
function handleExpressFile(event) {
    const file = event.target.files?.[0];
    if (file) processExpressFile(file);
}

function processExpressFile(file) {
    if (file.size > 15 * 1024 * 1024) {
        alert('Файл слишком большой (макс. 15 МБ)');
        return;
    }
    state.expressFile = file;
    trackEvent('v2_express_upload');

    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('express-preview-image').src = e.target.result;
        document.getElementById('express-upload-zone').style.display = 'none';
        document.getElementById('express-preview').style.display = 'block';
        document.getElementById('express-generate-btn').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function resetExpressUpload() {
    state.expressFile = null;
    document.getElementById('express-upload-zone').style.display = '';
    document.getElementById('express-preview').style.display = 'none';
    document.getElementById('express-generate-btn').style.display = 'none';
    const input = document.getElementById('express-file-input');
    if (input) input.value = '';
}

// Express Generation
async function startExpressGeneration() {
    if (!state.expressFile || !state.selectedExpressStyle) return;
    trackEvent('v2_express_generate_start', { style_id: state.selectedExpressStyle.id });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('heavy');

    showScreen('express-generating');
    startProgressAnimation('express-progress-bar', 'express-generating-tip');

    const formData = new FormData();
    formData.append('init_data', state.initData);
    formData.append('style_id', state.selectedExpressStyle.id);
    formData.append('photo', state.expressFile);

    try {
        const resp = await fetch('/app/api/v2/express-generate', {
            method: 'POST',
            body: formData,
        });
        const data = await resp.json();

        if (resp.status === 402) {
            stopProgressAnimation('express-progress-bar');
            showScreen('nocredits');
            return;
        }
        if (!resp.ok) throw new Error(data.error || 'Generation failed');

        state.expressTaskId = data.task_id;
        pollExpressStatus();
    } catch (e) {
        console.error('Express generation error:', e);
        stopProgressAnimation('express-progress-bar');
        alert('Ошибка генерации: ' + e.message);
        showScreen('express-upload');
    }
}

async function pollExpressStatus() {
    if (!state.expressTaskId) return;
    try {
        const resp = await fetch(`/app/api/status/${state.expressTaskId}`, {
            headers: { 'X-Telegram-Init-Data': state.initData },
        });
        const data = await resp.json();

        if (data.status === 'done') {
            stopProgressAnimation('express-progress-bar');
            trackEvent('v2_express_generate_done', { style_id: state.selectedExpressStyle?.id });

            if (!state.expressCredits.free_used) {
                state.expressCredits.free_used = true;
            } else {
                state.expressCredits.fast = Math.max(0, state.expressCredits.fast - 1);
            }
            updateBalanceDisplays();

            document.getElementById('express-result-image').src = data.result_url;
            showScreen('express-result');
            if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
        } else if (data.status === 'error') {
            stopProgressAnimation('express-progress-bar');
            alert('Ошибка генерации. Попробуйте ещё раз.');
            showScreen('express-upload');
            if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('error');
        } else {
            setTimeout(pollExpressStatus, 2000);
        }
    } catch (e) {
        console.error('Poll error:', e);
        setTimeout(pollExpressStatus, 3000);
    }
}

function downloadExpressResult() {
    const img = document.getElementById('express-result-image');
    if (!img.src) return;
    trackEvent('v2_express_download', { style_id: state.selectedExpressStyle?.id });
    const link = document.createElement('a');
    link.href = img.src;
    link.download = `prismalab_express_${state.selectedExpressStyle?.id || 'photo'}.jpg`;
    link.click();
    if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
}

// === PHOTOSETS FLOW ===

async function loadPhotosets() {
    showScreen('photosets');
    const grid = document.getElementById('photosets-grid');
    grid.innerHTML = Array(4).fill('<div class="photoset-card skeleton"><div class="photoset-card-cover-placeholder"></div><div class="photoset-card-info"><div style="height:14px;width:80%;background:var(--bg-glass);border-radius:4px"></div></div></div>').join('');

    try {
        // Load packs + persona styles in parallel
        const [packsResp, stylesResp] = await Promise.all([
            fetch('/app/api/v2/photosets', { headers: { 'X-Telegram-Init-Data': state.initData } }),
            fetch('/app/api/persona-styles', { headers: { 'X-Telegram-Init-Data': state.initData } }),
        ]);

        const packsData = await packsResp.json();
        const stylesData = await stylesResp.json();

        state.packs = (packsData.packs || []).map(p => ({ ...p, kind: 'pack' }));
        state.packsUseCredits = !!packsData.packs_use_credits;
        state.personaStyles = (stylesData.styles || []).map(s => ({ ...s, kind: 'style' }));

        renderPhotosets();
    } catch (e) {
        console.error('Load photosets error:', e);
        grid.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-secondary)">Ошибка загрузки</div>';
    }
}

function filterPhotosets(kind) {
    state.photosetFilter = kind;
    document.querySelectorAll('.photosets-tabs .category-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.kind === kind);
    });
    renderPhotosets();
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
}

function renderPhotosets() {
    const grid = document.getElementById('photosets-grid');
    let items = [];

    if (state.photosetFilter === 'all' || state.photosetFilter === 'pack') {
        items = items.concat(state.packs);
    }
    if (state.photosetFilter === 'all' || state.photosetFilter === 'style') {
        items = items.concat(state.personaStyles);
    }

    if (!items.length) {
        grid.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-secondary);grid-column:1/-1">Фотосеты скоро появятся</div>';
        return;
    }

    grid.innerHTML = items.map((item, i) => {
        if (item.kind === 'pack') {
            const priceText = state.packsUseCredits
                ? `${item.credit_cost || item.expected_images} кр`
                : `${item.price_rub} ₽`;
            return `
            <div class="photoset-card fade-in" style="animation-delay:${i * 0.05}s" onclick="openPackDetail(${item.id})">
                ${item.cover_url
                    ? `<img class="photoset-card-cover" src="${item.cover_url}" alt="" loading="lazy">`
                    : '<div class="photoset-card-cover-placeholder">📸</div>'}
                <div class="photoset-card-info">
                    <div class="photoset-card-title">${item.title}</div>
                    <div class="photoset-card-meta">
                        <span class="photoset-card-badge badge-pack">Пак</span>
                        <span class="photoset-card-price">${item.expected_images} фото · ${priceText}</span>
                    </div>
                </div>
            </div>`;
        } else {
            return `
            <div class="photoset-card fade-in" style="animation-delay:${i * 0.05}s" onclick="openStyleDetail(${item.id})">
                ${item.image_url
                    ? `<img class="photoset-card-cover" src="${item.image_url}" alt="" loading="lazy">`
                    : '<div class="photoset-card-cover-placeholder">🎭</div>'}
                <div class="photoset-card-info">
                    <div class="photoset-card-title">${item.title}</div>
                    <div class="photoset-card-meta">
                        <span class="photoset-card-badge badge-style">Образ</span>
                        <span class="photoset-card-price">${item.credit_cost || 4} кр · 4 фото</span>
                    </div>
                </div>
            </div>`;
        }
    }).join('');
}

// Pack Detail
async function openPackDetail(packId) {
    trackEvent('v2_photoset_detail', { kind: 'pack', id: packId });
    showScreen('photoset-pack-detail');
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');

    try {
        const resp = await fetch(`/app/api/packs/${packId}`, {
            headers: { 'X-Telegram-Init-Data': state.initData },
        });
        const pack = await resp.json();
        state.selectedPack = pack;

        document.getElementById('photoset-pack-title').textContent = pack.title;
        document.getElementById('photoset-pack-count').textContent = pack.expected_images;

        if (state.packsUseCredits) {
            const cc = pack.credit_cost || pack.expected_images;
            document.getElementById('photoset-pack-buy-text').textContent = `Купить за ${cc} ${pluralCredits(cc)}`;
        } else {
            document.getElementById('photoset-pack-buy-text').textContent = `Купить ${pack.price_rub} ₽`;
        }

        const gallery = document.getElementById('photoset-pack-gallery');
        gallery.innerHTML = (pack.examples || []).slice(0, 10).map(url =>
            `<img src="${url}" alt="" loading="lazy" onclick="openLightbox('${url}')">`
        ).join('');
    } catch (e) {
        console.error('Pack detail error:', e);
    }
}

async function buyPhotosetPack() {
    if (!state.selectedPack) return;
    trackEvent('v2_photoset_buy', { pack_id: state.selectedPack.id, use_credits: state.packsUseCredits });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('heavy');

    if (state.packsUseCredits) {
        // Покупка за кредиты
        const cc = state.selectedPack.credit_cost || state.selectedPack.expected_images;
        if (state.photosetsCredits < cc) {
            document.getElementById('nocredits-text').textContent =
                `Нужно ${cc} ${pluralCredits(cc)}, у вас ${state.photosetsCredits}.`;
            showScreen('nocredits');
            return;
        }
        try {
            const resp = await fetch(`/app/api/v2/packs/${state.selectedPack.id}/buy-credits`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ init_data: state.initData }),
            });
            const data = await resp.json();

            if (resp.status === 402) {
                document.getElementById('nocredits-text').textContent = data.message || 'Недостаточно кредитов';
                showScreen('nocredits');
                return;
            }
            if (!resp.ok) throw new Error(data.error || 'Purchase failed');

            state.photosetsCredits = data.credits_balance ?? state.photosetsCredits;
            updateBalanceDisplays();
            showScreen('pack-paid');
        } catch (e) {
            console.error('Pack buy credits error:', e);
            alert('Ошибка покупки: ' + e.message);
        }
    } else {
        // Покупка за ₽ (старый flow)
        try {
            const resp = await fetch(`/app/api/packs/${state.selectedPack.id}/buy`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ init_data: state.initData }),
            });
            const data = await resp.json();

            if (data.payment_url) {
                if (tg) { tg.openLink(data.payment_url); }
                else { window.open(data.payment_url, '_blank'); }
            }
        } catch (e) {
            console.error('Pack buy error:', e);
            alert('Ошибка оплаты');
        }
    }
}

// Style (Persona) Detail — 4 photo generation
function openStyleDetail(styleId) {
    const style = state.personaStyles.find(s => s.id === styleId);
    if (!style) return;
    state.selectedPersonaStyle = style;
    trackEvent('v2_photoset_detail', { kind: 'style', id: styleId });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');

    document.getElementById('photoset-style-title').textContent = style.title;
    document.getElementById('photoset-style-desc').textContent = style.description || '';

    const creditCost = style.credit_cost || 4;
    document.getElementById('photoset-style-cost-value').textContent = creditCost;
    document.getElementById('photoset-style-cost-word').textContent = pluralCredits(creditCost);

    // Preview grid (placeholder for 4 photos)
    const preview = document.getElementById('photoset-style-preview');
    if (style.image_url) {
        preview.innerHTML = `
            <img src="${style.image_url}" alt="">
            <div class="preview-placeholder">?</div>
            <div class="preview-placeholder">?</div>
            <div class="preview-placeholder">?</div>
        `;
    } else {
        preview.innerHTML = Array(4).fill('<div class="preview-placeholder">?</div>').join('');
    }

    resetPhotosetUpload();
    showScreen('photoset-style-detail');
}

// Photoset Upload
function handlePhotosetFile(event) {
    const file = event.target.files?.[0];
    if (file) processPhotosetFile(file);
}

function processPhotosetFile(file) {
    if (file.size > 15 * 1024 * 1024) {
        alert('Файл слишком большой (макс. 15 МБ)');
        return;
    }
    state.photosetFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('photoset-preview-image').src = e.target.result;
        document.getElementById('photoset-upload-zone').style.display = 'none';
        document.getElementById('photoset-preview').style.display = 'block';
        document.getElementById('photoset-generate-btn').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function resetPhotosetUpload() {
    state.photosetFile = null;
    document.getElementById('photoset-upload-zone').style.display = '';
    document.getElementById('photoset-preview').style.display = 'none';
    document.getElementById('photoset-generate-btn').style.display = 'none';
    const input = document.getElementById('photoset-file-input');
    if (input) input.value = '';
}

// Photoset Generation (4 photos)
async function startPhotosetGeneration() {
    if (!state.photosetFile || !state.selectedPersonaStyle) return;

    const style = state.selectedPersonaStyle;
    const creditCost = style.credit_cost || 4;

    if (state.photosetsCredits < creditCost) {
        document.getElementById('nocredits-text').textContent =
            `Нужно ${creditCost} ${pluralCredits(creditCost)}, у вас ${state.photosetsCredits}.`;
        showScreen('nocredits');
        return;
    }

    trackEvent('v2_photoset_generate_start', { style_id: style.id });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('heavy');

    showScreen('photoset-generating');
    startProgressAnimation('photoset-progress-bar', 'photoset-generating-tip', true);

    const requestId = uuid();
    const formData = new FormData();
    formData.append('init_data', state.initData);
    formData.append('request_id', requestId);
    formData.append('photo', state.photosetFile);

    try {
        const resp = await fetch(`/app/api/v2/photosets/style/${style.id}/generate`, {
            method: 'POST',
            body: formData,
        });
        const data = await resp.json();

        stopProgressAnimation('photoset-progress-bar');

        if (resp.status === 402) {
            showScreen('nocredits');
            return;
        }
        if (resp.status === 409) {
            alert('Генерация уже запущена. Подождите завершения.');
            showScreen('photosets');
            return;
        }
        if (!resp.ok && resp.status !== 200) {
            throw new Error(data.error || 'Generation failed');
        }

        state.photosetResult = data;
        state.photosetsCredits = data.credits_balance ?? state.photosetsCredits;
        updateBalanceDisplays();

        showPhotosetResult(data);

        trackEvent('v2_photoset_generate_done', {
            style_id: style.id,
            success_count: data.success_count,
            status: data.status,
        });

        if (data.status === 'done' || data.status === 'partial') {
            if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
        } else {
            if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('error');
        }
    } catch (e) {
        console.error('Photoset generation error:', e);
        stopProgressAnimation('photoset-progress-bar');
        alert('Ошибка генерации: ' + e.message);
        showScreen('photoset-style-detail');
        if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('error');
    }
}

function showPhotosetResult(data) {
    const grid = document.getElementById('photoset-result-grid');
    const stats = document.getElementById('photoset-result-stats');

    grid.innerHTML = (data.images || []).map((url, i) =>
        `<img src="${url}" alt="Фото ${i + 1}" onclick="openLightbox('${url}')">`
    ).join('');

    let statsText = `${data.success_count}/${data.requested_count} фото`;
    if (data.credits_spent > 0) statsText += ` · Списано ${data.credits_spent} ${pluralCredits(data.credits_spent)}`;
    if (data.credits_refunded > 0) statsText += ` · Возврат ${data.credits_refunded}`;
    stats.textContent = statsText;

    showScreen('photoset-result');
}

function downloadAllPhotoset() {
    if (!state.photosetResult?.images?.length) return;
    state.photosetResult.images.forEach((url, i) => {
        const link = document.createElement('a');
        link.href = url;
        link.download = `prismalab_photoset_${i + 1}.jpg`;
        link.click();
    });
    if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
}

// === Progress Animation ===

let _progressIntervals = {};

function startProgressAnimation(barId, tipId, isPhotoset) {
    let value = 0;
    const bar = document.getElementById(barId);
    if (bar) bar.style.width = '0%';

    const tips = isPhotoset
        ? ['AI анализирует фото...', 'Генерируем вариант 1/4...', 'Генерируем вариант 2/4...', 'Генерируем вариант 3/4...', 'Финальный вариант...']
        : ['AI анализирует ваше фото...', 'Применяем стиль...', 'Добавляем детали...', 'Финальные штрихи...', 'Почти готово...'];

    let tipIndex = 0;

    _progressIntervals[barId] = setInterval(() => {
        if (value < 90) {
            value += Math.random() * (isPhotoset ? 1.5 : 3) + 0.5;
            if (bar) bar.style.width = Math.min(value, 90) + '%';
        }
        if (value > tipIndex * 20 + 10 && tipIndex < tips.length) {
            const tipEl = document.getElementById(tipId);
            if (tipEl) tipEl.textContent = tips[tipIndex];
            tipIndex++;
        }
    }, 500);
}

function stopProgressAnimation(barId) {
    if (_progressIntervals[barId]) {
        clearInterval(_progressIntervals[barId]);
        delete _progressIntervals[barId];
    }
    const bar = document.getElementById(barId);
    if (bar) bar.style.width = '100%';
}

// === Lightbox ===

function openLightbox(url) {
    const lb = document.getElementById('lightbox');
    const img = document.getElementById('lightbox-img');
    if (lb && img) {
        img.src = url;
        lb.style.display = 'flex';
    }
}

function closeLightbox() {
    const lb = document.getElementById('lightbox');
    if (lb) lb.style.display = 'none';
}

// === Drag & Drop ===

function setupDragDrop() {
    // Express upload zone
    const expressZone = document.getElementById('express-upload-zone');
    if (expressZone) {
        expressZone.addEventListener('dragover', e => { e.preventDefault(); expressZone.classList.add('dragover'); });
        expressZone.addEventListener('dragleave', () => expressZone.classList.remove('dragover'));
        expressZone.addEventListener('drop', e => {
            e.preventDefault(); expressZone.classList.remove('dragover');
            const file = e.dataTransfer?.files?.[0];
            if (file) processExpressFile(file);
        });
    }

    // Photoset upload zone
    const photosetZone = document.getElementById('photoset-upload-zone');
    if (photosetZone) {
        photosetZone.addEventListener('dragover', e => { e.preventDefault(); photosetZone.classList.add('dragover'); });
        photosetZone.addEventListener('dragleave', () => photosetZone.classList.remove('dragover'));
        photosetZone.addEventListener('drop', e => {
            e.preventDefault(); photosetZone.classList.remove('dragover');
            const file = e.dataTransfer?.files?.[0];
            if (file) processPhotosetFile(file);
        });
    }
}
