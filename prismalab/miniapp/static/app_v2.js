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
    // V3 Express
    v3Categories: [],
    v3Tags: [],
    v3SelectedCategory: 'all',
    v3SelectedTags: [],
    selectedProvider: 'seedream',
    // Photosets flow
    photosets: [],
    packs: [],
    personaStyles: [],
    selectedPack: null,
    selectedPersonaStyle: null,
    // General
    hasPersona: false,
    packsUseCredits: false,
    // Profile history
    historyItems: [],
    historyTotal: 0,
    historyOffset: 0,
    // Custom prompt
    customCapabilities: null,
    customProvider: 'seedream',
    customFiles: [],
    customTaskId: null,
    customRequestId: null,
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
    const maxAttempts = 3;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
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
                return;
            }

            if (resp.status === 403) {
                showBlocked('Mini App сейчас доступна только тестовому аккаунту.');
                return;
            }

            // Для временных backend-сбоев пробуем повторно.
            if ((resp.status >= 500 || resp.status === 503) && attempt < maxAttempts) {
                await new Promise((resolve) => setTimeout(resolve, attempt * 700));
                continue;
            }

            showBlocked('Не удалось авторизовать Mini App.');
            return;
        } catch (e) {
            console.error('Auth error:', e);
            if (attempt < maxAttempts) {
                await new Promise((resolve) => setTimeout(resolve, attempt * 700));
                continue;
            }
            showBlocked('Ошибка подключения. Попробуйте позже.');
            return;
        }
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

    // Custom balance (same as express)
    const mainCustom = document.getElementById('main-custom-balance');
    if (mainCustom) mainCustom.textContent = expressBalance;
    const customCredits = document.getElementById('custom-credits-count');
    if (customCredits) customCredits.textContent = expressBalance;

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
    const target = document.getElementById(`screen-${name}`);
    if (!target) return;
    // Уже на этом экране — ничего не делаем
    if (target.classList.contains('active')) return;
    const screens = document.querySelectorAll('.screen');
    screens.forEach(s => {
        if (s.classList.contains('active')) {
            s.classList.add('slide-out');
            setTimeout(() => s.classList.remove('active', 'slide-out'), 300);
        }
    });
    setTimeout(() => {
        target.classList.add('active');
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
    if (window.EXPRESS_V3) {
        loadExpressCatalog();
    } else {
        loadExpressThemes();
    }
}

function goToPhotosets() {
    trackEvent('v2_nav_photosets');
    loadPhotosets();
}

async function goToProfile() {
    trackEvent('v2_nav_profile');
    updateBalanceDisplays();
    showScreen('profile');
    await loadProfileHistory(true);
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
    // Hide provider choice in V2
    const providerChoice = document.getElementById('provider-choice');
    if (providerChoice) providerChoice.style.display = 'none';
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
    if (window.EXPRESS_V3 && state.selectedProvider) {
        formData.append('provider', state.selectedProvider);
    }

    const generateUrl = window.EXPRESS_V3 ? '/app/api/v3/express/generate' : '/app/api/v2/express-generate';
    try {
        const resp = await fetch(generateUrl, {
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
            const note = document.getElementById('express-result-note');
            if (note) {
                note.textContent = data.tg_sent
                    ? 'Фото отправлено в Telegram и сохранено в профиле.'
                    : 'Фото сохранено в профиле. В Telegram отправить не удалось.';
            }
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

// === PROFILE HISTORY ===

async function loadProfileHistory(reset = true) {
    if (reset) {
        state.historyItems = [];
        state.historyOffset = 0;
        state.historyTotal = 0;
    }
    const grid = document.getElementById('profile-history-grid');
    const moreBtn = document.getElementById('profile-history-more');
    if (!grid || !moreBtn) return;

    if (reset) {
        grid.innerHTML = Array(6)
            .fill('<div class="style-card skeleton"><div style="height:100%"></div></div>')
            .join('');
    }

    try {
        const limit = 18;
        const url = `/app/api/v3/history?mode=express&limit=${limit}&offset=${state.historyOffset}`;
        const resp = await fetch(url, {
            headers: { 'X-Telegram-Init-Data': state.initData },
        });
        if (!resp.ok) {
            throw new Error(`History HTTP ${resp.status}`);
        }
        const data = await resp.json();
        const items = data.items || [];
        state.historyItems = reset ? items : state.historyItems.concat(items);
        state.historyOffset = state.historyItems.length;
        state.historyTotal = data.total || state.historyItems.length;
        renderProfileHistory();
        moreBtn.style.display = state.historyOffset < state.historyTotal ? '' : 'none';
    } catch (e) {
        console.error('Load profile history error:', e);
        if (reset) grid.innerHTML = '<div class="empty-state">Не удалось загрузить историю</div>';
        moreBtn.style.display = 'none';
    }
}

function renderProfileHistory() {
    const grid = document.getElementById('profile-history-grid');
    if (!grid) return;
    if (!state.historyItems.length) {
        grid.innerHTML = '<div class="empty-state">История пока пустая</div>';
        return;
    }
    grid.innerHTML = '';
    state.historyItems.forEach((item) => {
        if (!item.image_url) return;
        const wrapper = document.createElement('div');
        wrapper.className = 'profile-history-item';

        const img = document.createElement('img');
        img.src = item.image_url;
        img.alt = item.style_title || 'Generation';
        img.loading = 'lazy';
        img.addEventListener('click', () => openLightbox(item.image_url));
        wrapper.appendChild(img);

        const provider = document.createElement('div');
        provider.className = 'profile-history-provider';
        provider.textContent = item.provider === 'nano-banana-pro' ? 'Nano' : 'Seedream';
        wrapper.appendChild(provider);
        grid.appendChild(wrapper);
    });
    if (!grid.children.length) {
        grid.innerHTML = '<div class="empty-state">История пока пустая</div>';
    }
}

// === EXPRESS V3 FLOW ===

async function loadExpressCatalog(keepFilters) {
    if (!keepFilters) showScreen('express-catalog');
    const grid = document.getElementById('v3-styles-grid');
    if (!keepFilters) {
        state.v3SelectedCategory = 'all';
        state.v3SelectedTags = [];
    }
    grid.innerHTML = Array(6).fill('<div class="style-card skeleton"><div style="height:100%"></div></div>').join('');

    try {
        let url = '/app/api/v3/express/catalog';
        const params = [];
        if (state.v3SelectedCategory && state.v3SelectedCategory !== 'all') {
            params.push(`category=${encodeURIComponent(state.v3SelectedCategory)}`);
        }
        if (state.v3SelectedTags.length > 0) {
            params.push(`tags=${state.v3SelectedTags.map(encodeURIComponent).join(',')}`);
        }
        if (params.length) url += '?' + params.join('&');

        const resp = await fetch(url, {
            headers: { 'X-Telegram-Init-Data': state.initData },
        });
        const data = await resp.json();

        state.v3Categories = data.categories || [];
        state.v3Tags = data.tags || [];
        state.expressCredits = data.credits || { fast: 0, free_used: false };
        state.selectedProvider = data.last_provider || 'seedream';

        renderV3Categories();
        renderV3Tags();
        renderV3Styles(data.styles || []);
        updateBalanceDisplays();

        // Update V3 credits display
        const bal = data.credits?.balance ?? 0;
        const el = document.getElementById('v3-credits-count');
        if (el) el.textContent = bal;
    } catch (e) {
        console.error('Load catalog error:', e);
        grid.innerHTML = '<div class="empty-state">Ошибка загрузки</div>';
    }
}

function escAttr(s) { return String(s).replace(/&/g,'&amp;').replace(/'/g,'&#39;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function escHtml(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function renderV3Categories() {
    const container = document.getElementById('v3-category-pills');
    let html = `<button class="cat-pill ${state.v3SelectedCategory === 'all' ? 'active' : ''}" onclick="selectV3Category('all')">Все</button>`;
    for (const cat of state.v3Categories) {
        const active = state.v3SelectedCategory === cat.slug ? 'active' : '';
        html += `<button class="cat-pill ${active}" onclick="selectV3Category('${escAttr(cat.slug)}')">${escHtml(cat.title)}</button>`;
    }
    container.innerHTML = html;
}

function renderV3Tags() {
    const container = document.getElementById('v3-tag-pills');
    if (state.v3Tags.length === 0) {
        container.style.display = 'none';
        return;
    }
    container.style.display = '';
    let html = '';
    for (const tag of state.v3Tags) {
        const active = state.v3SelectedTags.includes(tag.slug) ? 'active' : '';
        html += `<button class="tag-pill ${active}" onclick="toggleV3Tag('${escAttr(tag.slug)}')">${escHtml(tag.title)}</button>`;
    }
    container.innerHTML = html;
}

function renderV3Styles(styles) {
    const grid = document.getElementById('v3-styles-grid');
    if (!styles.length) {
        grid.innerHTML = '<div class="empty-state">Нет стилей для выбранных фильтров</div>';
        return;
    }
    const GRADIENTS = [
        'linear-gradient(135deg, #667eea, #764ba2)',
        'linear-gradient(135deg, #f093fb, #f5576c)',
        'linear-gradient(135deg, #4facfe, #00f2fe)',
        'linear-gradient(135deg, #43e97b, #38f9d7)',
        'linear-gradient(135deg, #fa709a, #fee140)',
        'linear-gradient(135deg, #a18cd1, #fbc2eb)',
    ];
    grid.innerHTML = '';
    styles.forEach((s, i) => {
        const card = document.createElement('div');
        card.className = 'style-card';
        card.addEventListener('click', () => selectV3Style(s));

        const hasImg = s.image_url && s.image_url.startsWith('http');
        if (hasImg) {
            card.style.backgroundImage = `url(${CSS.escape(s.image_url)})`;
            card.style.backgroundSize = 'cover';
            card.style.backgroundPosition = 'center';
        } else {
            card.style.background = GRADIENTS[i % GRADIENTS.length];
            const emojiDiv = document.createElement('div');
            emojiDiv.className = 'style-emoji';
            emojiDiv.textContent = s.emoji || '🎨';
            card.appendChild(emojiDiv);
        }

        const info = document.createElement('div');
        info.className = 'style-card-info';
        const label = document.createElement('span');
        label.className = 'style-label';
        label.textContent = (s.emoji || '') + ' ' + s.label;
        info.appendChild(label);
        card.appendChild(info);
        grid.appendChild(card);
    });
}

function selectV3Category(slug) {
    state.v3SelectedCategory = slug;
    state.v3SelectedTags = [];
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
    loadExpressCatalog(true);
}

function toggleV3Tag(slug) {
    const idx = state.v3SelectedTags.indexOf(slug);
    if (idx >= 0) {
        state.v3SelectedTags.splice(idx, 1);
    } else {
        state.v3SelectedTags.push(slug);
    }
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
    loadExpressCatalog(true);
}

function selectV3Style(style) {
    state.selectedExpressStyle = style;
    state.expressFile = null;
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
    document.getElementById('express-selected-style').textContent = `${style.emoji || ''} ${style.label}`;
    document.getElementById('express-upload-zone').style.display = '';
    document.getElementById('express-preview').style.display = 'none';
    document.getElementById('express-generate-btn').style.display = 'none';
    // Show provider choice in V3
    const providerChoice = document.getElementById('provider-choice');
    if (providerChoice) {
        providerChoice.style.display = '';
        updateProviderUI();
    }
    showScreen('express-upload');
}

// Provider choice
function selectProvider(provider) {
    state.selectedProvider = provider;
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
    updateProviderUI();
}

function updateProviderUI() {
    document.querySelectorAll('.provider-option').forEach(el => {
        el.classList.toggle('selected', el.dataset.provider === state.selectedProvider);
    });
}

function showProviderInfo() {
    document.getElementById('provider-tooltip').style.display = '';
}

function hideProviderInfo() {
    document.getElementById('provider-tooltip').style.display = 'none';
}

// === PHOTOSETS FLOW ===

async function loadPhotosets() {
    showScreen('photosets');
    const grid = document.getElementById('photosets-grid');
    grid.innerHTML = Array(4).fill('<div class="photoset-card skeleton"><div class="photoset-card-cover-placeholder"></div><div class="photoset-card-info"><div style="height:14px;width:80%;background:var(--bg-glass);border-radius:4px"></div></div></div>').join('');

    try {
        const resp = await fetch('/app/api/v2/photosets', {
            headers: { 'X-Telegram-Init-Data': state.initData },
        });
        const data = await resp.json();

        state.photosets = (data.photosets || []).map(item => ({
            ...item,
            preview_urls: Array.isArray(item.preview_urls) ? item.preview_urls : [],
        }));
        state.packsUseCredits = !!data.packs_use_credits;
        state.packs = state.photosets
            .filter(item => item.type === 'pack')
            .map(item => ({ id: item.entity_id, title: item.title, credit_cost: item.credit_cost, num_images: item.num_images, preview_urls: item.preview_urls }));
        state.personaStyles = state.photosets
            .filter(item => item.type === 'style')
            .map(item => ({
                id: item.entity_id,
                slug: item.slug || '',
                title: item.title,
                description: item.description || '',
                credit_cost: item.credit_cost,
                num_images: item.num_images,
                preview_urls: item.preview_urls,
                image_url: item.preview_urls?.[0] || '',
            }));

        renderPhotosets();
    } catch (e) {
        console.error('Load photosets error:', e);
        grid.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-secondary)">Ошибка загрузки</div>';
    }
}

function renderPhotosetCardPreview(item) {
    const previews = (item.preview_urls || []).filter(Boolean);
    if (previews.length > 0) {
        return `<img class="photoset-card-cover" src="${previews[0]}" alt="" loading="lazy">`;
    }
    return `<div class="photoset-card-cover-placeholder">📸</div>`;
}

function renderPhotosets() {
    const grid = document.getElementById('photosets-grid');
    const items = state.photosets || [];

    if (!items.length) {
        grid.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-secondary);grid-column:1/-1">Фотосеты скоро появятся</div>';
        return;
    }

    grid.innerHTML = items.map((item, i) => {
        const creditCost = Number(item.credit_cost || 0);
        const numImages = Number(item.num_images || 0);
        const openAction = item.type === 'pack' ? `openPackDetail(${item.entity_id})` : `openStyleDetail(${item.entity_id})`;
        return `
            <div class="photoset-card fade-in" style="animation-delay:${i * 0.05}s" onclick="${openAction}">
                ${renderPhotosetCardPreview(item)}
                <div class="photoset-card-info">
                    <div class="photoset-card-title">${item.title || ''}</div>
                    <div class="photoset-card-meta">
                        <span class="photoset-card-price">${numImages} фото · &#128142; ${creditCost}</span>
                    </div>
                </div>
            </div>`;
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

    const creditCost = style.credit_cost;
    document.getElementById('photoset-style-cost-value').textContent = creditCost;
    document.getElementById('photoset-style-cost-word').textContent = pluralCredits(creditCost);

    // Preview grid (2x2 из preview_urls, fallback до 4 плейсхолдеров)
    const preview = document.getElementById('photoset-style-preview');
    const previews = (style.preview_urls || []).filter(Boolean).slice(0, 4);
    const previewTiles = previews.map(url => `<img src="${url}" alt="" loading="lazy">`);
    while (previewTiles.length < 4) {
        previewTiles.push('<div class="preview-placeholder">?</div>');
    }
    preview.innerHTML = previewTiles.join('');

    // Generate button — gate by persona
    const generateBtn = document.getElementById('photoset-generate-btn');
    const noPersonaBlock = document.getElementById('photoset-no-persona');
    if (state.hasPersona) {
        generateBtn.style.display = '';
        noPersonaBlock.style.display = 'none';
    } else {
        generateBtn.style.display = 'none';
        noPersonaBlock.style.display = '';
    }

    showScreen('photoset-style-detail');
}

// Style batch generation (Astria via bot)
async function startStyleBatchGeneration() {
    if (!state.selectedPersonaStyle) return;

    const style = state.selectedPersonaStyle;
    const creditCost = style.credit_cost;

    if (state.photosetsCredits < creditCost) {
        document.getElementById('nocredits-text').textContent =
            `Нужно ${creditCost} ${pluralCredits(creditCost)}, у вас ${state.photosetsCredits}.`;
        showScreen('nocredits');
        return;
    }

    trackEvent('v2_style_batch_start', { style_id: style.id, slug: style.slug });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('heavy');

    const generateBtn = document.getElementById('photoset-generate-btn');
    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').textContent = 'Отправляем...';

    try {
        const resp = await fetch('/app/api/persona/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                init_data: state.initData,
                styles: [{ slug: style.slug }],
            }),
        });
        const data = await resp.json();

        if (!resp.ok) {
            if (resp.status === 402) {
                showScreen('nocredits');
                return;
            }
            throw new Error(data.error || 'Request failed');
        }

        trackEvent('v2_style_batch_sent', { style_id: style.id, count: data.count });

        if (data.bot_link && tg?.openTelegramLink) {
            tg.openTelegramLink(data.bot_link);
        } else if (data.bot_link) {
            window.open(data.bot_link, '_blank');
        }

        if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
    } catch (e) {
        console.error('Style batch error:', e);
        alert('Ошибка: ' + e.message);
        if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('error');
    } finally {
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').textContent = '\u2728 Сгенерировать (' + (style.num_images || 4) + ' фото)';
    }
}

function goToCreatePersona() {
    trackEvent('v2_style_create_persona');
    const botUsername = tg?.initDataUnsafe?.bot?.username;
    if (botUsername && tg?.openTelegramLink) {
        tg.openTelegramLink('https://t.me/' + botUsername);
    } else if (botUsername) {
        window.open('https://t.me/' + botUsername, '_blank');
    }
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


    // Custom prompt character counter
    const promptInput = document.getElementById('custom-prompt-input');
    if (promptInput) {
        promptInput.addEventListener('input', () => {
            const el = document.getElementById('custom-prompt-length');
            if (el) el.textContent = promptInput.value.length;
        });
    }
}


// === CUSTOM PROMPT FLOW ===

async function loadCustomCapabilities() {
    if (state.customCapabilities) return;
    try {
        const resp = await fetch('/app/api/v3/custom/capabilities');
        state.customCapabilities = await resp.json();
    } catch (e) {
        console.error('Load capabilities error:', e);
        state.customCapabilities = {
            providers: { seedream: { max_photos: 14 }, 'nano-banana-pro': { max_photos: 8 } },
            max_prompt_length: 2000, allowed_mime: ['image/jpeg','image/png','image/webp'], max_file_size_mb: 15,
        };
    }
}

function goToCustomPrompt() {
    showScreen('custom-prompt');
    loadCustomCapabilities().then(() => {
        updateCustomMaxPhotos();
        updateBalanceDisplays();
    });
    state.customFiles = [];
    state.customProvider = 'seedream';
    state.customRequestId = null;
    renderCustomPhotos();
    const textarea = document.getElementById('custom-prompt-input');
    if (textarea) { textarea.value = ''; }
    const counter = document.getElementById('custom-prompt-length');
    if (counter) counter.textContent = '0';
    // Reset provider buttons
    document.querySelectorAll('#custom-provider-options .provider-option').forEach(btn => {
        btn.classList.toggle('selected', btn.dataset.provider === 'seedream');
    });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
}

function selectCustomProvider(provider) {
    state.customProvider = provider;
    document.querySelectorAll('#custom-provider-options .provider-option').forEach(btn => {
        btn.classList.toggle('selected', btn.dataset.provider === provider);
    });
    updateCustomMaxPhotos();
    // Trim photos if exceeds new limit
    const caps = state.customCapabilities?.providers?.[provider];
    const max = caps?.max_photos || 8;
    if (state.customFiles.length > max) {
        state.customFiles = state.customFiles.slice(0, max);
        renderCustomPhotos();
    }
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
}

function updateCustomMaxPhotos() {
    const caps = state.customCapabilities?.providers?.[state.customProvider];
    const max = caps?.max_photos || 8;
    const el = document.getElementById('custom-max-photos');
    if (el) el.textContent = max;
    // Hide add button if at limit
    const addBtn = document.getElementById('custom-photo-add-btn');
    if (addBtn) addBtn.style.display = state.customFiles.length >= max ? 'none' : '';
}

function handleCustomFiles(event) {
    const files = Array.from(event.target.files || []);
    const caps = state.customCapabilities?.providers?.[state.customProvider];
    const max = caps?.max_photos || 8;
    const maxSize = (state.customCapabilities?.max_file_size_mb || 15) * 1024 * 1024;

    for (const file of files) {
        if (state.customFiles.length >= max) break;
        if (file.size > maxSize) {
            alert(`Файл ${file.name} слишком большой (макс ${state.customCapabilities?.max_file_size_mb || 15} МБ)`);
            continue;
        }
        state.customFiles.push(file);
    }
    renderCustomPhotos();
    updateCustomMaxPhotos();
    event.target.value = '';
}

function removeCustomPhoto(index) {
    state.customFiles.splice(index, 1);
    renderCustomPhotos();
    updateCustomMaxPhotos();
}

function renderCustomPhotos() {
    const grid = document.getElementById('custom-photos-grid');
    if (!grid) return;
    // Keep add button, remove previews
    grid.querySelectorAll('.custom-photo-preview').forEach(el => el.remove());
    const addBtn = document.getElementById('custom-photo-add-btn');

    state.customFiles.forEach((file, i) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'custom-photo-preview';
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.onload = () => URL.revokeObjectURL(img.src);
        wrapper.appendChild(img);
        const removeBtn = document.createElement('button');
        removeBtn.className = 'custom-photo-remove';
        removeBtn.textContent = '\u00d7';
        removeBtn.onclick = () => removeCustomPhoto(i);
        wrapper.appendChild(removeBtn);
        grid.insertBefore(wrapper, addBtn);
    });
}

function _generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}

async function startCustomGeneration() {
    const prompt = (document.getElementById('custom-prompt-input')?.value || '').trim();
    if (!prompt) {
        if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('error');
        document.getElementById('custom-prompt-input')?.focus();
        return;
    }

    const btn = document.getElementById('custom-generate-btn');
    if (btn) { btn.disabled = true; btn.querySelector('.btn-text').textContent = 'Генерация...'; }

    const requestId = _generateUUID();
    state.customRequestId = requestId;

    const formData = new FormData();
    formData.append('init_data', state.initData);
    formData.append('prompt', prompt);
    formData.append('provider', state.customProvider);
    formData.append('request_id', requestId);
    state.customFiles.forEach(file => formData.append('photos', file));

    try {
        const resp = await fetch('/app/api/v3/custom/generate', { method: 'POST', body: formData });
        const data = await resp.json();

        if (!resp.ok) {
            if (data.error === 'no_credits') {
                document.getElementById('nocredits-text').textContent = 'Для генерации нужны кредиты Экспресс';
                showScreen('nocredits');
                return;
            }
            throw new Error(data.error || data.message || 'Generation failed');
        }

        state.customTaskId = data.task_id;
        showScreen('express-generating');
        pollGenerationStatus(data.task_id);

    } catch (e) {
        console.error('Custom generation error:', e);
        alert('Ошибка: ' + e.message);
    } finally {
        if (btn) { btn.disabled = false; btn.querySelector('.btn-text').textContent = '✨ Создать'; }
    }
}
