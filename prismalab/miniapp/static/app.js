/**
 * PrismaLab Mini App — Frontend Logic
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
    credits: { fast: 0, free_used: false },
    selectedStyle: null,
    selectedFile: null,
    taskId: null,
    packsEnabled: false,
    packs: [],
    selectedPack: null,
    personaStyles: [],
    selectedPersonaStyle: null,
    personaStyleGenderFilter: '',
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

    // Drag & drop
    setupDragDrop();

    // Auth
    authenticate();
});

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
            state.credits = data.credits;
            state.gender = data.gender;
            state.packsEnabled = !!data.packs_enabled;
            state.hasPersona = !!data.has_persona;
            state.personaCredits = data.persona_credits || 0;
            state.personaCreditsOriginal = state.personaCredits;
            updateCreditsDisplay();
            const mainCreditsEl = document.getElementById('main-persona-credits');
            if (mainCreditsEl) mainCreditsEl.textContent = state.personaCredits;
            const mainCreditsWord = document.getElementById('main-persona-credits-word');
            if (mainCreditsWord) mainCreditsWord.textContent = pluralCredits(state.personaCredits);

            // Проверяем возврат из ЮKassa после оплаты пака
            const params = new URLSearchParams(window.location.search);
            if (params.get('pack_paid')) {
                showScreen('pack-paid');
                return;
            }

            trackEvent('miniapp_open');

            // Пол не известен — спрашиваем и сохраняем в профиль
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
                <h2 style="margin:0 0 12px 0;font-size:22px;">Mini App недоступна</h2>
                <p style="margin:0;color:#b5b7c0;font-size:15px;line-height:1.5;">${message}</p>
            </div>
        </div>
    `;
}

// === Navigation ===

function showScreen(name) {
    const screens = document.querySelectorAll('.screen');
    screens.forEach(s => {
        if (s.classList.contains('active')) {
            s.classList.add('slide-out');
            setTimeout(() => {
                s.classList.remove('active', 'slide-out');
            }, 300);
        }
    });

    setTimeout(() => {
        const target = document.getElementById(`screen-${name}`);
        if (target) {
            target.classList.add('active');
        }
    }, 50);

    // Показать/скрыть фиксированные футеры тарифов и генерации
    const createFooter = document.getElementById('persona-buy-footer');
    const topupFooter = document.getElementById('persona-topup-footer');
    const generateFooter = document.getElementById('persona-generate-footer');
    if (name === 'persona-styles') {
        if (!state.hasPersona) {
            if (createFooter) createFooter.style.display = 'flex';
            if (topupFooter) topupFooter.style.display = 'none';
            if (generateFooter) generateFooter.style.display = 'none';
        } else if (state.personaCredits === 0) {
            if (createFooter) createFooter.style.display = 'none';
            if (topupFooter) topupFooter.style.display = 'flex';
            if (generateFooter) generateFooter.style.display = 'none';
        } else {
            if (createFooter) createFooter.style.display = 'none';
            if (topupFooter) topupFooter.style.display = 'none';
            // Генерация показывается только если выбраны стили
            updateGenerateButton();
        }
    } else {
        if (createFooter) createFooter.style.display = 'none';
        if (topupFooter) topupFooter.style.display = 'none';
        if (generateFooter) generateFooter.style.display = 'none';
    }

    // Haptic feedback
    if (tg?.HapticFeedback) {
        tg.HapticFeedback.impactOccurred('light');
    }
}

function goBack(screen) {
    showScreen(screen);
    if (tg?.HapticFeedback) {
        tg.HapticFeedback.impactOccurred('light');
    }
}

// === Gender ===

async function selectGender(gender) {
    state.gender = gender;
    trackEvent('miniapp_gender_select', { gender });

    if (tg?.HapticFeedback) {
        tg.HapticFeedback.impactOccurred('medium');
    }

    // Сохраняем пол в профиль (как в основном боте)
    try {
        await fetch('/app/api/profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ init_data: state.initData, gender }),
        });
    } catch (e) {
        console.warn('Profile save error:', e);
    }

    showScreen('main');
}

// === Styles ===

async function loadStyles(gender) {
    const grid = document.getElementById('styles-grid');

    // Show skeleton loading
    grid.innerHTML = '';
    for (let i = 0; i < 6; i++) {
        const skeleton = document.createElement('div');
        skeleton.className = 'style-card skeleton';
        grid.appendChild(skeleton);
    }

    try {
        const headers = {};
        if (state.initData) headers['X-Telegram-Init-Data'] = state.initData;
        const resp = await fetch(`/app/api/styles?gender=${gender}`, { headers });
        if (resp.status === 403) {
            showBlocked('Mini App сейчас доступна только тестовому аккаунту.');
            return;
        }
        const data = await resp.json();

        grid.innerHTML = '';

        data.styles.forEach((style, index) => {
            const card = document.createElement('div');
            card.className = 'style-card';
            card.style.animationDelay = `${index * 0.05}s`;
            card.onclick = () => selectStyle(style);

            // Генерируем уникальный градиент для каждой карточки
            const hue1 = (index * 37 + 240) % 360;
            const hue2 = (hue1 + 40) % 360;

            card.innerHTML = `
                <div class="style-card-bg" style="background: linear-gradient(135deg, hsl(${hue1}, 60%, 25%), hsl(${hue2}, 70%, 15%)); opacity: 0.4;"></div>
                <div class="style-card-emoji">${style.emoji}</div>
                <div class="style-card-info">
                    <div class="style-card-name">${style.emoji} ${style.label}</div>
                </div>
            `;

            grid.appendChild(card);
        });
    } catch (e) {
        console.error('Load styles error:', e);
        grid.innerHTML = '<p style="color: var(--text-secondary); padding: 20px; text-align: center; grid-column: 1/-1;">Ошибка загрузки стилей</p>';
    }
}

function selectStyle(style) {
    state.selectedStyle = style;
    state.selectedFile = null;
    trackEvent('fast_style_select', { style_id: style.id });

    document.getElementById('selected-style-name').textContent = `${style.emoji} ${style.label}`;

    // Reset upload
    resetUpload();
    showScreen('upload');

    if (tg?.HapticFeedback) {
        tg.HapticFeedback.impactOccurred('medium');
    }
}

// === Upload ===

function setupDragDrop() {
    const zone = document.getElementById('upload-zone');
    if (!zone) return;

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
        }
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (file.size > 15 * 1024 * 1024) {
        alert('Файл слишком большой (максимум 15 МБ)');
        return;
    }

    state.selectedFile = file;
    trackEvent('fast_upload');

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('preview-image').src = e.target.result;
        document.getElementById('upload-zone').style.display = 'none';
        document.getElementById('upload-preview').style.display = 'block';
        document.getElementById('generate-btn').style.display = 'block';
    };
    reader.readAsDataURL(file);

    if (tg?.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('success');
    }
}

function resetUpload() {
    state.selectedFile = null;
    document.getElementById('upload-zone').style.display = 'flex';
    document.getElementById('upload-preview').style.display = 'none';
    document.getElementById('generate-btn').style.display = 'none';
    document.getElementById('file-input').value = '';
}

// === Generation ===

async function startGeneration() {
    if (!state.selectedFile || !state.selectedStyle) return;
    trackEvent('fast_generate_start', { style_id: state.selectedStyle.id });

    if (tg?.HapticFeedback) {
        tg.HapticFeedback.impactOccurred('heavy');
    }

    showScreen('generating');
    startProgressAnimation();

    const formData = new FormData();
    formData.append('init_data', state.initData);
    formData.append('style_id', state.selectedStyle.id);
    formData.append('photo', state.selectedFile);

    try {
        const resp = await fetch('/app/api/generate', {
            method: 'POST',
            body: formData,
        });

        const data = await resp.json();

        if (resp.status === 402) {
            showScreen('nocredits');
            return;
        }

        if (!resp.ok) {
            throw new Error(data.error || 'Generation failed');
        }

        state.taskId = data.task_id;
        pollStatus();
    } catch (e) {
        console.error('Generation error:', e);
        alert('Ошибка генерации: ' + e.message);
        showScreen('upload');
    }
}

let progressInterval = null;
let progressValue = 0;

function startProgressAnimation() {
    progressValue = 0;
    const bar = document.getElementById('progress-bar');
    bar.style.width = '0%';

    const tips = [
        'AI анализирует ваше фото...',
        'Применяем стиль...',
        'Добавляем детали...',
        'Финальные штрихи...',
        'Почти готово...',
    ];

    let tipIndex = 0;

    progressInterval = setInterval(() => {
        if (progressValue < 90) {
            progressValue += Math.random() * 3 + 0.5;
            bar.style.width = Math.min(progressValue, 90) + '%';
        }

        if (progressValue > tipIndex * 20 + 10 && tipIndex < tips.length) {
            document.getElementById('generating-tip').textContent = tips[tipIndex];
            tipIndex++;
        }
    }, 500);
}

function stopProgressAnimation() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    const bar = document.getElementById('progress-bar');
    bar.style.width = '100%';
}

async function pollStatus() {
    if (!state.taskId) return;

    try {
        const headers = {};
        if (state.initData) headers['X-Telegram-Init-Data'] = state.initData;
        const resp = await fetch(`/app/api/status/${state.taskId}`, { headers });
        if (resp.status === 403) {
            stopProgressAnimation();
            showBlocked('Mini App сейчас доступна только тестовому аккаунту.');
            return;
        }
        const data = await resp.json();

        if (data.status === 'done') {
            stopProgressAnimation();
            trackEvent('fast_generate_done', { style_id: state.selectedStyle?.id });

            // Update credits
            if (state.credits.free_used === false) {
                state.credits.free_used = true;
            } else {
                state.credits.fast = Math.max(0, state.credits.fast - 1);
            }
            updateCreditsDisplay();

            // Show result
            document.getElementById('result-image').src = data.result_url;
            showScreen('result');

            if (tg?.HapticFeedback) {
                tg.HapticFeedback.notificationOccurred('success');
            }
        } else if (data.status === 'error') {
            stopProgressAnimation();
            alert('Ошибка генерации. Попробуйте ещё раз.');
            showScreen('upload');

            if (tg?.HapticFeedback) {
                tg.HapticFeedback.notificationOccurred('error');
            }
        } else {
            // Still processing — poll again
            setTimeout(pollStatus, 2000);
        }
    } catch (e) {
        console.error('Poll error:', e);
        setTimeout(pollStatus, 3000);
    }
}

// === Result ===

function downloadResult() {
    const img = document.getElementById('result-image');
    if (!img.src) return;
    trackEvent('fast_download', { style_id: state.selectedStyle?.id });

    const link = document.createElement('a');
    link.href = img.src;
    link.download = `prismalab_${state.selectedStyle?.id || 'photo'}.jpg`;
    link.click();

    if (tg?.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('success');
    }
}

function tryAnotherStyle() {
    trackEvent('fast_try_another');
    state.selectedFile = null;
    state.selectedStyle = null;
    state.taskId = null;
    showScreen('styles');
}

// === Credits ===

function updateCreditsDisplay() {
    const badge = document.getElementById('credits-count');
    if (!badge) return;

    let total = state.credits.fast;
    if (!state.credits.free_used) total += 1;

    badge.textContent = total;
}

function buyCredits() {
    // Закрываем Mini App и отправляем в бота
    if (tg) {
        tg.sendData(JSON.stringify({ action: 'buy_credits' }));
        tg.close();
    }
}

// === Main Menu ===

async function goToPacks() {
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
    trackEvent('nav_packs');
    state.packCategory = state.packCategory || 'female';
    showScreen('packs');
    await loadPacks();
}

function goToProfile() {
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
    trackEvent('nav_profile');
    renderProfile();
    showScreen('profile');
}

function renderProfile() {
    const fastTotal = (state.credits?.fast || 0) + (state.credits?.free_used ? 0 : 1);
    document.getElementById('profile-fast-credits').textContent = fastTotal;
    document.getElementById('profile-persona-credits').textContent = state.personaCredits ?? 0;
    document.getElementById('profile-personas-count').textContent = state.hasPersona ? '1' : '0';
    document.getElementById('profile-gender').textContent = state.gender === 'male' ? 'Мужской 👨' : (state.gender === 'female' ? 'Женский 👩' : '—');
}

function selectPackCategory(category) {
    state.packCategory = category;
    trackEvent('pack_category_select', { category });
    document.querySelectorAll('.category-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.category === category);
    });
    renderPacksByCategory();
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
}

// === Packs ===

async function loadPacks() {
    const grid = document.getElementById('packs-grid');
    grid.innerHTML = `
        <div class="packs-loading" style="grid-column: 1/-1;">
            <div class="packs-prism"></div>
            <p class="packs-loading-text">Нужно немного времени, чтобы загрузить фотосеты. Пожалуйста, никуда не убегайте</p>
            <div class="packs-loading-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

    try {
        const headers = {};
        if (state.initData) headers['X-Telegram-Init-Data'] = state.initData;
        const resp = await fetch('/app/api/packs', { headers });
        const data = await resp.json();
        state.packs = data.packs || [];
        state.packCategory = state.packCategory || 'female';

        document.querySelectorAll('.category-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.category === state.packCategory);
        });

        renderPacksByCategory();
    } catch (e) {
        console.error('Load packs error:', e);
        grid.innerHTML = '<p style="color: var(--text-secondary); padding: 20px; text-align: center; grid-column: 1/-1;">Ошибка загрузки паков</p>';
    }
}

function renderPacksByCategory() {
    const grid = document.getElementById('packs-grid');
    grid.innerHTML = '';

    const category = state.packCategory || 'female';
    const filtered = (state.packs || []).filter(p => (p.category || 'female') === category);

    if (filtered.length === 0) {
        grid.innerHTML = '<p style="color: var(--text-secondary); padding: 20px; text-align: center; grid-column: 1/-1;">В этой категории пока нет фотосетов</p>';
        return;
    }

    filtered.forEach((pack, index) => {
        const card = document.createElement('div');
        card.className = 'pack-card';
        card.style.animationDelay = `${index * 0.08}s`;
        card.onclick = () => openPackDetail(pack.id);

        const coverStyle = pack.cover_url
            ? `background-image: url(${pack.cover_url}); background-size: cover; background-position: center;`
            : `background: linear-gradient(135deg, hsl(${index * 60 + 200}, 60%, 25%), hsl(${index * 60 + 240}, 70%, 15%));`;

        card.innerHTML = `
            <div class="pack-card-cover" style="${coverStyle}"></div>
            <div class="pack-card-info">
                <div class="pack-card-title">${pack.title}</div>
                <div class="pack-card-meta">${pack.expected_images} фото</div>
                <div class="pack-card-price">${pack.price_rub} &#8381;</div>
            </div>
        `;

        grid.appendChild(card);
    });
}

async function openPackDetail(packId) {
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
    trackEvent('pack_detail_view', { pack_id: packId });

    showScreen('pack-detail');

    const gallery = document.getElementById('pack-gallery');
    gallery.innerHTML = '<div class="pack-gallery-loading">Загрузка примеров...</div>';

    try {
        const headers = {};
        if (state.initData) headers['X-Telegram-Init-Data'] = state.initData;
        const resp = await fetch(`/app/api/packs/${packId}`, { headers });
        const pack = await resp.json();
        state.selectedPack = pack;

        document.getElementById('pack-detail-title').textContent = pack.title;
        document.getElementById('pack-detail-count').textContent = pack.expected_images;
        document.getElementById('pack-buy-text').textContent = `Купить ${pack.price_rub} \u20BD`;

        // Галерея примеров
        gallery.innerHTML = '';
        if (pack.examples && pack.examples.length > 0) {
            pack.examples.forEach(url => {
                const img = document.createElement('img');
                img.src = url;
                img.className = 'pack-gallery-img';
                img.loading = 'lazy';
                img.onclick = (e) => { e.stopPropagation(); openLightbox(url); };
                gallery.appendChild(img);
            });
        } else {
            gallery.innerHTML = '<div class="pack-gallery-loading">Нет примеров</div>';
        }
    } catch (e) {
        console.error('Pack detail error:', e);
        gallery.innerHTML = '<div class="pack-gallery-loading">Ошибка загрузки</div>';
    }
}

async function buyPack() {
    if (!state.selectedPack) return;
    trackEvent('pack_buy', { pack_id: state.selectedPack.id });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('heavy');

    const btn = document.getElementById('pack-buy-btn');
    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = 'Создаём платёж...';

    try {
        const resp = await fetch(`/app/api/packs/${state.selectedPack.id}/buy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ init_data: state.initData }),
        });

        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.error || 'Payment failed');
        }

        // Открываем ссылку на оплату
        if (data.payment_url && tg) {
            tg.openLink(data.payment_url);
        } else if (data.payment_url) {
            window.open(data.payment_url, '_blank');
        }
    } catch (e) {
        console.error('Buy pack error:', e);
        alert('Ошибка создания платежа: ' + e.message);
    } finally {
        btn.disabled = false;
        if (state.selectedPack) {
            btn.querySelector('.btn-text').textContent = `Купить ${state.selectedPack.price_rub} \u20BD`;
        }
    }
}

function openLightbox(src) {
    document.getElementById('lightbox-img').src = src;
    document.getElementById('lightbox').style.display = 'flex';
}

function closeLightbox() {
    document.getElementById('lightbox').style.display = 'none';
}

function closeMiniApp() {
    if (tg) {
        tg.close();
    }
}

// === Persona Styles ===

async function goToPersonaStyles() {
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
    trackEvent('nav_persona');
    showScreen('persona-styles');
    await loadPersonaStyles();
}

async function loadPersonaStyles() {
    const grid = document.getElementById('persona-styles-grid');
    grid.innerHTML = '';
    for (let i = 0; i < 4; i++) {
        const skeleton = document.createElement('div');
        skeleton.className = 'persona-style-card skeleton';
        grid.appendChild(skeleton);
    }

    // Три стейта:
    // 1) нет персоны → витрина + тарифы создания
    // 2) есть персона, 0 кредитов → витрина + тарифы докупа
    // 3) есть персона, credits > 0 → чекбоксы + баланс
    const needFooter = !state.hasPersona || (state.hasPersona && state.personaCredits === 0);
    const info = document.getElementById('persona-buy-info');
    const createFooter = document.getElementById('persona-buy-footer');
    const topupFooter = document.getElementById('persona-topup-footer');
    if (info) {
        info.style.display = needFooter ? 'block' : 'none';
        const infoText = info.querySelector('.persona-buy-info-text');
        if (infoText) {
            infoText.textContent = (state.hasPersona && state.personaCredits === 0)
                ? 'Ваша Персона уже создана, поэтому эти тарифы дешевле'
                : '1 кредит = 1 фото. Образы станут доступны после оплаты тарифа';
        }
    }
    // Показываем нужный футер
    if (!state.hasPersona) {
        if (createFooter) createFooter.style.display = 'flex';
        if (topupFooter) topupFooter.style.display = 'none';
    } else if (state.personaCredits === 0) {
        if (createFooter) createFooter.style.display = 'none';
        if (topupFooter) topupFooter.style.display = 'flex';
    } else {
        if (createFooter) createFooter.style.display = 'none';
        if (topupFooter) topupFooter.style.display = 'none';
    }

    // Показать баланс-бейдж если есть персона
    updatePersonaBalanceDisplay();

    // Грид padding: с футером тарифов — 120px, с кнопкой генерации — 80px, без — 32px
    if (needFooter) {
        grid.style.paddingBottom = '120px';
    } else if (state.hasPersona && state.personaCredits > 0) {
        grid.style.paddingBottom = '80px';
    } else {
        grid.style.paddingBottom = '32px';
    }

    // Сбросить выбор тарифов при загрузке
    state.selectedTariff = null;
    state.selectedTopupTariff = null;
    if (!state.selectedPersonaStyles) state.selectedPersonaStyles = [];
    updateTariffSelection();
    updateTopupTariffSelection();

    try {
        // Всегда грузим ВСЕ стили, фильтруем на клиенте
        const resp = await fetch('/app/api/persona-styles');
        const data = await resp.json();
        state.allPersonaStyles = Array.isArray(data) ? data : (data.styles || []);

        // Показываем/скрываем фильтры по наличию полов
        const hasMale = state.allPersonaStyles.some(s => s.gender === 'male');
        const hasFemale = state.allPersonaStyles.some(s => s.gender === 'female');
        const filtersEl = document.getElementById('persona-styles-filters');
        filtersEl.style.display = (hasMale && hasFemale) ? 'flex' : 'none';

        // Если фильтры скрыты — сбросить фильтр
        if (!(hasMale && hasFemale)) state.personaStyleGenderFilter = '';

        applyPersonaStyleFilter();
    } catch (e) {
        console.error('Load persona styles error:', e);
        grid.innerHTML = '<p style="color: var(--text-secondary); padding: 20px; text-align: center; grid-column: 1/-1;">Ошибка загрузки стилей</p>';
    }
}

function renderPersonaStyles() {
    const grid = document.getElementById('persona-styles-grid');
    grid.innerHTML = '';

    const styles = state.personaStyles;
    if (!styles.length) {
        grid.innerHTML = '<p style="color: var(--text-secondary); padding: 40px 20px; text-align: center; grid-column: 1/-1;">Стили скоро появятся</p>';
        return;
    }

    if (!state.selectedPersonaStyles) state.selectedPersonaStyles = [];

    const hasPersona = !!state.hasPersona;
    // Чекбоксы показываем если есть персона И изначально были кредиты
    // (personaCredits может быть 0 после выбора всех стилей — это ок)
    const canSelect = (state.personaCreditsOriginal || state.personaCredits) > 0;
    const showCheckboxes = hasPersona && canSelect;

    styles.forEach((style, index) => {
        const card = document.createElement('div');
        card.className = 'persona-style-card';
        if (!showCheckboxes) card.classList.add('showcase');
        card.dataset.styleId = style.id;
        card.style.animationDelay = `${index * 0.05}s`;

        const isSelected = showCheckboxes && state.selectedPersonaStyles.some(s => s.id === style.id);
        if (isSelected) card.classList.add('selected');

        const imgStyle = style.image_url
            ? `background-image: url('${style.image_url}')`
            : `background: linear-gradient(135deg, hsl(${(index * 47 + 200) % 360}, 50%, 30%), hsl(${((index * 47 + 200) % 360 + 50) % 360}, 60%, 20%))`;

        // Чекбокс: только если есть персона И кредиты > 0
        const checkboxHtml = showCheckboxes
            ? `<div class="persona-style-card-checkbox ${isSelected ? 'checked' : ''}" onclick="event.stopPropagation(); togglePersonaStyle(${style.id})">${isSelected ? '✓' : ''}</div>`
            : '';

        card.innerHTML = `
            <div class="persona-style-card-img" style="${imgStyle}"></div>
            <div class="persona-style-card-overlay">
                <div class="persona-style-card-title">${style.title}</div>
            </div>
            ${checkboxHtml}
        `;

        // Тап по карточке — лайтбокс для просмотра
        card.onclick = () => openPersonaStyleLightbox(style);

        grid.appendChild(card);
    });
}

function togglePersonaStyle(styleId) {
    if (!state.selectedPersonaStyles) state.selectedPersonaStyles = [];
    const idx = state.selectedPersonaStyles.findIndex(s => s.id === styleId);
    if (idx >= 0) {
        // Убираем — возвращаем кредит
        state.selectedPersonaStyles.splice(idx, 1);
        state.personaCredits++;
        if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
    } else {
        // Добавляем — проверяем баланс
        if (state.personaCredits <= 0) {
            if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('error');
            return;
        }
        const style = state.personaStyles.find(s => s.id === styleId);
        if (style) {
            state.selectedPersonaStyles.push(style);
            state.personaCredits--;
            trackEvent('persona_style_select', { style_id: styleId });
            if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
        }
    }
    updatePersonaBalanceDisplay();
    updateGenerateButton();
    renderPersonaStyles();
}

function openPersonaStyleLightbox(style) {
    trackEvent('persona_style_view', { style_id: style.id, title: style.title });
    state._lightboxStyle = style;
    const lb = document.getElementById('persona-style-lightbox');
    document.getElementById('persona-style-lightbox-img').src = style.image_url || '';
    document.getElementById('persona-style-lightbox-title').textContent = style.title;
    document.getElementById('persona-style-lightbox-desc').textContent = style.description || '';

    const btn = document.getElementById('persona-style-lightbox-btn');
    const hasCreditsForSelection = (state.personaCreditsOriginal || state.personaCredits) > 0;
    if (state.hasPersona && hasCreditsForSelection) {
        // С персоной и кредитами — можно выбирать/убирать
        const isSelected = (state.selectedPersonaStyles || []).some(s => s.id === style.id);
        // Если уже выбран — можно убрать; если нет — только если остались кредиты
        const canAdd = state.personaCredits > 0;
        btn.textContent = isSelected ? 'Убрать' : (canAdd ? 'Выбрать' : 'Нет кредитов');
        btn.className = 'persona-style-lightbox-btn' + (isSelected ? ' deselect' : '');
        btn.disabled = !isSelected && !canAdd;
        btn.style.display = 'block';
    } else {
        // Без персоны или 0 кредитов изначально — только просмотр
        btn.style.display = 'none';
    }

    lb.style.display = 'flex';
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
}

function closePersonaStyleLightbox(e) {
    if (e && e.target !== e.currentTarget && !e.target.classList.contains('persona-style-lightbox-close')) return;
    document.getElementById('persona-style-lightbox').style.display = 'none';
    state._lightboxStyle = null;
}

function selectPersonaStyleFromLightbox() {
    const style = state._lightboxStyle;
    if (!style) return;

    togglePersonaStyle(style.id);

    // Закрыть лайтбокс
    document.getElementById('persona-style-lightbox').style.display = 'none';
    state._lightboxStyle = null;
}

function applyPersonaStyleFilter() {
    const gender = state.personaStyleGenderFilter || '';
    const all = state.allPersonaStyles || [];
    state.personaStyles = gender ? all.filter(s => s.gender === gender) : all;
    renderPersonaStyles();
}

function filterPersonaStyles(gender) {
    state.personaStyleGenderFilter = gender;
    trackEvent('persona_style_filter', { gender });
    document.querySelectorAll('#persona-styles-filters .category-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.gender === gender);
    });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
    applyPersonaStyleFilter();
}

// === Баланс персоны ===

function updatePersonaBalanceDisplay() {
    const badge = document.getElementById('persona-balance-badge');
    const count = document.getElementById('persona-balance-count');
    if (!badge || !count) return;

    if (state.hasPersona) {
        badge.style.display = 'flex';
        count.textContent = state.personaCredits;
        // Подсветка если мало кредитов
        badge.classList.toggle('low', state.personaCredits <= 2 && state.personaCredits > 0);
        badge.classList.toggle('empty', state.personaCredits === 0);
    } else {
        badge.style.display = 'none';
    }
}

// === Покупка персоны из витрины ===

function selectPersonaTariff(credits) {
    if (state.selectedTariff === credits) {
        state.selectedTariff = null;
    } else {
        state.selectedTariff = credits;
        trackEvent('persona_buy_init', { credits });
    }
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
    updateTariffSelection();
}

function updateTariffSelection() {
    const btns = document.querySelectorAll('.persona-buy-btn');
    btns.forEach(b => {
        const c = parseInt(b.dataset.credits);
        b.classList.toggle('selected', c === state.selectedTariff);
    });
    const payBtn = document.getElementById('persona-buy-pay-btn');
    if (payBtn) {
        payBtn.style.display = state.selectedTariff ? 'block' : 'none';
    }
}

async function buyPersona() {
    const credits = state.selectedTariff;
    if (!credits) return;
    trackEvent('persona_buy_confirm', { credits });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('heavy');

    const payBtn = document.getElementById('persona-buy-pay-btn');
    payBtn.disabled = true;
    payBtn.textContent = 'Создаём платёж...';

    try {
        const resp = await fetch('/app/api/persona/buy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ init_data: state.initData, credits }),
        });

        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.error || 'Payment failed');
        }

        if (data.payment_url && tg) {
            tg.openLink(data.payment_url);
        } else if (data.payment_url) {
            window.open(data.payment_url, '_blank');
        }
    } catch (e) {
        console.error('Buy persona error:', e);
        alert('Ошибка создания платежа: ' + e.message);
    } finally {
        payBtn.disabled = false;
        payBtn.textContent = 'Оплатить';
    }
}

// === Докуп кредитов персоны ===

function selectTopupTariff(credits) {
    if (state.selectedTopupTariff === credits) {
        state.selectedTopupTariff = null;
    } else {
        state.selectedTopupTariff = credits;
        trackEvent('persona_topup_init', { credits });
    }
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
    updateTopupTariffSelection();
}

function updateTopupTariffSelection() {
    const btns = document.querySelectorAll('#persona-topup-footer .persona-buy-btn');
    btns.forEach(b => {
        const c = parseInt(b.dataset.credits);
        b.classList.toggle('selected', c === state.selectedTopupTariff);
    });
    const payBtn = document.getElementById('persona-topup-pay-btn');
    if (payBtn) {
        payBtn.style.display = state.selectedTopupTariff ? 'block' : 'none';
    }
}

async function buyTopup() {
    const credits = state.selectedTopupTariff;
    if (!credits) return;
    trackEvent('persona_topup_confirm', { credits });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('heavy');

    const payBtn = document.getElementById('persona-topup-pay-btn');
    payBtn.disabled = true;
    payBtn.textContent = 'Создаём платёж...';

    try {
        const resp = await fetch('/app/api/persona/topup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ init_data: state.initData, credits }),
        });

        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.error || 'Payment failed');
        }

        if (data.payment_url && tg) {
            tg.openLink(data.payment_url);
        } else if (data.payment_url) {
            window.open(data.payment_url, '_blank');
        }
    } catch (e) {
        console.error('Buy topup error:', e);
        alert('Ошибка создания платежа: ' + e.message);
    } finally {
        payBtn.disabled = false;
        payBtn.textContent = 'Оплатить';
    }
}

// === Генерация батча персона-стилей ===

function updateGenerateButton() {
    const footer = document.getElementById('persona-generate-footer');
    const btnText = document.getElementById('persona-generate-btn-text');
    if (!footer) return;

    const selected = state.selectedPersonaStyles || [];
    if (state.hasPersona && state.personaCredits >= 0 && selected.length > 0) {
        footer.style.display = 'flex';
        btnText.textContent = `Сгенерировать (${selected.length})`;
    } else {
        footer.style.display = 'none';
    }
}

async function generatePersonaBatch() {
    const selected = state.selectedPersonaStyles || [];
    if (!selected.length) return;
    trackEvent('persona_generate_batch', { styles_count: selected.length });

    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('heavy');

    const btn = document.getElementById('persona-generate-btn');
    btn.disabled = true;
    const btnText = document.getElementById('persona-generate-btn-text');
    btnText.textContent = 'Отправляю...';

    // Собираем данные для бота
    const styles = selected.map(s => ({
        id: s.id,
        slug: s.slug,
        title: s.title,
    }));

    try {
        const resp = await fetch('/app/api/persona/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ init_data: state.initData, styles }),
        });

        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.error || 'Failed');
        }

        // Открываем deeplink в бот — бот подхватит pending batch
        if (data.bot_link && tg) {
            tg.openTelegramLink(data.bot_link);
            tg.close();
        } else if (data.bot_link) {
            window.open(data.bot_link, '_blank');
        }
    } catch (e) {
        console.error('Generate batch error:', e);
        alert('Ошибка: ' + e.message);
        btn.disabled = false;
        btnText.textContent = `Сгенерировать (${selected.length})`;
    }
}
