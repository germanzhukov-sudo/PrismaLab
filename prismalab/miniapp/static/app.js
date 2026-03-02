/**
 * PrismaLab Mini App — Frontend Logic
 */

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
            updateCreditsDisplay();

            // Проверяем возврат из ЮKassa после оплаты пака
            const params = new URLSearchParams(window.location.search);
            if (params.get('pack_paid')) {
                showScreen('pack-paid');
                return;
            }

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

    // Показать/скрыть фиксированный футер тарифов
    const footer = document.getElementById('persona-buy-footer');
    if (footer) {
        footer.style.display = (name === 'persona-styles' && !state.hasPersona) ? 'flex' : 'none';
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

    const link = document.createElement('a');
    link.href = img.src;
    link.download = `prismalab_${state.selectedStyle?.id || 'photo'}.jpg`;
    link.click();

    if (tg?.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('success');
    }
}

function tryAnotherStyle() {
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
    state.packCategory = state.packCategory || 'female';
    showScreen('packs');
    await loadPacks();
}

function goToProfile() {
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
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

    // Показать/скрыть инфо + футер покупки в зависимости от наличия персоны
    const info = document.getElementById('persona-buy-info');
    const footer = document.getElementById('persona-buy-footer');
    if (info) info.style.display = state.hasPersona ? 'none' : 'block';
    if (footer) footer.style.display = state.hasPersona ? 'none' : 'flex';

    // Показать баланс-бейдж если есть персона
    updatePersonaBalanceDisplay();

    // Грид padding: с футером — 120px, без — 32px
    grid.style.paddingBottom = state.hasPersona ? '32px' : '120px';

    // Сбросить выбор тарифа при загрузке
    state.selectedTariff = null;
    if (!state.selectedPersonaStyles) state.selectedPersonaStyles = [];
    updateTariffSelection();

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
    const canSelect = state.personaCredits > 0;

    styles.forEach((style, index) => {
        const card = document.createElement('div');
        card.className = 'persona-style-card';
        if (!hasPersona) card.classList.add('showcase');
        card.dataset.styleId = style.id;
        card.style.animationDelay = `${index * 0.05}s`;

        const isSelected = hasPersona && state.selectedPersonaStyles.some(s => s.id === style.id);
        if (isSelected) card.classList.add('selected');

        const imgStyle = style.image_url
            ? `background-image: url('${style.image_url}')`
            : `background: linear-gradient(135deg, hsl(${(index * 47 + 200) % 360}, 50%, 30%), hsl(${((index * 47 + 200) % 360 + 50) % 360}, 60%, 20%))`;

        // Чекбокс показываем только если есть персона
        // Дизейблим если баланс = 0 и стиль не выбран (выбранные можно убирать)
        const isDisabled = hasPersona && !isSelected && !canSelect;
        const checkboxHtml = hasPersona
            ? `<div class="persona-style-card-checkbox ${isSelected ? 'checked' : ''} ${isDisabled ? 'disabled' : ''}" onclick="event.stopPropagation(); togglePersonaStyle(${style.id})">${isSelected ? '✓' : ''}</div>`
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
            if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
        }
    }
    updatePersonaBalanceDisplay();
    renderPersonaStyles();
}

function openPersonaStyleLightbox(style) {
    state._lightboxStyle = style;
    const lb = document.getElementById('persona-style-lightbox');
    document.getElementById('persona-style-lightbox-img').src = style.image_url || '';
    document.getElementById('persona-style-lightbox-title').textContent = style.title;
    document.getElementById('persona-style-lightbox-desc').textContent = style.description || '';

    const btn = document.getElementById('persona-style-lightbox-btn');
    if (state.hasPersona) {
        // С персоной — можно выбирать/убирать
        const isSelected = (state.selectedPersonaStyles || []).some(s => s.id === style.id);
        btn.textContent = isSelected ? 'Убрать' : 'Выбрать';
        btn.className = 'persona-style-lightbox-btn' + (isSelected ? ' deselect' : '');
        // Дизейблим «Выбрать» если нет кредитов и стиль ещё не выбран
        if (!isSelected && state.personaCredits <= 0) {
            btn.classList.add('disabled');
        }
        btn.style.display = 'block';
    } else {
        // Без персоны — только просмотр, кнопку скрываем
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
        // Повторный тап — снять выбор
        state.selectedTariff = null;
    } else {
        state.selectedTariff = credits;
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

// Старая функция selectPersonaStyle — теперь не используется, sendData вызывается отдельно
function selectPersonaStyle() {
    const style = state.selectedPersonaStyle;
    if (!style) return;

    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('heavy');

    if (tg) {
        tg.sendData(JSON.stringify({
            action: 'persona_style_selected',
            style_id: style.id,
            style_slug: style.slug,
            style_title: style.title,
        }));
        tg.close();
    }
}
