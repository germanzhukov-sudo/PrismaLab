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

function renderSectionLoading(text) {
    return `<div class="packs-loading">
        <div class="packs-prism"></div>
        <p class="packs-loading-text">${text}</p>
        <div class="packs-loading-dots"><span></span><span></span><span></span></div>
    </div>`;
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
    selectedPersonaStyles: [],
    personaCreditsOriginal: 0,
    photosetsCreditsPreview: 0,
    photosetsFilter: new Set(),
    photosetsLoaded: false,
    expressCatalogLoaded: false,
    photosetsLoadingPromise: null,
    // Tab switcher
    activeMainTab: 'express',
    _lastHiddenAt: 0,
    _scrollMap: {},
    _personaPendingShown: false,
    _packCache: {},            // { [packId]: pack data } — skip fetch on re-open
    _lastPackDetailId: null,   // skip DOM rewrite if same pack
    _lastStyleDetailId: null,  // skip preview rerender if same style
    featuredStyles: [],
    featuredPacks: [],
    featuredCustom: [],
    // General
    hasPersona: false,
    packsUseCredits: false,
    discountBadge: '',
    // Tariffs from API
    tariffs: {},
    selectedTariff: null,  // { mode: 'persona_create'|'persona_topup'|'fast', credits: N }
    _tariffReturnTo: 'main',
    // Profile history
    historyItems: [],
    historyTotal: 0,
    historyOffset: 0,
    historyMode: null,
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
    // Initialize tab switcher
    switchMainTab(state.activeMainTab || 'express', true);
    _initTabSwipe();
    // Keyboard accessibility: arrow keys on tab bar
    document.getElementById('main-tab-bar')?.addEventListener('keydown', (e) => {
        const tabs = Array.from(document.querySelectorAll('#main-tab-bar .tab-btn'));
        const currentIdx = tabs.findIndex(t => t.classList.contains('active'));
        if (e.key === 'ArrowRight' && currentIdx < tabs.length - 1) {
            e.preventDefault();
            tabs[currentIdx + 1].focus();
            switchMainTab(tabs[currentIdx + 1].dataset.tab);
        } else if (e.key === 'ArrowLeft' && currentIdx > 0) {
            e.preventDefault();
            tabs[currentIdx - 1].focus();
            switchMainTab(tabs[currentIdx - 1].dataset.tab);
        }
    });
    // Balance refresh when returning from payment
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            state._lastHiddenAt = Date.now();
        } else if (Date.now() - state._lastHiddenAt > 3000) {
            refreshBalance();
        }
    });
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
                state.tariffs = data.tariffs || {};
                state.discountBadge = data.discount_badge || '';
                state.featuredStyles = data.featured_styles || [];
                state.featuredPacks = data.featured_packs || [];
                state.featuredCustom = data.featured_custom || [];
                renderFeaturedStyles();
                renderFeaturedPacks();
                renderFeaturedCustom();
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

    // Header balance pill
    const hExpress = document.getElementById('header-express-bal');
    if (hExpress) hExpress.textContent = expressBalance;
    const hPhotosets = document.getElementById('header-photosets-bal');
    if (hPhotosets) hPhotosets.textContent = state.photosetsCredits;

    // Main screen
    const mainExpress = document.getElementById('main-express-balance');
    if (mainExpress) mainExpress.textContent = expressBalance;
    const mainPhotosets = document.getElementById('main-photosets-balance');
    if (mainPhotosets) mainPhotosets.textContent = state.photosetsCredits;
    const mainPhotosetsWord = document.getElementById('main-photosets-balance-word');
    if (mainPhotosetsWord) mainPhotosetsWord.textContent = pluralCredits(state.photosetsCredits);
    const mainExpressWord = document.getElementById('main-express-balance-word');
    if (mainExpressWord) mainExpressWord.textContent = pluralCredits(expressBalance);

    // Custom balance (same as express)
    const mainCustom = document.getElementById('main-custom-balance');
    if (mainCustom) mainCustom.textContent = expressBalance;
    const customCredits = document.getElementById('custom-credits-count');
    if (customCredits) customCredits.textContent = expressBalance;
    const mainCustomWord = document.getElementById('main-custom-balance-word');
    if (mainCustomWord) mainCustomWord.textContent = pluralCredits(expressBalance);

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

    // Discount badge
    const discountBadge = document.getElementById('photosets-discount-badge');
    if (discountBadge) {
        if (state.discountBadge && !state.hasPersona) {
            discountBadge.textContent = state.discountBadge;
            discountBadge.style.display = '';
        } else {
            discountBadge.style.display = 'none';
        }
    }

    // Main topup button
    // "Тарифы" button always visible — no conditional hide

    // Sync generate button visibility with current credits state (symmetric: hide AND show)
    const hasCredits = canGenerateExpress();
    const expressGenBtn = document.getElementById('express-generate-btn');
    if (expressGenBtn && state.expressFile) {
        // Express: show only if photo uploaded AND has credits
        expressGenBtn.style.display = hasCredits ? 'block' : 'none';
    }
    const customGenBtn = document.getElementById('custom-generate-btn');
    if (customGenBtn) customGenBtn.style.display = hasCredits ? '' : 'none';
}

// Balance refresh on visibility change (best-effort: fires when TG WebView stays alive)
async function refreshBalance() {
    try {
        const resp = await fetch('/app/api/auth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ init_data: state.initData }),
        });
        if (!resp.ok) return;
        const data = await resp.json();
        // Update ONLY balance — preserve form state (files, prompt, selections)
        state.expressCredits = data.credits || state.expressCredits;
        state.hasPersona = !!data.has_persona;
        state.photosetsCredits = data.persona_credits ?? state.photosetsCredits;
        state.personaCreditsOriginal = state.photosetsCredits;
        // Recalculate preview accounting for already-selected styles
        const selectedCost = (state.selectedPersonaStyles || []).reduce((sum, s) => sum + (s.credit_cost || 4), 0);
        state.photosetsCreditsPreview = Math.max(0, state.photosetsCredits - selectedCost);
        state.tariffs = data.tariffs || state.tariffs;
        updateBalanceDisplays();
        // Re-render footers for current visible screen
        const activeScreen = document.querySelector('.screen.active');
        if (activeScreen) {
            const screenName = activeScreen.id.replace('screen-', '');
            renderScreenFooters(screenName);
        }
    } catch (e) {
        // Silent — best effort
    }
}

// === Featured Styles Carousel ===

const _FEATURED_STYLE_FALLBACK_TITLES = [
    'Вечерний гламур',
    'Свадебный образ',
    'Студийный дым',
    'Клеопатра',
    'Бордовый бархат',
    'Лавандовый шёлк',
    'Кофе в отеле',
    'Драматический свет',
    'Дождливое окно',
];

function _makeShowcaseThumb(imageUrl, alt) {
    const thumb = document.createElement('div');
    thumb.className = 'express-showcase-thumb';
    if (imageUrl) {
        const img = document.createElement('img');
        img.src = imageUrl;
        img.alt = alt || '';
        img.loading = 'lazy';
        thumb.appendChild(img);
    } else {
        const placeholder = document.createElement('div');
        placeholder.className = 'express-showcase-thumb-placeholder';
        thumb.appendChild(placeholder);
    }
    return thumb;
}

function renderFeaturedStyles() {
    const container = document.getElementById('main-express-showcase');
    if (!container) return;

    const byTitle = new Map((state.featuredStyles || []).map((s) => [String(s.title || '').trim(), s]));
    const items = _FEATURED_STYLE_FALLBACK_TITLES.map((title) => {
        const fromApi = byTitle.get(title);
        return {
            title,
            image_url: fromApi ? String(fromApi.image_url || '').trim() : '',
        };
    });

    container.textContent = '';

    items.forEach((style) => {
        const item = document.createElement('div');
        item.className = 'express-showcase-item';
        // Click on item (padding/title) → navigate to section
        item.addEventListener('click', (e) => {
            e.stopPropagation();
            goToExpress();
        });

        const imageUrl = String(style.image_url || '').trim();
        const thumb = _makeShowcaseThumb(imageUrl, style.title);
        if (imageUrl) {
            // Click on thumb → lightbox (stops propagation to item)
            thumb.addEventListener('click', (e) => {
                e.stopPropagation();
                openLightbox(imageUrl);
            });
        }

        const title = document.createElement('div');
        title.className = 'express-showcase-title';
        title.textContent = style.title || '';

        item.appendChild(thumb);
        item.appendChild(title);
        container.appendChild(item);
    });

    const moreItem = document.createElement('div');
    moreItem.className = 'express-showcase-item express-showcase-more';
    moreItem.addEventListener('click', (e) => {
        e.stopPropagation();
        goToExpress();
    });
    const moreThumb = document.createElement('div');
    moreThumb.className = 'express-showcase-thumb';
    moreThumb.textContent = '→';
    const moreTitle = document.createElement('div');
    moreTitle.className = 'express-showcase-title';
    moreTitle.textContent = 'Все стили';
    moreItem.appendChild(moreThumb);
    moreItem.appendChild(moreTitle);
    container.appendChild(moreItem);
}

function renderFeaturedPacks() {
    const container = document.getElementById('main-photosets-showcase');
    if (!container) return;

    const packs = state.featuredPacks || [];
    container.textContent = '';

    packs.forEach((pack) => {
        const item = document.createElement('div');
        item.className = 'express-showcase-item';
        item.addEventListener('click', (e) => {
            e.stopPropagation();
            goToPhotosets();
        });

        const imageUrl = String(pack.image_url || '').trim();
        const thumb = _makeShowcaseThumb(imageUrl, pack.title);
        if (imageUrl) {
            thumb.addEventListener('click', (e) => {
                e.stopPropagation();
                openLightbox(imageUrl);
            });
        }

        const title = document.createElement('div');
        title.className = 'express-showcase-title';
        title.textContent = pack.title || '';

        const meta = document.createElement('div');
        meta.className = 'express-showcase-meta';
        meta.textContent = `${pack.num_images} фото`;

        item.appendChild(thumb);
        item.appendChild(title);
        item.appendChild(meta);
        container.appendChild(item);
    });

    const moreItem = document.createElement('div');
    moreItem.className = 'express-showcase-item express-showcase-more';
    moreItem.addEventListener('click', (e) => {
        e.stopPropagation();
        goToPhotosets();
    });
    const moreThumb = document.createElement('div');
    moreThumb.className = 'express-showcase-thumb';
    moreThumb.textContent = '→';
    const moreTitle = document.createElement('div');
    moreTitle.className = 'express-showcase-title';
    moreTitle.textContent = 'Все фотосеты';
    moreItem.appendChild(moreThumb);
    moreItem.appendChild(moreTitle);
    container.appendChild(moreItem);
}

function renderFeaturedCustom() {
    const container = document.getElementById('main-custom-showcase');
    if (!container) return;

    const items = state.featuredCustom || [];
    container.textContent = '';

    items.forEach((item) => {
        const el = document.createElement('div');
        el.className = 'express-showcase-item';
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            goToCustomPrompt();
        });

        const imageUrl = String(item.image_url || '').trim();
        const thumb = _makeShowcaseThumb(imageUrl, item.title);
        if (imageUrl) {
            thumb.addEventListener('click', (e) => {
                e.stopPropagation();
                openLightbox(imageUrl);
            });
        }

        const title = document.createElement('div');
        title.className = 'express-showcase-title';
        title.textContent = item.title || '';

        el.appendChild(thumb);
        el.appendChild(title);
        container.appendChild(el);
    });

    const moreItem = document.createElement('div');
    moreItem.className = 'express-showcase-item express-showcase-more';
    moreItem.addEventListener('click', (e) => {
        e.stopPropagation();
        goToCustomPrompt();
    });
    const moreThumb = document.createElement('div');
    moreThumb.className = 'express-showcase-thumb';
    moreThumb.textContent = '→';
    const moreTitle = document.createElement('div');
    moreTitle.className = 'express-showcase-title';
    moreTitle.textContent = 'и бесконечно много что ещё';
    moreItem.appendChild(moreThumb);
    moreItem.appendChild(moreTitle);
    container.appendChild(moreItem);
}

// === Tab Switcher ===

const _tabAccentColors = {
    express: 'linear-gradient(135deg, #FF6B35, #dc2626)',
    custom: 'linear-gradient(135deg, #00B4D8, #3b82f6)',
    photosets: 'linear-gradient(135deg, #9B5DE5, #ec4899)',
};

function switchMainTab(tab, force) {
    if (!force && state.activeMainTab === tab) return;
    state.activeMainTab = tab;
    if (tg?.HapticFeedback) tg.HapticFeedback.selectionChanged();

    // Update tab buttons
    document.querySelectorAll('#main-tab-bar .tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    // Slide to card
    const tabs = Array.from(document.querySelectorAll('#tab-cards-track .tab-card'));
    const idx = tabs.findIndex(c => c.dataset.tab === tab);
    const track = document.getElementById('tab-cards-track');
    if (track && idx >= 0) {
        track.style.transform = `translate3d(-${idx * 92}%, 0, 0)`;
    }

    // Update ambient glow
    const glow = document.getElementById('ambient-glow');
    if (glow && _tabAccentColors[tab]) {
        glow.style.background = _tabAccentColors[tab];
    }
}

function handleCardClick(card, navigateFn) {
    const tab = card.dataset.tab;
    if (tab !== state.activeMainTab) {
        // Inactive card — just switch to it
        switchMainTab(tab);
    } else {
        // Active card — navigate into it
        navigateFn();
    }
}

function _initTabSwipe() {
    const area = document.getElementById('tab-content-area');
    const track = document.getElementById('tab-cards-track');
    if (!area || !track) return;

    let startX = 0, startY = 0, currentX = 0;
    let isDragging = false, directionLocked = false, isHorizontal = false;
    let startRail = null; // showcase rail touched on touchstart (if any)
    let railCanGoLeft = false, railCanGoRight = false;
    const THRESHOLD = 40;
    const CARD_WIDTH_PCT = 92;

    function _getBaseOffset() {
        const tabs = Array.from(track.querySelectorAll('.tab-card'));
        const idx = tabs.findIndex(c => c.dataset.tab === state.activeMainTab);
        return idx >= 0 ? -idx * CARD_WIDTH_PCT : 0;
    }

    area.addEventListener('touchstart', (e) => {
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
        currentX = startX;
        isDragging = true;
        directionLocked = false;
        isHorizontal = false;
        track.style.transition = 'none';

        // Check if touch started inside a showcase rail
        startRail = e.target.closest('.express-showcase-rail');
        if (startRail) {
            // Remember rail's scroll state — used to decide if card swipe takes over
            railCanGoLeft = startRail.scrollLeft > 0;
            railCanGoRight = startRail.scrollLeft < (startRail.scrollWidth - startRail.clientWidth - 1);
        } else {
            railCanGoLeft = railCanGoRight = false;
        }
    }, { passive: true });

    area.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        currentX = e.touches[0].clientX;
        const dx = currentX - startX;
        const dy = e.touches[0].clientY - startY;

        // Lock direction — bias towards horizontal (6px horizontal vs 10px vertical)
        if (!directionLocked && (Math.abs(dx) > 6 || Math.abs(dy) > 10)) {
            directionLocked = true;
            isHorizontal = Math.abs(dx) >= Math.abs(dy);

            // If inside a showcase rail and rail CAN scroll in the swipe direction,
            // defer to rail (don't swipe cards). Otherwise → card swipe takes over.
            if (isHorizontal && startRail) {
                const swipeLeft = dx < 0;  // finger moved left → rail scrolls right
                const swipeRight = dx > 0; // finger moved right → rail scrolls left
                if ((swipeLeft && railCanGoRight) || (swipeRight && railCanGoLeft)) {
                    // Rail handles it — cancel card swipe
                    isDragging = false;
                    return;
                }
            }
        }
        if (!isHorizontal || !isDragging) return;

        // Block iOS horizontal rubber-band / elastic scroll → prevents header shake
        if (e.cancelable) e.preventDefault();

        // Drag the track
        const basePct = _getBaseOffset();
        const dragPx = dx;
        const areaWidth = area.offsetWidth;
        const dragPct = (dragPx / areaWidth) * 100;
        track.style.transform = `translate3d(${basePct + dragPct}%, 0, 0)`;
    }, { passive: false });

    area.addEventListener('touchend', () => {
        if (!isDragging) return;
        isDragging = false;
        track.style.transition = '';

        if (!isHorizontal) {
            // Snap back
            track.style.transform = `translate3d(${_getBaseOffset()}%, 0, 0)`;
            return;
        }

        const dx = currentX - startX;
        const tabs = Array.from(track.querySelectorAll('.tab-card'));
        const currentIdx = tabs.findIndex(c => c.dataset.tab === state.activeMainTab);

        if (dx < -THRESHOLD && currentIdx < tabs.length - 1) {
            switchMainTab(tabs[currentIdx + 1].dataset.tab);
        } else if (dx > THRESHOLD && currentIdx > 0) {
            switchMainTab(tabs[currentIdx - 1].dataset.tab);
        } else {
            // Snap back to current
            track.style.transform = `translate3d(${_getBaseOffset()}%, 0, 0)`;
        }
    }, { passive: true });

    // Desktop trackpad/mouse wheel — swipe cards horizontally
    let wheelAccumX = 0;
    let wheelTimeout = null;
    area.addEventListener('wheel', (e) => {
        // Guards
        if (e.ctrlKey) return; // pinch-zoom
        if (!document.getElementById('screen-main')?.classList.contains('active')) return;
        if (e.target.closest('.express-showcase-rail')) return; // let rail scroll
        if (Math.abs(e.deltaX) <= Math.abs(e.deltaY)) return; // vertical scroll

        e.preventDefault();
        wheelAccumX += e.deltaX;
        clearTimeout(wheelTimeout);
        wheelTimeout = setTimeout(() => {
            const tabs = Array.from(track.querySelectorAll('.tab-card'));
            const currentIdx = tabs.findIndex(c => c.dataset.tab === state.activeMainTab);
            if (wheelAccumX > THRESHOLD && currentIdx < tabs.length - 1) {
                switchMainTab(tabs[currentIdx + 1].dataset.tab);
            } else if (wheelAccumX < -THRESHOLD && currentIdx > 0) {
                switchMainTab(tabs[currentIdx - 1].dataset.tab);
            }
            wheelAccumX = 0;
        }, 100);
    }, { passive: false });
}

// === Navigation ===

// Classic display: none/flex + window scroll navigation.
// state._scrollMap[screenId] = window.scrollY перед сменой экрана.
// При возврате — force reflow → window.scrollTo saved. Мягкий fade-in 0.2s в CSS.

function saveScroll(_screenName) { /* legacy no-op — scroll saved inside showScreen */ }

function showScreen(name) {
    const target = document.getElementById(`screen-${name}`);
    if (!target) return;
    if (target.classList.contains('active')) return;

    // Save scroll of current screen before hiding it.
    const currentActive = document.querySelector('.screen.active');
    if (currentActive) {
        state._scrollMap[currentActive.id] = window.scrollY || 0;
        currentActive.classList.remove('active');
    }

    // Footers up-front — до activation, чтобы layout нового screen учёл их сразу.
    renderScreenFooters(name);

    // Activate new screen.
    target.classList.add('active');

    // Force reflow — браузер применяет display:flex и вычисляет layout синхронно.
    // eslint-disable-next-line no-unused-expressions
    void target.offsetHeight;

    // Restore scroll (saved) или reset to top для новых открытий.
    const savedY = state._scrollMap[target.id];
    window.scrollTo(0, savedY != null ? savedY : 0);

    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
}

// Перед открытием detail-экрана — удалить его запомненный scroll, чтобы открылся с верха.
function resetScreenScroll(name) {
    delete state._scrollMap[`screen-${name}`];
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

function openExpressInfo(e) {
    if (e) e.stopPropagation();
    trackEvent('v2_express_info_open');
    showScreen('express-info');
}

function openCustomInfo(e) {
    if (e) e.stopPropagation();
    trackEvent('v2_custom_info_open');
    showScreen('custom-info');
}

function openPhotosetsInfo(e) {
    if (e) e.stopPropagation();
    trackEvent('v2_photosets_info_open');
    showScreen('photosets-info');
}

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
    grid.innerHTML = renderSectionLoading('Нужно немного времени для загрузки, пожалуйста, никуда не убегайте');

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
    grid.innerHTML = renderSectionLoading('Нужно немного времени для загрузки, пожалуйста, никуда не убегайте');

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
                : `<div class="style-card-bg" style="background:${gradients[i % gradients.length]}"></div>`
            }
            <div class="style-card-info">
                <div class="style-card-name">${style.label}</div>
            </div>
        </div>`;
    }).join('');
}

function selectExpressStyle(style) {
    state.selectedExpressStyle = style;
    trackEvent('v2_express_style_select', { style_id: style.id });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');

    document.getElementById('express-selected-style').textContent = style.label;
    // Hide provider choice in V2
    const providerChoice = document.getElementById('provider-choice');
    if (providerChoice) providerChoice.style.display = 'none';

    // Restore preview if photo already uploaded, else fresh upload
    if (state.expressFile) {
        document.getElementById('express-upload-zone').style.display = 'none';
        document.getElementById('express-preview').style.display = 'block';
        document.getElementById('express-generate-btn').style.display = canGenerateExpress() ? 'block' : 'none';
        const previewImg = document.getElementById('express-preview-image');
        if (previewImg && !previewImg.src.startsWith('data:')) {
            const reader = new FileReader();
            reader.onload = (e) => { previewImg.src = e.target.result; };
            reader.readAsDataURL(state.expressFile);
        }
    } else {
        resetExpressUpload();
    }
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
        document.getElementById('express-generate-btn').style.display = canGenerateExpress() ? 'block' : 'none';
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
    // Progress animation removed — generating screen now shows static exit hint

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
            // stopProgressAnimation removed — no progress bar on generating screen
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
            // stopProgressAnimation removed — no progress bar on generating screen
            trackEvent('v2_express_generate_done', { style_id: state.selectedExpressStyle?.id });

            if (!state.expressCredits.free_used) {
                state.expressCredits.free_used = true;
            } else {
                state.expressCredits.fast = Math.max(0, state.expressCredits.fast - 1);
            }
            updateBalanceDisplays();

            document.getElementById('express-result-image').src = data.image_url || data.result_url;
            const note = document.getElementById('express-result-note');
            if (note) {
                note.textContent = data.tg_sent
                    ? 'Фото отправлено в Telegram и сохранено в профиле.'
                    : 'Фото сохранено в профиле. В Telegram отправить не удалось.';
            }
            showScreen('express-result');
            if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
        } else if (data.status === 'error') {
            // stopProgressAnimation removed — no progress bar on generating screen
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

async function pollCustomStatus() {
    if (!state.customTaskId) return;
    try {
        const resp = await fetch(`/app/api/status/${state.customTaskId}`, {
            headers: { 'X-Telegram-Init-Data': state.initData },
        });
        const data = await resp.json();

        if (data.status === 'done') {
            // stopProgressAnimation removed — no progress bar on generating screen
            trackEvent('v2_custom_generate_done');

            if (!state.expressCredits.free_used) {
                state.expressCredits.free_used = true;
            } else {
                state.expressCredits.fast = Math.max(0, state.expressCredits.fast - 1);
            }
            updateBalanceDisplays();

            document.getElementById('express-result-image').src = data.image_url || data.result_url;
            const note = document.getElementById('express-result-note');
            if (note) {
                note.textContent = data.tg_sent
                    ? 'Фото отправлено в Telegram и сохранено в профиле.'
                    : 'Фото сохранено в профиле. В Telegram отправить не удалось.';
            }
            showScreen('express-result');
            if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
        } else if (data.status === 'error') {
            // stopProgressAnimation removed — no progress bar on generating screen
            alert('Ошибка генерации. Попробуйте ещё раз.');
            showScreen('custom-prompt');
            if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('error');
        } else {
            setTimeout(pollCustomStatus, 2000);
        }
    } catch (e) {
        console.error('Poll custom error:', e);
        setTimeout(pollCustomStatus, 3000);
    }
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
        grid.innerHTML = renderSectionLoading('Нужно немного времени для загрузки, пожалуйста, никуда не убегайте');
    }

    try {
        const limit = 18;
        const modeParam = state.historyMode ? `&mode=${state.historyMode}` : '';
        const url = `/app/api/v3/history?limit=${limit}&offset=${state.historyOffset}${modeParam}`;
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

        const badge = document.createElement('div');
        badge.className = 'profile-history-badge';
        const modeLabels = {
            express: `Экспресс: ${item.style_title || ''}`,
            custom: 'Своя идея',
            photoset: `Фотосет: ${item.style_title || ''}`,
        };
        badge.textContent = modeLabels[item.mode] || item.mode || 'Экспресс';
        wrapper.appendChild(badge);
        grid.appendChild(wrapper);
    });
    if (!grid.children.length) {
        grid.innerHTML = '<div class="empty-state">История пока пустая</div>';
    }
}

function filterHistory(mode) {
    state.historyMode = mode;
    if (tg?.HapticFeedback) tg.HapticFeedback.selectionChanged();

    // Update tab active state
    document.querySelectorAll('#history-tabs .history-tab').forEach(btn => {
        const btnMode = btn.dataset.mode || null;  // "" → null
        btn.classList.toggle('active', btnMode === mode);
    });

    loadProfileHistory(true);
}

// === EXPRESS V3 FLOW ===

async function loadExpressCatalog(keepFilters) {
    // Кеш: если каталог уже загружен и не смена фильтров — показываем из DOM.
    // Обновляем баланс из актуального state (локальные списания после генерации).
    if (!keepFilters && state.expressCatalogLoaded) {
        showScreen('express-catalog');
        updateBalanceDisplays();
        const cachedBal = state.expressCredits.free_used ? state.expressCredits.fast : (state.expressCredits.fast + 1);
        const credEl = document.getElementById('v3-credits-count');
        if (credEl) credEl.textContent = cachedBal;
        renderScreenFooters('express-catalog');
        return;
    }
    if (!keepFilters) showScreen('express-catalog');
    const grid = document.getElementById('v3-styles-grid');
    if (!keepFilters) {
        state.v3SelectedCategory = 'all';
        state.v3SelectedTags = [];
    }
    grid.innerHTML = renderSectionLoading('Нужно немного времени для загрузки, пожалуйста, никуда не убегайте');

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
        renderScreenFooters('express-catalog');
        state.expressCatalogLoaded = true;
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
        card.addEventListener('click', () => {
            if (canGenerateExpress()) {
                selectV3Style(s);
            } else {
                openStylePreviewLightbox(s);
            }
        });

        const hasImg = s.image_url && s.image_url.startsWith('http');
        if (hasImg) {
            card.style.backgroundImage = `url(${CSS.escape(s.image_url)})`;
            card.style.backgroundSize = 'cover';
            card.style.backgroundPosition = 'center';
        } else {
            card.style.background = GRADIENTS[i % GRADIENTS.length];
        }

        const info = document.createElement('div');
        info.className = 'style-card-info';
        const label = document.createElement('span');
        label.className = 'style-label';
        label.textContent = s.label;
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
    // Save catalog scroll position for restore on back (Task 6b)
    resetScreenScroll('express-upload'); // детали стиля — открываем с верха
    state.selectedExpressStyle = style;
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
    document.getElementById('express-selected-style').textContent = style.label;

    // Show selected style preview image
    const stylePreview = document.getElementById('upload-style-preview');
    const stylePreviewImg = document.getElementById('upload-style-preview-img');
    if (stylePreview && stylePreviewImg) {
        const previewUrl = style.image_url || (style.preview_urls && style.preview_urls[0]) || '';
        if (previewUrl) {
            stylePreviewImg.src = previewUrl;
            stylePreview.style.display = '';
        } else {
            stylePreview.style.display = 'none';
        }
    }

    // Restore preview if photo already uploaded
    if (state.expressFile) {
        document.getElementById('express-upload-zone').style.display = 'none';
        document.getElementById('express-preview').style.display = 'block';
        document.getElementById('express-generate-btn').style.display = canGenerateExpress() ? 'block' : 'none';
        const previewImg = document.getElementById('express-preview-image');
        if (previewImg && !previewImg.src.startsWith('data:')) {
            const reader = new FileReader();
            reader.onload = (e) => { previewImg.src = e.target.result; };
            reader.readAsDataURL(state.expressFile);
        }
    } else {
        document.getElementById('express-upload-zone').style.display = '';
        document.getElementById('express-preview').style.display = 'none';
        document.getElementById('express-generate-btn').style.display = 'none';
    }

    // Show provider choice in V3
    const providerChoice = document.getElementById('provider-choice');
    if (providerChoice) {
        providerChoice.style.display = '';
        updateProviderUI();
        _updateGenerateLabel('express');
    }
    showScreen('express-upload');
}

// Credits check — centralized
function canGenerateExpress() {
    return state.expressCredits.fast > 0 || !state.expressCredits.free_used;
}

// Provider choice — centralized helpers
function _providerLabel(p) {
    return p === 'nano-banana-pro' ? 'Nano Banana Pro' : 'Seedream';
}

function _updateGenerateLabel(screen) {
    // "Создать" по центру, provider chip справа (через absolute positioning в CSS).
    // classList на .btn-text, inner layout — span для текста + span для chip.
    const render = (provider) =>
        `Создать<span class="btn-provider-hint">${provider}</span>`;
    if (screen === 'express') {
        const btn = document.querySelector('#express-generate-btn .btn-text');
        if (btn) {
            btn.classList.add('btn-text--with-provider');
            btn.innerHTML = render(_providerLabel(state.selectedProvider));
        }
    } else if (screen === 'custom') {
        const btn = document.querySelector('#custom-generate-btn .btn-text');
        if (btn) {
            btn.classList.add('btn-text--with-provider');
            btn.innerHTML = render(_providerLabel(state.customProvider));
        }
    }
}

function selectProvider(provider) {
    state.selectedProvider = provider;
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
    updateProviderUI();
    _updateGenerateLabel('express');
}

function updateProviderUI() {
    // Scoped to Express provider options only
    document.querySelectorAll('#express-provider-options .provider-option').forEach(el => {
        el.classList.toggle('selected', el.dataset.provider === state.selectedProvider);
    });
}

// === PHOTOSETS FLOW ===

async function loadPhotosets(forceReload) {
    // Task 4: paid persona_create, but no persona uploaded yet → modal
    if (!state._personaPendingShown &&
        !state.hasPersona &&
        (state.personaCreditsOriginal || state.photosetsCredits || 0) > 0) {
        openPersonaPendingModal();
        return;   // не грузим экран, юзер уйдёт в бота по кнопке "Понятно"
    }
    showScreen('photosets');
    // Hide "Докупить +" immediately — state.hasPersona is known from auth
    const topupBtn = document.getElementById('photosets-topup-btn');
    if (topupBtn) topupBtn.style.display = state.hasPersona ? '' : 'none';

    // Return instantly with cached catalog — no loader, no DOM rewrite on re-entry.
    // Grid уже rendered at first load. innerHTML rewrite вызывает visual flicker →
    // не перерисовываем заново, просто показываем экран с уже имеющимся content.
    if (!forceReload && state.photosetsLoaded) {
        updatePhotosetsPreviewBalance();
        ensurePhotosetsFooter();
        return;
    }

    // Prevent duplicate requests on repeated taps.
    if (state.photosetsLoadingPromise) {
        await state.photosetsLoadingPromise;
        return;
    }

    const grid = document.getElementById('photosets-grid');
    grid.innerHTML = renderSectionLoading('Нужно немного времени, чтобы загрузить фотосеты. Пожалуйста, никуда не убегайте');

    state.photosetsLoadingPromise = (async () => {
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

            state.personaCreditsOriginal = state.photosetsCredits;
            state.photosetsCreditsPreview = state.photosetsCredits;
            state.selectedPersonaStyles = [];
            state.photosetsLoaded = true;
            renderPhotosets();
            ensurePhotosetsFooter();
            updatePhotosetsPreviewBalance();
        } catch (e) {
            console.error('Load photosets error:', e);
            grid.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-secondary)">Ошибка загрузки</div>';
        } finally {
            state.photosetsLoadingPromise = null;
        }
    })();

    await state.photosetsLoadingPromise;
}

function renderPhotosetCardPreview(item) {
    const previews = (item.preview_urls || []).filter(Boolean);
    if (previews.length > 0) {
        return `<img class="photoset-card-cover" src="${previews[0]}" alt="" loading="lazy">`;
    }
    return `<div class="photoset-card-cover-placeholder">📸</div>`;
}

function filterPhotosets(filter) {
    if (tg?.HapticFeedback) tg.HapticFeedback.selectionChanged();
    if (filter === 'all') {
        state.photosetsFilter.clear();
    } else {
        if (state.photosetsFilter.has(filter)) {
            state.photosetsFilter.delete(filter);
        } else {
            state.photosetsFilter.add(filter);
        }
    }
    document.querySelectorAll('#photosets-filter-tabs .category-tab').forEach(btn => {
        const f = btn.dataset.filter;
        if (f === 'all') {
            btn.classList.toggle('active', state.photosetsFilter.size === 0);
        } else {
            btn.classList.toggle('active', state.photosetsFilter.has(f));
        }
    });
    renderPhotosets();
}

function renderPhotosets() {
    const grid = document.getElementById('photosets-grid');
    let items = state.photosets || [];

    // Client-side multi-filter
    const filters = state.photosetsFilter;
    if (filters.size > 0) {
        items = items.filter(item => {
            if (filters.has('available') && item.is_locked) return false;
            const categoryFilters = new Set([...filters].filter(f => f !== 'available'));
            if (categoryFilters.size > 0) return categoryFilters.has(item.category);
            return true;
        });
    }

    if (!items.length) {
        if (filters.has('available')) {
            grid.innerHTML = `<div class="photosets-empty-state" style="grid-column:1/-1">
                <p>У вас не хватает кредитов для фотосета. Пополните баланс</p>
                <div class="persona-buy-footer-buttons" id="photosets-empty-topup-buttons"></div>
                <button class="persona-buy-pay-btn" id="photosets-empty-topup-pay-btn" style="display:none" onclick="buyTariffPage()">Оплатить</button>
            </div>`;
            _renderTariffButtonsInto('photosets-empty-topup-buttons', 'photosets-empty-topup-pay-btn',
                state.hasPersona ? 'persona_topup' : 'persona_create',
                state.hasPersona ? (state.tariffs.persona_topup || []) : (state.tariffs.persona_create || []));
        } else {
            grid.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-secondary);grid-column:1/-1">Фотосеты скоро появятся</div>';
        }
        return;
    }

    const lockSvg = '<svg viewBox="0 0 24 24" fill="none"><rect x="5" y="11" width="14" height="10" rx="2" fill="#fff"/><path d="M8 11V7a4 4 0 018 0v4" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>';
    const showCheckboxes = state.hasPersona && state.personaCreditsOriginal > 0;

    grid.innerHTML = items.map((item) => {
        const creditCost = Number(item.credit_cost || 0);
        const numImages = Number(item.num_images || 0);
        const isLocked = !!item.is_locked;
        const openAction = item.type === 'pack' ? `openPackDetail(${item.entity_id})` : `openStyleDetail(${item.entity_id})`;

        // Badge: checkbox on unlocked styles, lock on locked items
        let badgeHtml = '';
        if (item.type === 'style' && showCheckboxes && !isLocked) {
            const isSelected = state.selectedPersonaStyles.some(s => s.id === item.entity_id);
            const isDisabled = !isSelected && state.photosetsCreditsPreview < creditCost;
            badgeHtml = `<div class="photoset-card-checkbox ${isSelected ? 'checked' : ''} ${isDisabled ? 'disabled' : ''}"
                onclick="event.stopPropagation(); togglePhotosetStyle(${item.entity_id})">${isSelected ? '✓' : ''}</div>`;
        } else if (isLocked) {
            badgeHtml = `<div class="photoset-card-lock-badge">${lockSvg}</div>`;
        }

        return `
            <div class="photoset-card" onclick="${openAction}">
                ${badgeHtml}
                ${renderPhotosetCardPreview(item)}
                <div class="photoset-card-info">
                    <div class="photoset-card-title">${item.title || ''}</div>
                    <div class="photoset-card-meta">${numImages} фото</div>
                    <div class="photoset-card-cost">
                        <span class="badge-prism"></span> ${creditCost} ${pluralCredits(creditCost)}
                    </div>
                </div>
            </div>`;
    }).join('');
}

// Pack Detail — with cache: повторное открытие того же пака instant (no fetch, no loading).
async function openPackDetail(packId) {
    trackEvent('v2_photoset_detail', { kind: 'pack', id: packId });
    document.querySelectorAll('.detail-credits-count').forEach(el => {
        el.textContent = state.photosetsCreditsPreview;
    });

    // Cache hit: данные пака уже загружены раньше → instant show.
    const cached = state._packCache[packId];
    if (cached) {
        // Если DOM уже содержит этот пак (последний показанный) — не трогаем DOM вообще.
        if (state._lastPackDetailId !== packId) {
            _fillPackDetailDOM(cached);
            state._lastPackDetailId = packId;
        }
        state.selectedPack = cached;
        resetScreenScroll('photoset-pack-detail');  // всегда открываем с верха
        showScreen('photoset-pack-detail');
        if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
        return;
    }

    // Cache miss: очищаем UI и показываем loading placeholder до fetch.
    state.selectedPack = null;
    state._lastPackDetailId = null;
    const titleEl = document.getElementById('photoset-pack-title');
    if (titleEl) titleEl.textContent = 'Загрузка…';
    const galleryClear = document.getElementById('photoset-pack-gallery');
    if (galleryClear) galleryClear.replaceChildren();
    const packCostPillClear = document.getElementById('photoset-pack-cost');
    if (packCostPillClear) packCostPillClear.style.display = 'none';
    const buyBtnInit = document.getElementById('photoset-pack-buy-btn');
    if (buyBtnInit) buyBtnInit.style.display = 'none';
    const noPersInit = document.getElementById('pack-no-persona');
    if (noPersInit) noPersInit.style.display = 'none';
    const topupInit = document.getElementById('pack-topup');
    if (topupInit) topupInit.style.display = 'none';

    resetScreenScroll('photoset-pack-detail');
    showScreen('photoset-pack-detail');
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');

    try {
        const resp = await fetch(`/app/api/packs/${packId}`, {
            headers: { 'X-Telegram-Init-Data': state.initData },
        });
        const pack = await resp.json();
        state.selectedPack = pack;
        state._packCache[packId] = pack;
        state._lastPackDetailId = packId;
        _fillPackDetailDOM(pack);
    } catch (e) {
        console.error('Pack detail error:', e);
    }
}

// Fill DOM фотосет-пака из объекта pack. Используется в openPackDetail
// как на cache hit (без fetch), так и после успешного fetch.
function _fillPackDetailDOM(pack) {
    document.getElementById('photoset-pack-title').textContent = pack.title;

    // Credit cost pill (унифицировано с деталью стиля)
    const packCostPill = document.getElementById('photoset-pack-cost');
    const packCostValue = document.getElementById('photoset-pack-cost-value');
    const packCostWord = document.getElementById('photoset-pack-cost-word');
    const packCostImages = document.getElementById('photoset-pack-cost-images');
    const pcc = pack.credit_cost || pack.expected_images || 4;
    const pimg = pack.expected_images || 4;
    if (packCostPill && packCostValue && packCostWord && packCostImages) {
        packCostValue.textContent = String(pcc);
        packCostWord.textContent = pluralCredits(pcc);
        packCostImages.textContent = `${pimg} фото`;
        packCostPill.style.display = '';
    }

    if (state.packsUseCredits) {
        const cc = pack.credit_cost || pack.expected_images;
        document.getElementById('photoset-pack-buy-text').textContent = `Купить за ${cc} ${pluralCredits(cc)}`;
    } else {
        document.getElementById('photoset-pack-buy-text').textContent = `Купить ${pack.price_rub} ₽`;
    }

    // Галерея — через безопасный DOM API вместо innerHTML-интерполяции (XSS hardening)
    const gallery = document.getElementById('photoset-pack-gallery');
    gallery.replaceChildren();
    (pack.examples || []).slice(0, 10).forEach(url => {
        const img = document.createElement('img');
        img.src = url;
        img.alt = '';
        img.loading = 'lazy';
        img.addEventListener('click', () => openLightbox(url));
        gallery.appendChild(img);
    });

    // Locked pack gating
    const buyBtn = document.getElementById('photoset-pack-buy-btn');
    const packNoPersona = document.getElementById('pack-no-persona');
    const packTopup = document.getElementById('pack-topup');
    buyBtn.style.display = '';
    if (packNoPersona) packNoPersona.style.display = 'none';
    if (packTopup) packTopup.style.display = 'none';

    if (!state.hasPersona) {
        buyBtn.style.display = 'none';
        if (packNoPersona) {
            packNoPersona.style.display = '';
            _renderTariffButtonsInto('pack-create-buttons', 'pack-create-pay-btn', 'persona_create', state.tariffs.persona_create || []);
        }
    } else if (state.packsUseCredits) {
        const cc = pack.credit_cost || pack.expected_images;
        if (state.photosetsCredits < cc) {
            buyBtn.style.display = 'none';
            if (packTopup) {
                packTopup.style.display = '';
                const packDeficit = cc - state.photosetsCredits;
                document.getElementById('pack-topup-hint').textContent = `Для этого фотосета не хватает ${packDeficit} ${pluralCredits(packDeficit)}. Пополните баланс`;
                _renderTariffButtonsInto('pack-topup-buttons', 'pack-topup-pay-btn', 'persona_topup', state.tariffs.persona_topup || []);
            }
        }
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

    // Тот же стиль, что и в прошлый раз — не трогаем title/desc/preview DOM вообще.
    // Только gates (кнопки, тарифы, баланс) могут поменяться, их обновим ниже.
    const sameStyle = state._lastStyleDetailId === styleId;
    if (!sameStyle) {
        document.getElementById('photoset-style-title').textContent = style.title;
        document.getElementById('photoset-style-desc').textContent = style.description || '';
        document.getElementById('photoset-style-cost-value').textContent = style.credit_cost;
        document.getElementById('photoset-style-cost-word').textContent = pluralCredits(style.credit_cost);

        // Preview rail — rerender только для нового стиля.
        const preview = document.getElementById('photoset-style-preview');
        const previews = (style.preview_urls || []).filter(Boolean).slice(0, 4);
        preview.replaceChildren();
        previews.forEach(url => {
            const img = document.createElement('img');
            img.src = url;
            img.alt = '';
            img.loading = 'lazy';
            img.addEventListener('click', () => openLightbox(url));
            preview.appendChild(img);
        });
        for (let i = previews.length; i < 4; i++) {
            const placeholder = document.createElement('div');
            placeholder.className = 'preview-placeholder';
            placeholder.textContent = '?';
            preview.appendChild(placeholder);
        }
        state._lastStyleDetailId = styleId;
    }

    // Header badge (баланс может быть актуальный) — обновляется всегда.
    document.querySelectorAll('.detail-credits-count').forEach(el => {
        el.textContent = state.photosetsCreditsPreview;
    });

    const creditCost = style.credit_cost;

    // Gate: generate / no-persona / not-enough-credits
    const generateBtn = document.getElementById('photoset-generate-btn');
    const noPersonaBlock = document.getElementById('photoset-no-persona');
    const topupBlock = document.getElementById('photoset-style-topup');
    generateBtn.style.display = 'none';
    noPersonaBlock.style.display = 'none';
    topupBlock.style.display = 'none';

    const inCheckboxMode = state.hasPersona && state.personaCreditsOriginal > 0;

    if (!state.hasPersona) {
        noPersonaBlock.style.display = '';
        _renderTariffButtonsInto('style-create-buttons', 'style-create-pay-btn', 'persona_create', state.tariffs.persona_create || []);
    } else if (inCheckboxMode) {
        // Task 6a: в детали стиля — прямая генерация одного стиля (не multi-select toggle).
        // Multi-select остаётся только на карточках списка (чекбоксы), не в детали.
        const canAfford = state.photosetsCredits >= creditCost;
        if (canAfford) {
            generateBtn.style.display = '';
            const numImg = state.selectedPersonaStyle?.num_images || 4;
            generateBtn.querySelector('.btn-text').textContent = `\u2728 Сгенерировать (${numImg} фото)`;
            generateBtn.onclick = () => startStyleBatchGeneration();
        } else {
            // Can't afford with preview balance — show topup
            topupBlock.style.display = '';
            const topupHint = document.getElementById('photoset-style-topup')?.querySelector('.topup-hint');
            if (topupHint) {
                const deficit = creditCost - state.photosetsCreditsPreview;
                topupHint.textContent = `Для этого фотосета не хватает ${deficit} ${pluralCredits(deficit)}. Пополните баланс`;
            }
            _renderTariffButtonsInto('style-topup-buttons', 'style-topup-pay-btn', 'persona_topup', state.tariffs.persona_topup || []);
        }
    } else {
        // Has persona but 0 credits original — show topup
        topupBlock.style.display = '';
        const topupHint = document.getElementById('photoset-style-topup')?.querySelector('.topup-hint');
        if (topupHint) topupHint.textContent = `Для этого фотосета не хватает ${creditCost} ${pluralCredits(creditCost)}. Пополните баланс`;
        _renderTariffButtonsInto('style-topup-buttons', 'style-topup-pay-btn', 'persona_topup', state.tariffs.persona_topup || []);
    }

    resetScreenScroll('photoset-style-detail');
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

        // Task 5c: API теперь списывает кредиты синхронно → читаем серверный баланс
        if (data.credits_balance != null) {
            state.photosetsCredits = data.credits_balance;
            state.photosetsCreditsPreview = data.credits_balance;
            if (typeof updateBalanceDisplays === 'function') updateBalanceDisplays();
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
    closeMiniApp();
}

// === Photoset Multi-Select (Checkboxes) ===

function togglePhotosetStyle(entityId) {
    const idx = state.selectedPersonaStyles.findIndex(s => s.id === entityId);
    if (idx >= 0) {
        const style = state.selectedPersonaStyles[idx];
        state.selectedPersonaStyles.splice(idx, 1);
        state.photosetsCreditsPreview += (style.credit_cost || 1);
        if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('light');
    } else {
        const style = state.personaStyles.find(s => s.id === entityId);
        if (!style) return;
        const cost = style.credit_cost || 1;
        if (state.photosetsCreditsPreview < cost) {
            if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('error');
            return;
        }
        state.selectedPersonaStyles.push(style);
        state.photosetsCreditsPreview -= cost;
        trackEvent('v2_photoset_style_select', { style_id: entityId });
        if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('medium');
    }
    updatePhotosetsPreviewBalance();
    ensurePhotosetsFooter();
    renderPhotosets();
}

function updatePhotosetsPreviewBalance() {
    // ONLY update photosets catalog header — not main screen or profile
    const pCredits = document.getElementById('photosets-credits-count');
    if (pCredits) pCredits.textContent = state.photosetsCreditsPreview;
    // Hide "Докупить +" when no persona — footer handles that state
    const topupBtn = document.getElementById('photosets-topup-btn');
    if (topupBtn) topupBtn.style.display = state.hasPersona ? '' : 'none';
}

/**
 * Single source of truth for all photosets footers.
 * Declarative: sets correct state based on current conditions.
 * Called after any photosets state change (load, toggle style, navigate).
 */
function ensurePhotosetsFooter() {
    const createFooter = document.getElementById('photosets-create-footer');
    const topupFooter = document.getElementById('photosets-topup-footer');
    const generateFooter = document.getElementById('photosets-generate-footer');
    const grid = document.getElementById('photosets-grid');

    // Default: hide all photosets footers
    if (createFooter) createFooter.style.display = 'none';
    if (topupFooter) topupFooter.style.display = 'none';
    if (generateFooter) generateFooter.style.display = 'none';
    if (grid) grid.classList.remove('has-footer');

    const selected = state.selectedPersonaStyles || [];

    // Priority 1: Generate button (persona + selected styles)
    if (state.hasPersona && selected.length > 0) {
        if (generateFooter) {
            generateFooter.style.display = 'flex';
            const btnText = document.getElementById('photosets-generate-btn-text');
            if (btnText) btnText.textContent = `Сгенерировать (${selected.length})`;
        }
        if (grid) grid.classList.add('has-footer');
        return;
    }

    // Priority 2: No persona — show create tariff pills
    if (!state.hasPersona) {
        if (createFooter) {
            _renderTariffButtonsInto('create-tariff-buttons', 'create-pay-btn', 'persona_create', state.tariffs.persona_create || []);
            createFooter.style.display = '';
            if (grid) grid.classList.add('has-footer');
        }
        return;
    }

    // Has persona (with or without credits, no selection) → no sticky footer
    // User uses header "Докупить +" button instead
}

async function generatePhotosetBatch() {
    const selected = state.selectedPersonaStyles || [];
    if (!selected.length) return;
    trackEvent('v2_photoset_batch', { count: selected.length });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('heavy');

    const btn = document.getElementById('photosets-generate-btn');
    btn.disabled = true;
    document.getElementById('photosets-generate-btn-text').textContent = 'Отправляю...';

    try {
        const resp = await fetch('/app/api/persona/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ init_data: state.initData, styles: selected.map(s => ({ slug: s.slug })) }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            if (resp.status === 402) { showScreen('nocredits'); return; }
            throw new Error(data.error || 'Failed');
        }

        if (data.bot_link && tg?.openTelegramLink) { tg.openTelegramLink(data.bot_link); }
        else if (data.bot_link) { window.open(data.bot_link, '_blank'); }
        if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
    } catch (e) {
        console.error('Batch error:', e);
        alert('Ошибка: ' + e.message);
        if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('error');
    } finally {
        btn.disabled = false;
        document.getElementById('photosets-generate-btn-text').textContent = `Сгенерировать (${selected.length})`;
    }
}

// === Unified Tariff System ===

function renderScreenFooters(screen) {
    // 1. Hide ALL footers + reset tariff selection
    ['photosets-create-footer', 'photosets-topup-footer', 'photosets-generate-footer',
     'express-buy-footer', 'custom-buy-footer'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });
    document.querySelectorAll('.has-footer').forEach(el => el.classList.remove('has-footer'));
    if (screen !== 'tariffs-page') state.selectedTariff = null;
    ['create-pay-btn', 'express-footer-pay-btn', 'custom-footer-pay-btn'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) { btn.style.display = 'none'; btn.disabled = false; }
    });
    ['photosets-create-footer', 'express-buy-footer', 'custom-buy-footer'].forEach(id => {
        const footer = document.getElementById(id);
        if (footer) footer.querySelectorAll('.persona-buy-btn.selected').forEach(b => b.classList.remove('selected'));
    });

    // 2. Show footers for current screen
    if (screen === 'photosets') {
        ensurePhotosetsFooter();
    }

    // Express / Custom — show buy footer when no credits
    const hasExpressCredits = state.expressCredits.fast > 0 || !state.expressCredits.free_used;
    if (screen === 'express-catalog' && !hasExpressCredits) {
        const f = document.getElementById('express-buy-footer');
        if (f) {
            _renderTariffButtonsInto('express-footer-buttons', 'express-footer-pay-btn', 'fast', state.tariffs.fast || []);
            f.style.display = '';
            const grid = document.getElementById('v3-styles-grid');
            if (grid) grid.classList.add('has-footer');
        }
    }
    if (screen === 'custom-prompt' && !hasExpressCredits) {
        const f = document.getElementById('custom-buy-footer');
        if (f) {
            _renderTariffButtonsInto('custom-footer-buttons', 'custom-footer-pay-btn', 'fast', state.tariffs.fast || []);
            f.style.display = '';
            const body = document.querySelector('.custom-prompt-body');
            if (body) body.classList.add('has-footer');
        }
    }
}

// Render tariff buttons into any container
function _renderTariffButtonsInto(containerId, payBtnId, mode, tariffs) {
    const container = document.getElementById(containerId);
    const payBtn = document.getElementById(payBtnId);
    if (!container) return;
    if (payBtn) payBtn.style.display = 'none';
    const items = [...tariffs].sort((a, b) => a.credits - b.credits);
    if (!items.length) { container.innerHTML = '<div style="color:var(--text-secondary);text-align:center">Тарифы загружаются...</div>'; return; }
    // Префикс-иконка перед числом кредитов: молния для Express/Custom (fast),
    // призма для всех persona-тарифов (create/topup). Консистентно между разделами.
    const prefix = mode === 'fast'
        ? '&#9889; '
        : '<span class="badge-prism"></span> ';
    container.innerHTML = items.map(t => `
        <button class="persona-buy-btn" data-credits="${t.credits}"
                onclick="selectInlineTariff(this, ${t.credits}, ${t.price}, '${payBtnId}', '${mode}')">
            <span class="persona-buy-btn-price">${t.price} ₽</span>
            <span class="persona-buy-btn-credits">${prefix}${t.credits} ${pluralCredits(t.credits)}</span>
        </button>
    `).join('');
}

function selectInlineTariff(btn, credits, price, payBtnId, mode) {
    // Toggle: повторный клик на том же тарифе отжимает его.
    if (state.selectedTariff && state.selectedTariff.mode === mode && state.selectedTariff.credits === credits) {
        state.selectedTariff = null;
        btn.classList.remove('selected');
        const payBtn = document.getElementById(payBtnId);
        if (payBtn) payBtn.style.display = 'none';
        if (tg?.HapticFeedback) tg.HapticFeedback.selectionChanged();
        return;
    }
    state.selectedTariff = { mode, credits };
    if (tg?.HapticFeedback) tg.HapticFeedback.selectionChanged();
    btn.parentElement.querySelectorAll('.persona-buy-btn').forEach(b => {
        b.classList.toggle('selected', Number(b.dataset.credits) === credits);
    });
    // Сбросить selection и скрыть pay-btn ДРУГИХ секций (на all-tariffs page).
    const sectionsContainer = document.getElementById('tariffs-page-sections');
    if (sectionsContainer) {
        sectionsContainer.querySelectorAll('.persona-buy-btn.selected').forEach(b => {
            if (b.parentElement !== btn.parentElement) b.classList.remove('selected');
        });
        sectionsContainer.querySelectorAll('.persona-buy-pay-btn').forEach(b => {
            if (b.id !== payBtnId) b.style.display = 'none';
        });
    }
    const payBtn = document.getElementById(payBtnId);
    if (payBtn) { payBtn.textContent = `Оплатить ${price} ₽`; payBtn.style.display = ''; }
}

// All tariffs screen (from main "Тарифы" button)
function openAllTariffsScreen() {
    state._tariffReturnTo = 'main';
    state.selectedTariff = null;
    trackEvent('v2_all_tariffs');
    const sections = document.getElementById('tariffs-page-sections');
    const singleButtons = document.getElementById('tariffs-page-buttons');
    if (singleButtons) singleButtons.innerHTML = '';
    document.getElementById('tariffs-page-info').textContent = '';
    // Глобальный pay-btn скрываем — в multi-section экране у каждой секции свой pay-btn СРАЗУ ПОД табами.
    const globalPayBtn = document.getElementById('tariffs-page-pay-btn');
    if (globalPayBtn) { globalPayBtn.style.display = 'none'; globalPayBtn.disabled = false; }

    let html = '';

    // Section 1: Express & Custom
    html += `<div class="tariff-section">
        <h3 class="tariff-section-title">Экспресс и Своя идея</h3>
        <div class="tariff-text">
            <strong>Экспресс</strong> и <strong>Своя идея</strong> используют общий баланс.
            Вы пополняете кредиты один раз и дальше тратите их в любом из этих разделов:<br><br>
            — выбираете готовые стили в <strong>Экспресс</strong><br>
            — или создаёте фото по своему описанию в <strong>Своя идея</strong><br><br>
            <span class="tariff-highlight">1 кредит = 1 фото</span><br>
            в любой доступной модели: Seedream или Nano Banana Pro<br><br>
            Ниже — тарифы на пополнение.
        </div>
        <div class="persona-buy-footer-buttons" id="all-fast-buttons"></div>
        <button class="persona-buy-pay-btn" id="all-fast-pay-btn" style="display:none" onclick="buyTariffPage()">Оплатить</button>
    </div>`;

    // Section 2: Photosets
    html += `<div class="tariff-section">
        <h3 class="tariff-section-title">Фотосеты</h3>
        <div class="tariff-text">
            <strong>Фотосеты</strong> — это премиальный формат для тех, кто хочет не просто красивые картинки, а узнаваемый результат с высоким сходством.<br><br>
            Стоимость зависит от тематики и количества фото в пакете.
            Перед первым заказом мы попросим 10 ваших фото и на их основе создадим <strong>персональную модель</strong>.<br><br>
            Это позволяет добиться более точной передачи лица:
            черты, взгляд, форма лица, мимика и общее ощущение "это действительно я".<br><br>
            <strong>Обучение модели уже входит в стоимость первого фотосета.</strong>
        </div>`;

    if (!state.hasPersona) {
        html += `<div class="tariff-subsection">
            <div class="tariff-subsection-label">Создание модели + кредиты</div>
            <div class="persona-buy-footer-buttons" id="all-create-buttons"></div>
            <button class="persona-buy-pay-btn" id="all-create-pay-btn" style="display:none" onclick="buyTariffPage()">Оплатить</button>
        </div>
        <div class="tariff-text tariff-text-spaced">
            После создания модели следующие заказы будут <strong>существенно дешевле</strong> — останется только выбрать новый сет и получить готовую серию фото.
        </div>
        <div class="tariff-subsection tariff-pills-disabled">
            <div class="tariff-subsection-label">Пополнение кредитов для фотосетов</div>
            <div class="persona-buy-footer-buttons" id="all-topup-buttons-preview"></div>
            <div class="tariff-disabled-note">Доступно после создания модели</div>
        </div>`;
    } else {
        html += `<div class="tariff-subsection">
            <div class="tariff-subsection-label">Пополнение кредитов</div>
            <div class="persona-buy-footer-buttons" id="all-topup-buttons"></div>
            <button class="persona-buy-pay-btn" id="all-topup-pay-btn" style="display:none" onclick="buyTariffPage()">Оплатить</button>
        </div>`;
    }

    html += `</div>`;
    sections.innerHTML = html;

    // Render tariff pills — каждая секция использует СВОЙ pay-btn, кнопка появляется сразу под табами.
    _renderTariffButtonsInto('all-fast-buttons', 'all-fast-pay-btn', 'fast', state.tariffs.fast || []);
    if (!state.hasPersona) {
        _renderTariffButtonsInto('all-create-buttons', 'all-create-pay-btn', 'persona_create', state.tariffs.persona_create || []);
        _renderTariffButtonsInto('all-topup-buttons-preview', 'all-fast-pay-btn', 'persona_topup', state.tariffs.persona_topup || []);
        // Disable preview topup buttons
        const previewContainer = document.getElementById('all-topup-buttons-preview');
        if (previewContainer) {
            previewContainer.querySelectorAll('.persona-buy-btn').forEach(btn => {
                btn.disabled = true;
                btn.style.pointerEvents = 'none';
            });
        }
    } else {
        _renderTariffButtonsInto('all-topup-buttons', 'all-topup-pay-btn', 'persona_topup', state.tariffs.persona_topup || []);
    }
    showScreen('tariffs-page');
}

// Unified tariff screen (persona_create / persona_topup / fast)
function openUnifiedTariffScreen(mode, returnTo) {
    state.selectedTariff = null;
    state._tariffReturnTo = returnTo || 'main';
    trackEvent('v2_tariff_screen', { mode });

    // Clear sections (used by openAllTariffsScreen)
    const sections = document.getElementById('tariffs-page-sections');
    if (sections) sections.innerHTML = '';

    const titles = {
        persona_create: 'Персона + кредиты для генерации',
        persona_topup: 'Пополните кредиты — выгоднее пакетом',
        fast: 'Докупите фото для экспресс-генерации',
    };
    document.getElementById('tariffs-page-info').textContent = titles[mode] || '';
    _renderTariffButtonsInto('tariffs-page-buttons', 'tariffs-page-pay-btn', mode, state.tariffs[mode] || []);
    showScreen('tariffs-page');
}

function goBackTariffs() {
    goBack(state._tariffReturnTo || 'main');
}

async function buyTariffPage() {
    if (!state.selectedTariff) return;
    const { mode, credits } = state.selectedTariff;
    const endpoints = {
        persona_create: '/app/api/persona/buy',
        persona_topup: '/app/api/persona/topup',
        fast: '/app/api/fast/buy',
    };
    const endpoint = endpoints[mode];
    if (!endpoint) return;

    trackEvent('v2_tariff_buy', { mode, credits });
    if (tg?.HapticFeedback) tg.HapticFeedback.impactOccurred('heavy');

    const payBtn = document.getElementById('tariffs-page-pay-btn');
    if (payBtn) payBtn.disabled = true;

    try {
        const resp = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ init_data: state.initData, credits }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Payment failed');

        if (data.payment_url) {
            if (tg) { tg.openLink(data.payment_url); }
            else { window.open(data.payment_url, '_blank'); }
            // For persona_create: close Mini App — bot will guide photo upload
            if (mode === 'persona_create' && tg) {
                tg.close();
            }
        }
    } catch (e) {
        console.error('Tariff buy error:', e);
        alert('Ошибка оплаты: ' + e.message);
    } finally {
        if (payBtn) payBtn.disabled = false;
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
    const overlay = document.getElementById('lightbox-overlay');
    if (overlay) overlay.style.display = 'none';
}

function openPersonaInfoModal() {
    const m = document.getElementById('persona-info-modal');
    if (m) m.style.display = 'flex';
}
function closePersonaInfoModal() {
    const m = document.getElementById('persona-info-modal');
    if (m) m.style.display = 'none';
}

function openPersonaPendingModal() {
    const m = document.getElementById('persona-pending-modal');
    if (m) m.style.display = 'flex';
    state._personaPendingShown = true;
}
function closePersonaPendingModal() {
    const m = document.getElementById('persona-pending-modal');
    if (m) m.style.display = 'none';
    if (tg?.close) tg.close();
}

function openStylePreviewLightbox(style) {
    const previewUrl = style.image_url || (style.preview_urls && style.preview_urls[0]) || '';
    if (!previewUrl) return;
    const lb = document.getElementById('lightbox');
    const img = document.getElementById('lightbox-img');
    const overlay = document.getElementById('lightbox-overlay');
    const nameEl = document.getElementById('lightbox-style-name');
    const costEl = document.getElementById('lightbox-style-cost');
    if (!lb || !img) return;
    img.src = previewUrl;
    if (overlay && nameEl && costEl) {
        nameEl.textContent = style.label;
        costEl.textContent = 'Стоимость — ⚡ 1 кредит';
        overlay.style.display = '';
    }
    lb.style.display = 'flex';
    trackEvent('v2_style_preview_nocredits', { style_id: style.id });
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
            providers: { seedream: { max_photos: 5 }, 'nano-banana-pro': { max_photos: 8 } },
            max_prompt_length: 2000, allowed_mime: ['image/jpeg','image/png','image/webp'], max_file_size_mb: 15,
        };
    }
}

function goToCustomPrompt() {
    showScreen('custom-prompt');
    loadCustomCapabilities().then(() => {
        updateCustomMaxPhotos();
        updateBalanceDisplays();
        renderScreenFooters('custom-prompt');
    });
    state.customFiles = [];
    state.customProvider = 'seedream';
    state.customRequestId = null;
    renderCustomPhotos();
    const textarea = document.getElementById('custom-prompt-input');
    if (textarea) { textarea.value = ''; }
    const counter = document.getElementById('custom-prompt-length');
    if (counter) counter.textContent = '0';
    // Reset provider buttons + generate label
    document.querySelectorAll('#custom-provider-options .provider-option').forEach(btn => {
        btn.classList.toggle('selected', btn.dataset.provider === 'seedream');
    });
    _updateGenerateLabel('custom');
    // Hide generate if no credits — footer shows tariffs
    const customGenBtn = document.getElementById('custom-generate-btn');
    if (customGenBtn) customGenBtn.style.display = canGenerateExpress() ? '' : 'none';
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
    _updateGenerateLabel('custom');
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
        pollCustomStatus();

    } catch (e) {
        console.error('Custom generation error:', e);
        alert('Ошибка: ' + e.message);
    } finally {
        if (btn) { btn.disabled = false; _updateGenerateLabel('custom'); }
    }
}
