/* ═══════════════════════════════════════════════════════════
   NAVIS AI – Frontend Application
   ═══════════════════════════════════════════════════════════ */

class NavisApp {
        constructor() {
                this.els = {
                        messages: document.getElementById('messages'),
                        userInput: document.getElementById('userInput'),
                        sendBtn: document.getElementById('sendBtn'),
                        stopBtn: document.getElementById('stopBtn'),
                        voiceBtn: document.getElementById('voiceBtn'),
                        trainToggle: document.getElementById('trainToggle'),
                        trainingPanel: document.getElementById('trainingPanel'),
                        closeTraining: document.getElementById('closeTraining'),
                        addTraining: document.getElementById('addTraining'),
                        trainQuestion: document.getElementById('trainQuestion'),
                        trainAnswer: document.getElementById('trainAnswer'),
                        trainingList: document.getElementById('trainingList'),
                        overlay: document.getElementById('overlay'),
                        chatContainer: document.getElementById('chatContainer'),
                        welcomeHero: document.getElementById('welcomeHero'),
                        resetBtn: document.getElementById('resetBtn'),
                        ttsToggle: document.getElementById('ttsToggle'),
                        toastContainer: document.getElementById('toastContainer'),
                };

                this.isRecording = false;
                this.continuousListening = false;
                this.recognition = null;
                this.ttsEnabled = true;
                this.isProcessing = false;
                this.isSpeaking = false;
                this.voicesLoaded = false;
                this.abortController = null;
                this.currentTypingEl = null;

                this.init();
        }

        init() {
                this.initSpeechRecognition();
                this.initTTS();
                this.bindEvents();
                this.loadTrainingData();
                this.autoResize();
                // Mark TTS toggle as active by default
                if (this.els.ttsToggle) this.els.ttsToggle.classList.add('active');
                if (typeof marked !== 'undefined') {
                        marked.setOptions({ breaks: true, gfm: true });
                }
        }

        /* ── TTS Init (cross-platform voice loading) ────────── */
        initTTS() {
                if (!window.speechSynthesis) return;
                // Voices load asynchronously on many platforms
                const loadVoices = () => {
                        this.voices = window.speechSynthesis.getVoices();
                        if (this.voices.length) this.voicesLoaded = true;
                };
                loadVoices();
                window.speechSynthesis.onvoiceschanged = loadVoices;
        }

        /* ── Speech Recognition ─────────────────────────────── */
        initSpeechRecognition() {
                const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (!SR) return;
                this.recognition = new SR();
                this.recognition.continuous = false;
                this.recognition.interimResults = true;
                const langSelect = document.getElementById('langSelect');
                this.recognition.lang = langSelect ? langSelect.value : 'en-IN';

                if (langSelect) {
                        langSelect.addEventListener('change', (e) => {
                                if (this.recognition) this.recognition.lang = e.target.value;
                        });
                }

                this.recognition.onresult = (e) => {
                        const transcript = Array.from(e.results)
                                .map(r => r[0].transcript).join('');
                        this.els.userInput.value = transcript;
                        this.autoResize();
                        if (e.results[0] && e.results[0].isFinal) {
                                this.stopRecording();
                                // Add a very short delay to allow UI to update
                                setTimeout(() => this.sendMessage(), 50);
                        }
                };
                this.recognition.onerror = (e) => {
                        console.error('Speech error:', e.error);
                        this.stopRecording();
                        if (e.error === 'not-allowed') {
                                this.continuousListening = false;
                                this.toast('Microphone access denied', 'error');
                        }
                };
                this.recognition.onend = () => {
                        this.stopRecording();
                        // Auto-restart if continuous mode is ON, and bot is not busy
                        if (this.continuousListening && !this.isProcessing && !this.isSpeaking) {
                                setTimeout(() => {
                                        if (this.continuousListening && !this.isProcessing && !this.isSpeaking && !this.isRecording) {
                                                this.startRecording();
                                        }
                                }, 150);
                        }
                };
        }

        startRecording() {
                if (!this.recognition) { this.toast('Voice not supported in this browser', 'error'); return; }
                if (this.isRecording) return;
                this.isRecording = true;
                this.els.voiceBtn.classList.add('recording');
                this.els.userInput.placeholder = 'Listening...';
                try { this.recognition.start(); } catch (e) { }
        }

        stopRecording() {
                if (!this.isRecording) return;
                this.isRecording = false;
                this.els.voiceBtn.classList.remove('recording');
                this.els.userInput.placeholder = 'Ask Navis anything...';
                try { this.recognition.stop(); } catch (e) { }
        }

        /* ── Servo Mouth Sync ────────────────────────────────── */
        /**
         * Notifies the Python backend to open (state=1) or close (state=0)
         * the ESP32 mouth servo. Called at TTS start and TTS end.
         */
        setMouthState(state) {
                fetch('/api/mouth', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ state }),
                }).catch(() => { /* silently ignore if server is down */ });
        }

        /* ── Text-to-Speech (always on, cross-platform) ────── */
        /**
         * Get the current language selection from the dropdown.
         */
        getSelectedLang() {
                const langSelect = document.getElementById('langSelect');
                return langSelect ? langSelect.value : 'en-IN';
        }

        /**
         * Detect language from text content as a fallback.
         * Checks for Devanagari script, Kannada script, and common Hindi transliterated words.
         */
        detectLanguage(text) {
                if (/[\u0900-\u097F]/.test(text)) return 'hi-IN';
                if (/[\u0C80-\u0CFF]/.test(text)) return 'kn-IN';
                // Common Hindi/Hinglish words (Latin script) — catches LLM responses in Hinglish
                const hindiWords = /\b(hai|hain|ka|ki|ke|kya|nahi|nahin|aur|mein|yeh|woh|toh|bhi|kaise|kab|kaha|kyun|aap|hum|tum|ji|tha|thi|the|ho|hota|hoti|karo|karte|karna|accha|bahut|baat|bol|dekho|suno|matlab|zaroor|namaste|dhanyavaad|shukriya|kaam|aise|waise|lekin|par|abhi|sabhi)\b/i;
                if (hindiWords.test(text)) return 'hi-IN';
                return 'en-IN';
        }

        speak(text, langHint) {
                if (!this.ttsEnabled || !window.speechSynthesis) {
                        this.onSpeechDone();
                        return;
                }
                window.speechSynthesis.cancel();
                // Clean markdown syntax for natural speech
                const clean = text
                        .replace(/```[\s\S]*?```/g, ' code block ')   // code blocks
                        .replace(/`([^`]+)`/g, '$1')                  // inline code
                        .replace(/#{1,6}\s*/g, '')                    // headings
                        .replace(/[*_~]{1,3}/g, '')                   // bold/italic/strike
                        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')     // links
                        .replace(/[>[\]()]/g, '')                    // remaining markdown
                        .replace(/\n+/g, '. ')                        // newlines to pauses
                        .replace(/\.\s*\./g, '.')                     // double periods
                        .trim();

                if (!clean) { this.onSpeechDone(); return; }

                this.isSpeaking = true;
                this.speechStopped = false;
                this.showStopBtn();
                this.setMouthState(1);  // ← Signal ESP32: mouth OPEN

                const utt = new SpeechSynthesisUtterance(clean);
                utt.rate = 0.95;
                utt.pitch = 1;

                // Determine language: use backend hint > text detection (now works with native script) > dropdown
                const detectedLang = this.detectLanguage(clean);
                const lang = langHint || detectedLang || this.getSelectedLang();
                const isHindi = lang.startsWith('hi');
                const isKannada = lang.startsWith('kn');

                // Cross-platform voice selection
                const voices = this.voices || window.speechSynthesis.getVoices();

                // Female filter (catches many defaults)
                const isFemale = (v) => /Female|Samantha|Zira|Veena|Heera|Neerja|Victoria|Karen|Moira|Tessa|Luciana|Monica|Lekha|Flo|Grandma|Kathy/i.test(v.name);

                let preferred;
                if (isHindi) {
                        // Hindi voice: works even when text is in Latin script (Hinglish)
                        preferred = voices.find(v => v.name.includes('Google हिन्दी'))
                                || voices.find(v => v.name.includes('Microsoft Hemant'))
                                || voices.find(v => v.lang.startsWith('hi') && !isFemale(v))
                                || voices.find(v => v.lang.startsWith('hi'))
                                || voices.find(v => v.name.includes('Rishi'))
                                || voices[0];
                        // For Hindi, explicitly set lang so the synthesis engine handles Hindi phonetics
                        utt.lang = 'hi-IN';
                } else if (isKannada) {
                        preferred = voices.find(v => v.name.includes('Soumya'))
                                || voices.find(v => v.lang.startsWith('kn'))
                                || voices[0];
                        utt.lang = 'kn-IN';
                } else {
                        // Strictly prioritize Indian English accents
                        preferred = voices.find(v => v.name === 'Rishi')
                                || voices.find(v => v.name === 'Ravi')
                                || voices.find(v => v.name === 'Microsoft Ravi')
                                || voices.find(v => v.name === 'Google UK English Male')
                                || voices.find(v => v.name === 'Daniel')
                                || voices.find(v => v.lang === 'en-IN' && !isFemale(v))
                                || voices.find(v => v.lang.startsWith('en') && !isFemale(v))
                                || voices[0];
                        utt.lang = 'en-IN';
                }

                if (preferred) {
                        utt.voice = preferred;
                }

                // Chrome bug workaround: long texts get cut off
                if (clean.length > 200) {
                        this.speakChunked(clean, utt.rate, utt.pitch, preferred, utt.lang);
                } else {
                        utt.onend = () => this.onSpeechDone();
                        utt.onerror = () => this.onSpeechDone();
                        window.speechSynthesis.speak(utt);
                }
        }

        speakChunked(text, rate, pitch, voice, lang) {
                const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
                let i = 0;
                const speakNext = () => {
                        if (this.speechStopped || i >= sentences.length) { this.onSpeechDone(); return; }
                        const utt = new SpeechSynthesisUtterance(sentences[i].trim());
                        utt.rate = rate;
                        utt.pitch = pitch;
                        if (voice) utt.voice = voice;
                        if (lang) utt.lang = lang;
                        utt.onend = () => { i++; speakNext(); };
                        utt.onerror = () => { i++; speakNext(); };
                        window.speechSynthesis.speak(utt);
                };
                speakNext();
        }

        onSpeechDone() {
                this.isSpeaking = false;
                this.setMouthState(0);  // ← Signal ESP32: mouth CLOSE
                if (!this.isProcessing) {
                        this.showSendBtn();
                        if (this.continuousListening) {
                                setTimeout(() => {
                                        if (this.continuousListening && !this.isProcessing && !this.isSpeaking && !this.isRecording) {
                                                this.startRecording();
                                        }
                                }, 500); // 500ms delay to clear audio buffer
                        }
                }
        }

        showStopBtn() {
                this.els.sendBtn.style.display = 'none';
                this.els.stopBtn.style.display = 'flex';
        }

        showSendBtn() {
                this.els.sendBtn.style.display = 'flex';
                this.els.stopBtn.style.display = 'none';
        }

        /* ── Events ─────────────────────────────────────────── */
        bindEvents() {
                // Send
                this.els.sendBtn.addEventListener('click', () => this.sendMessage());
                this.els.userInput.addEventListener('keydown', (e) => {
                        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.sendMessage(); }
                });

                // Stop
                this.els.stopBtn.addEventListener('click', () => this.stopResponse());
                document.addEventListener('keydown', (e) => {
                        if (e.key === 'Escape' && this.isProcessing) this.stopResponse();
                });

                // Voice
                this.els.voiceBtn.addEventListener('click', () => {
                        if (this.continuousListening) {
                                this.continuousListening = false;
                                this.stopRecording();
                                this.toast('Continuous listening OFF', 'info');
                        } else {
                                this.continuousListening = true;
                                this.startRecording();
                                this.toast('Continuous listening ON', 'success');
                        }
                });

                // TTS toggle
                this.els.ttsToggle.addEventListener('click', () => {
                        this.ttsEnabled = !this.ttsEnabled;
                        this.els.ttsToggle.classList.toggle('active', this.ttsEnabled);
                        if (!this.ttsEnabled) window.speechSynthesis?.cancel();
                        this.toast(this.ttsEnabled ? 'Voice replies ON' : 'Voice replies OFF', 'info');
                });

                // Training panel
                this.els.trainToggle.addEventListener('click', () => this.openTraining());
                this.els.closeTraining.addEventListener('click', () => this.closeTraining());
                this.els.overlay.addEventListener('click', () => this.closeTraining());
                this.els.addTraining.addEventListener('click', () => this.addTrainingData());

                // Quick actions
                document.querySelectorAll('.quick-btn').forEach(btn => {
                        btn.addEventListener('click', () => {
                                this.els.userInput.value = btn.dataset.msg;
                                this.sendMessage();
                        });
                });

                // Reset
                this.els.resetBtn.addEventListener('click', () => this.resetChat());

                // Auto-resize
                this.els.userInput.addEventListener('input', () => this.autoResize());
        }

        autoResize() {
                const ta = this.els.userInput;
                ta.style.height = 'auto';
                ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
        }

        /* ── Chat ───────────────────────────────────────────── */
        async sendMessage() {
                const text = this.els.userInput.value.trim();
                if (!text || this.isProcessing) return;

                // Hide welcome
                if (this.els.welcomeHero) {
                        this.els.welcomeHero.style.display = 'none';
                }

                // Capture current language selection before sending
                const selectedLang = this.getSelectedLang();

                this.addMessage(text, 'user');
                this.els.userInput.value = '';
                this.autoResize();
                this.isProcessing = true;

                // Show stop button, hide send button
                this.showStopBtn();

                // Create abort controller for this request
                this.abortController = new AbortController();

                // Show typing indicator
                this.currentTypingEl = this.showTyping();

                try {
                        const res = await fetch('/api/chat', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ message: text, lang: selectedLang }),
                                signal: this.abortController.signal
                        });
                        const data = await res.json();
                        if (this.currentTypingEl) this.currentTypingEl.remove();

                        const sourceLabel = data.source === 'trained' ? '🎓 Trained' : data.source === 'ai' ? '✨ AI' : '';
                        this.addMessage(data.response, 'navis', sourceLabel);
                        this.isProcessing = false;
                        // Use language from backend response, fallback to dropdown selection
                        this.speak(data.response, data.lang || selectedLang);
                } catch (err) {
                        if (this.currentTypingEl) this.currentTypingEl.remove();
                        if (err.name !== 'AbortError') {
                                this.addMessage('Sorry, I couldn\'t process that. Please check the server.', 'navis', '⚠️ Error');
                        }
                        this.isProcessing = false;
                        this.showSendBtn();
                }

                this.abortController = null;
                this.currentTypingEl = null;
        }

        stopResponse() {
                // 1. Cancel the fetch request
                if (this.abortController) {
                        this.abortController.abort();
                        this.abortController = null;
                }

                // 2. Mark speech as stopped (prevents chunked speech from continuing)
                this.isSpeaking = false;
                this.speechStopped = true;
                this.setMouthState(0);  // ← Signal ESP32: mouth CLOSE (user aborted)

                // 3. Force-stop text-to-speech (multiple attempts for Safari/iOS reliability)
                if (window.speechSynthesis) {
                        window.speechSynthesis.cancel();
                        // Safari sometimes needs a pause→resume→cancel cycle
                        window.speechSynthesis.pause();
                        window.speechSynthesis.resume();
                        window.speechSynthesis.cancel();
                        // Double-check after a short delay (last resort for stubborn browsers)
                        setTimeout(() => {
                                if (window.speechSynthesis.speaking) {
                                        window.speechSynthesis.cancel();
                                }
                        }, 100);
                }

                // 4. Remove typing indicator
                if (this.currentTypingEl) {
                        this.currentTypingEl.remove();
                        this.currentTypingEl = null;
                }

                // 5. Always show feedback
                this.toast('🛑 Stopped', 'info');

                this.isProcessing = false;
                this.showSendBtn();

                if (this.continuousListening) {
                        setTimeout(() => {
                                if (this.continuousListening && !this.isProcessing && !this.isSpeaking && !this.isRecording) {
                                        this.startRecording();
                                }
                        }, 500);
                }
        }

        addMessage(text, sender, sourceTag = '') {
                const div = document.createElement('div');
                div.className = `message ${sender}`;

                const avatar = document.createElement('div');
                avatar.className = 'avatar';
                if (sender === 'navis') {
                        avatar.innerHTML = '<img src="/static/images/robomanthan_logo.png" onerror="this.outerHTML=\'N\'">';
                } else {
                        avatar.textContent = 'You';
                }

                const bubble = document.createElement('div');
                bubble.className = 'bubble';

                if (sender === 'navis' && typeof marked !== 'undefined') {
                        bubble.innerHTML = marked.parse(text);
                } else {
                        bubble.textContent = text;
                }

                if (sourceTag) {
                        const tag = document.createElement('span');
                        tag.className = 'source-tag';
                        tag.textContent = sourceTag;
                        bubble.appendChild(tag);
                }

                div.appendChild(avatar);
                div.appendChild(bubble);
                this.els.messages.appendChild(div);
                this.scrollToBottom();
        }

        showTyping() {
                const div = document.createElement('div');
                div.className = 'message navis';
                div.innerHTML = `
            <div class="avatar"><img src="/static/images/robomanthan_logo.png" onerror="this.outerHTML='N'"></div>
            <div class="bubble"><div class="typing-indicator">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div></div>`;
                this.els.messages.appendChild(div);
                this.scrollToBottom();
                return div;
        }

        scrollToBottom() {
                this.els.chatContainer.scrollTop = this.els.chatContainer.scrollHeight;
        }

        async resetChat() {
                try { await fetch('/api/reset', { method: 'POST' }); } catch (e) { }
                this.els.messages.innerHTML = '';
                if (this.els.welcomeHero) {
                        this.els.messages.appendChild(this.els.welcomeHero);
                        this.els.welcomeHero.style.display = '';
                }
                this.toast('Chat reset', 'info');
        }

        /* ── Training Panel ─────────────────────────────────── */
        openTraining() {
                this.els.trainingPanel.classList.add('open');
                this.els.overlay.classList.add('active');
                this.loadTrainingData();
        }

        closeTraining() {
                this.els.trainingPanel.classList.remove('open');
                this.els.overlay.classList.remove('active');
        }

        async loadTrainingData() {
                try {
                        const res = await fetch('/api/training-data');
                        const data = await res.json();
                        this.renderTrainingList(data.qa_pairs || []);
                } catch (e) {
                        this.els.trainingList.innerHTML = '<p class="training-empty">Could not load data</p>';
                }
        }

        renderTrainingList(pairs) {
                if (!pairs.length) {
                        this.els.trainingList.innerHTML = '<p class="training-empty">No custom training yet.<br>Add Q&A pairs above!</p>';
                        return;
                }
                this.els.trainingList.innerHTML = pairs.map(qa => `
            <div class="training-item">
                <div class="ti-q">Q: ${this.esc(qa.question)}</div>
                <div class="ti-a">A: ${this.esc(qa.answer)}</div>
                <button class="ti-delete" data-id="${qa.id}">✕ Remove</button>
            </div>
        `).join('');

                this.els.trainingList.querySelectorAll('.ti-delete').forEach(btn => {
                        btn.addEventListener('click', () => this.deleteTraining(btn.dataset.id));
                });
        }

        async addTrainingData() {
                const q = this.els.trainQuestion.value.trim();
                const a = this.els.trainAnswer.value.trim();
                if (!q || !a) { this.toast('Fill in both fields', 'error'); return; }

                try {
                        const res = await fetch('/api/train', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ question: q, answer: a })
                        });
                        if (res.ok) {
                                this.els.trainQuestion.value = '';
                                this.els.trainAnswer.value = '';
                                this.toast('Training added!', 'success');
                                this.loadTrainingData();
                        }
                } catch (e) { this.toast('Failed to add', 'error'); }
        }

        async deleteTraining(id) {
                try {
                        await fetch(`/api/training-data/${id}`, { method: 'DELETE' });
                        this.toast('Removed', 'info');
                        this.loadTrainingData();
                } catch (e) { }
        }

        /* ── Utilities ──────────────────────────────────────── */
        esc(str) {
                const d = document.createElement('div');
                d.textContent = str;
                return d.innerHTML;
        }

        toast(msg, type = 'info') {
                const t = document.createElement('div');
                t.className = `toast ${type}`;
                t.textContent = msg;
                this.els.toastContainer.appendChild(t);
                setTimeout(() => t.remove(), 3200);
        }
}

// ── Boot ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => new NavisApp());
