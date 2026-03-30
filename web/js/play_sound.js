const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

app.registerExtension({
    name: "KJNodes.PlaySound",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PlaySoundKJ") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            onExecuted?.call(this, output);

            const audios = output?.audio;
            if (!audios?.length) return;

            const modeWidget = this.widgets?.find(w => w.name === "mode");
            const volumeWidget = this.widgets?.find(w => w.name === "volume");
            const durationWidget = this.widgets?.find(w => w.name === "duration");
            const mode = modeWidget?.value ?? "always";
            const volume = volumeWidget?.value ?? 0.5;
            const duration = durationWidget?.value ?? 0;

            // on_change: skip if audio content hasn't changed
            if (mode === "on_change") {
                const audioHash = output?.audio_hash?.[0];
                if (audioHash != null && this._kjLastAudioHash === audioHash) return;
                this._kjLastAudioHash = audioHash;
            }

            // Clean up previous state
            if (this._kjStatusListener) {
                api.removeEventListener("status", this._kjStatusListener);
                this._kjStatusListener = null;
            }
            clearTimeout(this._kjQueueDebounce);
            this._kjPendingAudio = null;

            if (this._kjPlayingAudio) {
                this._kjPlayingAudio.pause();
                this._kjPlayingAudio = null;
            }
            if (this._kjPlayTimer != null) {
                clearTimeout(this._kjPlayTimer);
                this._kjPlayTimer = null;
            }

            const startPlayback = () => {
                const { filename, subfolder, type } = audios[0];
                const params = new URLSearchParams({
                    filename: filename ?? "",
                    subfolder: subfolder ?? "",
                    type: type ?? "temp",
                });
                const url = api.apiURL(`/view?${params.toString()}`);
                const audio = new Audio(url);
                audio.volume = Math.max(0, Math.min(1, volume));
                audio.play().catch(() => {});
                this._kjPlayingAudio = audio;
                if (duration > 0) {
                    this._kjPlayTimer = setTimeout(() => {
                        audio.pause();
                        this._kjPlayingAudio = null;
                        this._kjPlayTimer = null;
                    }, duration * 1000);
                }
            };

            if (mode === "on_empty_queue") {
                this._kjPendingAudio = startPlayback;
                this._kjStatusListener = ({ detail }) => {
                    const remaining = detail?.exec_info?.queue_remaining ?? 0;
                    if (remaining === 0) {
                        // Debounce: confirm queue is truly empty
                        // (status can briefly show 0 between dispatches)
                        clearTimeout(this._kjQueueDebounce);
                        this._kjQueueDebounce = setTimeout(() => {
                            if (this._kjPendingAudio) {
                                this._kjPendingAudio();
                                this._kjPendingAudio = null;
                            }
                            api.removeEventListener("status", this._kjStatusListener);
                            this._kjStatusListener = null;
                        }, 1000);
                    } else {
                        clearTimeout(this._kjQueueDebounce);
                    }
                };
                api.addEventListener("status", this._kjStatusListener);
            } else {
                startPlayback();
            }
        };
    },
});
