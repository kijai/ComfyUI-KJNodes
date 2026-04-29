const { app } = window.comfyAPI.app;

function snapshotOld(oldNode) {
    const data = oldNode.serialize();
    const widgetValuesByName = {};
    for (const w of oldNode.widgets || []) widgetValuesByName[w.name] = w.value;

    const inputSnapshot = (oldNode.inputs || []).map(i => ({
        name: i.name,
        widgetName: i.widget?.name ?? null,
        linkId: i.link ?? null,
    }));
    const outputLinksByName = {};
    for (const o of oldNode.outputs || []) {
        if (o.links?.length) outputLinksByName[o.name] = [...o.links];
    }
    return { data, widgetValuesByName, inputSnapshot, outputLinksByName };
}

function applyCosmetics(newNode, old) {
    if (old.pos) newNode.pos = [...old.pos];
    if (old.size) {
        const min = newNode.computeSize?.() || [0, 0];
        newNode.size = [Math.max(old.size[0], min[0]), Math.max(old.size[1], min[1])];
    }
    if (old.title) newNode.title = old.title;
    if (old.color) newNode.color = old.color;
    if (old.bgcolor) newNode.bgcolor = old.bgcolor;
    if (old.flags) newNode.flags = { ...(newNode.flags || {}), ...old.flags };
    if (typeof old.mode === "number") newNode.mode = old.mode;
    if (old.properties) newNode.properties = { ...(newNode.properties || {}), ...old.properties };
}

function fixNode(oldNode, comfyClass, { resetValues = false } = {}) {
    const graph = app.canvas.graph;
    const snap = snapshotOld(oldNode);

    const newNode = LiteGraph.createNode(comfyClass);
    if (!newNode) {
        console.error(`[KJNodes.FixNode] Unknown node type: ${comfyClass}`);
        return null;
    }

    try {
        graph.add(newNode, false);
        applyCosmetics(newNode, snap.data);

        // Restore widget values by name (live widgets, not widgets_values array).
        if (!resetValues) {
            for (const w of newNode.widgets || []) {
                if (Object.prototype.hasOwnProperty.call(snap.widgetValuesByName, w.name)) {
                    try { w.value = snap.widgetValuesByName[w.name]; } catch {}
                }
            }
        }

        // Reconnect inputs by name.
        for (const inp of snap.inputSnapshot) {
            if (inp.linkId == null) continue;
            const link = app.graph.links[inp.linkId];
            if (!link) continue;
            const src = app.graph.getNodeById(link.origin_id);
            const newSlot = newNode.findInputSlot?.(inp.name);
            if (!src || newSlot == null || newSlot < 0) continue;
            try { src.connect(link.origin_slot, newNode, newSlot); } catch {}
        }

        // Reconnect outputs by name.
        (newNode.outputs || []).forEach((out, i) => {
            const linkIds = snap.outputLinksByName[out.name];
            if (!linkIds) return;
            for (const id of linkIds) {
                const link = app.graph.links[id];
                if (!link) continue;
                const tgt = app.graph.getNodeById(link.target_id);
                if (!tgt) continue;
                try { newNode.connect(i, tgt, link.target_slot); } catch {}
            }
        });

        graph.remove(oldNode);
        app.graph.afterChange();
        requestAnimationFrame(() => app.canvas.setDirty(true, true));
        return newNode;
    } catch (err) {
        console.error("[KJNodes.FixNode] Aborting, rolling back:", err);
        try { graph.remove(newNode); } catch {}
        return null;
    }
}

window.kjNodes = window.kjNodes || {};
window.kjNodes.recreateNode = fixNode;
