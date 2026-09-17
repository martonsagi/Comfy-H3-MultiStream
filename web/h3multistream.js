// SPDX-FileCopyrightText: 2026 Márton Sági
// SPDX-License-Identifier: GPL-3.0-only
// H3 MultiStream controls: main menu -> H3 MultiStream (release VAE workers, clear caches, status).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function toast(severity, summary, detail) {
  const t = app.extensionManager?.toast;
  if (t?.add) {
    t.add({ severity, summary, detail, life: severity === "error" ? 8000 : 5000 });
  } else {
    console[severity === "error" ? "error" : "log"](`[H3 MultiStream] ${summary}: ${detail}`);
    if (severity === "error") alert(`${summary}\n${detail}`);
  }
}

async function call(path, method = "POST") {
  const res = await api.fetchApi(path, { method });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `${res.status} ${res.statusText}`);
  return body;
}

function describeStatus(s) {
  const plans = s.gpu_plans || {};
  const planLines = Object.keys(plans).map((k) => `GPUs (${k}): ${plans[k].plan}`);
  const st = s.last_split_step || {};
  const step = st.ranks
    ? `Last split step: ${st.seconds}s on ${st.ranks.join("+")} (heads ${st.heads_per_rank.join("/")}, exchange ${st.exchange} ${st.exchange_seconds_per_rank.join("/")}s, ${st.moved_gib} GiB moved)`
    : "Last split step: none yet";
  const workers = (s.vae_workers || [])
    .map((w) => `GPU ${w.gpu}${w.slot ? ` slot ${w.slot}` : ""}: ${w.state}${w.busy ? " (busy)" : ""}`)
    .join(", ") || "none";
  const streams = (s.side_streams || []).join(", ") || "none";
  const hooks = (s.hooks || []).map((h) => `${h.hook} (${(h.methods || []).join("/")})`).join(", ") || "none";
  const caches = (s.caches || []).map((c) => `${c.group} ${c.GiB} GiB`).join(", ") || "none";
  const te = s.text_encoder_outputs || {};
  return [
    ...(planLines.length ? planLines : ["GPUs: no plan yet"]),
    step,
    `VAE workers: ${workers}`,
    `Prefetch side streams: ${streams}`,
    `Hooks attached: ${hooks}`,
    `Weight caches: ${caches}`,
    `Text-encoder outputs: ${te.entries ?? 0} (hits ${te.hits ?? 0}, misses ${te.misses ?? 0})`,
    s.ram || "",
  ].join("\n");
}

app.registerExtension({
  name: "H3MultiStream.Controls",
  commands: [
    {
      id: "H3MultiStream.ReleaseVAEWorkers",
      label: "Release VAE workers",
      icon: "pi pi-power-off",
      function: async () => {
        try {
          const r = await call("/h3multistream/vae_workers/release");
          toast(r.busy ? "warn" : "success", "H3 MultiStream", r.message);
        } catch (e) {
          toast("error", "Release VAE workers failed", String(e));
        }
      },
    },
    {
      id: "H3MultiStream.DetachHooks",
      label: "Detach all hooks (text encoder / VAE)",
      icon: "pi pi-link",
      function: async () => {
        try {
          const r = await call("/h3multistream/hooks/detach");
          toast(r.busy ? "warn" : "success", "H3 MultiStream", r.message);
        } catch (e) {
          toast("error", "Detach hooks failed", String(e));
        }
      },
    },
    {
      id: "H3MultiStream.ReleaseSideStreams",
      label: "Release prefetch side streams",
      icon: "pi pi-forward",
      function: async () => {
        try {
          const r = await call("/h3multistream/streams/release");
          toast(r.busy ? "warn" : "success", "H3 MultiStream", r.message);
        } catch (e) {
          toast("error", "Release side streams failed", String(e));
        }
      },
    },
    {
      id: "H3MultiStream.ClearWeightCaches",
      label: "Clear weight caches (DiT / text encoder / VAE)",
      icon: "pi pi-trash",
      function: async () => {
        try {
          const r = await call("/h3multistream/cache/clear_weights");
          toast("success", "H3 MultiStream", r.message);
        } catch (e) {
          toast("error", "Clear weight caches failed", String(e));
        }
      },
    },
    {
      id: "H3MultiStream.ClearTextEncoderOutputs",
      label: "Clear text-encoder output cache",
      icon: "pi pi-eraser",
      function: async () => {
        try {
          const r = await call("/h3multistream/cache/clear_outputs");
          toast("success", "H3 MultiStream", r.message);
        } catch (e) {
          toast("error", "Clear text-encoder outputs failed", String(e));
        }
      },
    },
    {
      id: "H3MultiStream.Status",
      label: "Show status",
      icon: "pi pi-info-circle",
      function: async () => {
        try {
          toast("info", "H3 MultiStream status", describeStatus(await call("/h3multistream/status", "GET")));
        } catch (e) {
          toast("error", "Status failed", String(e));
        }
      },
    },
  ],
  menuCommands: [
    {
      // top-level path: the new frontend lists it in the main (File) dropdown, like KJNodes' ["KJNodes", ...]
      path: ["H3 MultiStream"],
      commands: [
        "H3MultiStream.ReleaseVAEWorkers",
        "H3MultiStream.DetachHooks",
        "H3MultiStream.ReleaseSideStreams",
        "H3MultiStream.ClearWeightCaches",
        "H3MultiStream.ClearTextEncoderOutputs",
        "H3MultiStream.Status",
      ],
    },
  ],
});
