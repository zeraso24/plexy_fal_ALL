import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "PlexardFalMultiImage";

let schemaPromise = null;
function getSchema() {
  if (!schemaPromise) {
    schemaPromise = api
      .fetchApi("/plexard_fal/models")
      .then((r) => r.json())
      .catch((e) => {
        console.warn("[PLEXARD_FAL] Could not load model schema:", e);
        schemaPromise = null;
        return null;
      });
  }
  return schemaPromise;
}

const findWidget = (node, name) => node.widgets?.find((w) => w.name === name);

function setVisible(w, visible) {
  if (!w.__plx) w.__plx = { type: w.type, computeSize: w.computeSize };
  w.hidden = !visible;
  if (w.element) {
    // DOM widgets (multiline text)
    w.element.hidden = !visible;
    w.element.style.display = visible ? "" : "none";
  } else {
    w.type = visible ? w.__plx.type : "plx-hidden";
  }
  w.computeSize = visible ? w.__plx.computeSize : () => [0, -4];
}

function applyOptions(node, schema, spec) {
  // Restrict combo lists to the values the selected model accepts
  for (const [name, values] of Object.entries(spec.options || {})) {
    const w = findWidget(node, name);
    if (!w || !w.options) continue;
    w.options.values = values;
    if (!values.includes(w.value)) {
      const preferred = schema.defaults?.[name];
      w.value = values.includes(preferred) ? preferred : values[0];
    }
  }
  // Numeric limits (num_images, max_images)
  for (const [name, [lo, hi]] of Object.entries(spec.ranges || {})) {
    const w = findWidget(node, name);
    if (!w || !w.options) continue;
    w.options.min = lo;
    w.options.max = hi;
    if (w.value > hi) w.value = hi;
    if (w.value < lo) w.value = lo;
  }
}

async function refresh(node) {
  const schema = await getSchema();
  if (!schema) return;
  const model = findWidget(node, "model")?.value;
  const spec = schema.models[model];
  if (!spec) return;

  const show = new Set([...schema.always, ...spec.widgets]);
  const preset = findWidget(node, "size_preset")?.value;

  for (const name of schema.managed) {
    const w = findWidget(node, name);
    if (!w) continue;
    let visible = show.has(name);
    if (name === "custom_width" || name === "custom_height") {
      visible = visible && preset === "custom";
    }
    setVisible(w, visible);
  }

  applyOptions(node, schema, spec);

  // Re-evaluate custom size visibility in case the preset value was just reset
  const preset2 = findWidget(node, "size_preset")?.value;
  if (preset2 !== preset) {
    for (const name of ["custom_width", "custom_height"]) {
      const w = findWidget(node, name);
      if (w) setVisible(w, show.has(name) && preset2 === "custom");
    }
  }

  const size = node.computeSize();
  node.setSize([Math.max(node.size[0], size[0]), size[1]]);
  app.graph?.setDirtyCanvas(true, true);
}

function hook(node) {
  if (node.__plxHooked) return;
  node.__plxHooked = true;
  for (const name of ["model", "size_preset"]) {
    const w = findWidget(node, name);
    if (!w) continue;
    const original = w.callback;
    w.callback = function () {
      const r = original?.apply(this, arguments);
      refresh(node);
      return r;
    };
  }
}

app.registerExtension({
  name: "plexard.fal.multiimage",
  async nodeCreated(node) {
    if (node.comfyClass !== NODE_ID) return;
    hook(node);
    setTimeout(() => refresh(node), 0);
  },
  async loadedGraphNode(node) {
    if (node.comfyClass !== NODE_ID) return;
    hook(node);
    setTimeout(() => refresh(node), 0);
  },
});
