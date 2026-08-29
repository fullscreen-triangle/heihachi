/**
 * The output column.
 *
 * Five views over one run: what mishima said, what sangoma said, the
 * runtime graph, the studio, and the accountability query. They are tabs
 * rather than panes because a producer looks at one at a time, and the
 * column collapses entirely when they want the editor wide.
 */

import type {
  CheckResult,
  Diagnostic,
  ExportEvent,
  GraphSnapshot,
  RunResult,
  StudioStatus,
  Value,
} from '../lib/protocol';
import { el, fmt } from '../lib/dom';

export type TabId = 'mishima' | 'sangoma' | 'graph' | 'studio';

export interface OutputState {
  tab: TabId;
  mishima: RunResult | null;
  sangoma: RunResult | null;
  mishimaCheck: CheckResult | null;
  sangomaCheck: CheckResult | null;
  graph: GraphSnapshot | null;
  exports: ExportEvent[];
  status: StudioStatus | null;
}

export function renderOutput(state: OutputState): HTMLElement {
  switch (state.tab) {
    case 'mishima':
      return renderRun(state.mishima, state.mishimaCheck, 'mishima', '.mma');
    case 'sangoma':
      return renderRun(state.sangoma, state.sangomaCheck, 'sangoma', '.sgn');
    case 'graph':
      return renderGraph(state.graph);
    case 'studio':
      return renderStudio(state.exports, state.status);
  }
}

// ── run output ─────────────────────────────────────────────────────

function renderRun(
  run: RunResult | null,
  check: CheckResult | null,
  language: string,
  ext: string,
): HTMLElement {
  const pane = el('div', { class: 'pane' });

  if (!check && !run) {
    pane.append(
      el('div', { class: 'empty' }, [
        `Nothing run yet. Open a ${ext} file and press Run.`,
        el('br'),
        el('br'),
        el('span', { class: 'hint' }, [
          'Check reports diagnostics without running; a refused program ',
          'does not run, and its diagnostics are the result.',
        ]),
      ]),
    );
    return pane;
  }

  const diagnostics = run?.report ? check?.diagnostics ?? [] : check?.diagnostics ?? [];
  if (diagnostics.length > 0) {
    for (const d of diagnostics) pane.append(renderDiagnostic(d));
  }

  if (check && !check.accepted) {
    pane.append(
      el('div', { class: 'hint' }, [
        'This program was refused, so it did not run. Every refusal above ',
        'carries a remedy.',
      ]),
    );
    return pane;
  }

  if (!run) return pane;

  // A contested closure is not an error. It reports that several
  // irreconcilable regions were reached and none is licensed over the
  // others -- which is usually more useful than either alone.
  if (run.decline) {
    const box = el('div', { class: 'decline-box' }, [
      el('h4', {}, ['declined — the evidence does not single one out']),
      el('div', {}, [`reached: ${run.decline.classes.join('  ·  ')}`]),
      el('div', { class: 'hint' }, [
        `probes invoked: ${run.decline.probes_invoked.join(', ')}`,
      ]),
    ]);
    if (run.decline.discriminating_probe) {
      box.append(
        el('div', { class: 'hint' }, [
          `the probe that would separate them: ${run.decline.discriminating_probe}`,
        ]),
      );
    }
    pane.append(box);
  }

  pane.append(renderStats(run));

  if (run.emissions.length > 0) {
    pane.append(renderValues(run.emissions));
  } else {
    pane.append(
      el('div', { class: 'hint' }, [`${language} emitted nothing this run.`]),
    );
  }
  return pane;
}

function renderDiagnostic(d: Diagnostic): HTMLElement {
  return el('div', { class: `diag ${d.severity}` }, [
    el('div', { class: 'rule' }, [`${d.severity} · ${d.rule}`]),
    el('div', { class: 'msg' }, [d.message]),
    el('div', { class: 'remedy' }, [`remedy: ${d.remedy}`]),
    el('div', { class: 'where' }, [`line ${d.line}`]),
  ]);
}

/**
 * The run's numbers. Note there is no pass/fail here, and there must not
 * be: the runtime holds no expectation, so it has nothing to compare an
 * achieved value against. Anomalies are counted, not judged.
 */
function renderStats(run: RunResult): HTMLElement {
  const r = run.report;
  return el('div', { class: 'stat-row' }, [
    stat('record', r.record),
    stat('emissions', r.emissions),
    stat('anomalies', r.anomalies),
    stat('edges', r.induced_edges),
  ]);
}

function stat(k: string, v: number | string): HTMLElement {
  return el('div', { class: 'stat' }, [
    el('div', { class: 'k' }, [k]),
    el('div', { class: 'v' }, [String(v)]),
  ]);
}

function renderValues(values: Value[]): HTMLElement {
  const rows = values.map((v) =>
    el('tr', { class: v.kind === 'reading' ? '' : v.kind }, [
      el('td', {}, [String(v.record)]),
      el('td', {}, [v.channel]),
      el('td', { class: 'num' }, [fmt(v.magnitude)]),
      el('td', {}, [v.unit]),
      el('td', { class: 'num floor' }, [`±${fmt(v.floor)}`]),
    ]),
  );
  return el('table', { class: 'values' }, [
    el('thead', {}, [
      el('tr', {}, [
        el('th', {}, ['#']),
        el('th', {}, ['channel']),
        el('th', {}, ['value']),
        el('th', {}, ['unit']),
        el('th', {}, ['resolution']),
      ]),
    ]),
    el('tbody', {}, rows),
  ]);
}

// ── graph ──────────────────────────────────────────────────────────

/**
 * The runtime graph, laid out on a circle.
 *
 * Deliberately not force-directed: a layout that moves while you read it
 * makes it impossible to tell whether a node appeared or merely drifted,
 * and the point of the view is to watch nodes appear.
 */
function renderGraph(graph: GraphSnapshot | null): HTMLElement {
  if (!graph || graph.nodes.length === 0) {
    return el('div', { class: 'pane' }, [
      el('div', { class: 'empty' }, [
        'The graph is empty.',
        el('br'),
        el('br'),
        el('span', { class: 'hint' }, [
          'Render something into the watched folder, or run a program. ',
          'Nodes appear as subtasks are raised.',
        ]),
      ]),
    ]);
  }

  const size = 460;
  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2 - 64;
  const positions = new Map<string, [number, number]>();

  graph.nodes.forEach((node, i) => {
    const angle = (i / graph.nodes.length) * Math.PI * 2 - Math.PI / 2;
    positions.set(node.tau, [
      cx + radius * Math.cos(angle),
      cy + radius * Math.sin(angle),
    ]);
  });

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'graph');
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

  for (const [from, to] of graph.edges) {
    const a = positions.get(from);
    const b = positions.get(to);
    if (!a || !b) continue;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('class', 'edge');
    line.setAttribute('x1', String(a[0]));
    line.setAttribute('y1', String(a[1]));
    line.setAttribute('x2', String(b[0]));
    line.setAttribute('y2', String(b[1]));
    svg.append(line);
  }

  const trajectory = new Set(graph.trajectory);
  for (const node of graph.nodes) {
    const [x, y] = positions.get(node.tau)!;
    const anomalous = node.values.some((v) => v.kind === 'anomaly');
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute(
      'class',
      `node${anomalous ? ' anomalous' : ''}${
        trajectory.has(node.tau) ? ' in-trajectory' : ''
      }`,
    );

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', String(x));
    circle.setAttribute('cy', String(y));
    circle.setAttribute('r', String(6 + Math.min(10, node.values.length)));
    g.append(circle);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', String(x));
    label.setAttribute('y', String(y - 16));
    label.setAttribute('text-anchor', 'middle');
    label.textContent = node.tau;
    g.append(label);

    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = `${node.tau}\n${node.values.length} values, ${node.chunks.length} chunks`;
    g.append(title);

    svg.append(g);
  }

  const wrap = el('div', { class: 'graph-wrap' });
  wrap.append(svg);

  const legend = el('div', { class: 'pane hint' }, [
    `${graph.nodes.length} nodes · ${graph.edges.length} induced edges · `,
    `record ${graph.report.record}`,
    el('br'),
    'Filled nodes carried propagated information; an emission nothing read ',
    'forms no edge, and nothing rejected it.',
  ]);

  return el('div', { style: 'display:flex;flex-direction:column;height:100%' }, [
    wrap,
    legend,
  ]);
}

// ── studio ─────────────────────────────────────────────────────────

function renderStudio(
  exports: ExportEvent[],
  status: StudioStatus | null,
): HTMLElement {
  const pane = el('div', { class: 'pane' });

  if (status) {
    pane.append(
      el('div', { class: 'stat-row' }, [
        stat('renders seen', status.fl_exports_seen),
        stat('model', status.ollama_reachable ? status.ollama_model : 'offline'),
      ]),
    );
    pane.append(
      el('div', { class: 'hint', style: 'margin-bottom:12px' }, [
        status.fl_watching
          ? `watching ${status.fl_watching}`
          : 'watching nothing — restart the daemon with --watch <render folder>',
      ]),
    );
  }

  if (exports.length === 0) {
    pane.append(
      el('div', { class: 'empty' }, [
        'No renders yet.',
        el('br'),
        el('br'),
        el('span', { class: 'hint' }, [
          'Export from FL Studio into the watched folder. Nothing is ',
          'installed into FL and nothing is controlled remotely: you render ',
          'as normal, and the render is measured and committed.',
        ]),
      ]),
    );
    return pane;
  }

  for (const e of [...exports].reverse()) {
    const card = el('div', { class: 'export' }, [
      el('h4', {}, [e.stem]),
    ]);

    if (e.audio.measurements.length > 0) {
      card.append(renderMeasurements(e));
    }

    if (e.project?.devices.length) {
      card.append(
        el('div', { class: 'chips' }, [
          ...e.project.devices.map((d) => el('span', { class: 'chip device' }, [d])),
        ]),
      );
    }

    if (e.distinctions.length > 0) {
      card.append(
        el('div', { class: 'chips' }, [
          ...e.distinctions.map((d) => el('span', { class: 'chip' }, [d])),
        ]),
      );
    }

    // A rung that could not run says so. Three different explanations are
    // three different lines, never a blank.
    for (const note of e.notes) {
      card.append(el('div', { class: 'note hint' }, [note]));
    }
    pane.append(card);
  }
  return pane;
}

function renderMeasurements(e: ExportEvent): HTMLElement {
  return el('table', { class: 'values' }, [
    el(
      'tbody',
      {},
      e.audio.measurements.map((m) =>
        el('tr', {}, [
          el('td', {}, [m.channel]),
          el('td', { class: 'num' }, [m.value.toFixed(3)]),
          el('td', {}, [m.unit]),
          el('td', { class: 'num floor' }, [`±${fmt(m.floor)}`]),
        ]),
      ),
    ),
  ]);
}
