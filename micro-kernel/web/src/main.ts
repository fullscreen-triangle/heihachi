/**
 * The bench.
 *
 * Three collapsable columns: files, editor, output. The IDE computes
 * nothing -- it sends source to the daemon to be checked and run, and
 * renders what the graph says. That division is why the browser can be
 * anywhere while the compute stays on the machine.
 */

import { daemon, type ConnectionState } from './lib/daemon';
import { clear, el, on } from './lib/dom';
import { renderOutput, type OutputState, type TabId } from './panels/output';
import type {
  CheckResult,
  ExportEvent,
  FileEntry,
  GraphSnapshot,
  RunResult,
  ServerMessage,
  StudioStatus,
} from './lib/protocol';

interface AppState {
  connection: ConnectionState;
  files: FileEntry[];
  openPath: string | null;
  source: string;
  dirty: boolean;
  collapsed: { files: boolean; output: boolean };
  output: OutputState;
  busy: boolean;
  pairError: string | null;
}

const state: AppState = {
  connection: { status: 'unpaired' },
  files: [],
  openPath: null,
  source: '',
  dirty: false,
  collapsed: { files: false, output: false },
  output: {
    tab: 'studio',
    mishima: null,
    sangoma: null,
    mishimaCheck: null,
    sangomaCheck: null,
    graph: null,
    exports: [],
    status: null,
  },
  busy: false,
  pairError: null,
};

const root = document.getElementById('app')!;

function languageOf(path: string | null): 'mishima' | 'sangoma' {
  return path?.endsWith('.sgn') ? 'sangoma' : 'mishima';
}

// ── render ─────────────────────────────────────────────────────────

function render(): void {
  clear(root);
  root.append(renderTopBar());
  if (state.connection.status === 'unpaired' || state.connection.status === 'failed') {
    root.append(renderPairScreen());
    return;
  }
  root.append(renderColumns());
}

function renderTopBar(): HTMLElement {
  const conn = state.connection;
  const dotClass =
    conn.status === 'connected' ? 'live' : conn.status === 'connecting' ? 'warn' : 'bad';
  const label =
    conn.status === 'connected'
      ? `daemon ${conn.version}`
      : conn.status === 'connecting'
        ? 'connecting'
        : conn.status === 'failed'
          ? conn.reason
          : 'not paired';

  const status = state.output.status;
  const bar = el('div', { class: 'topbar' }, [
    el('div', { class: 'brand' }, ['heihachi ', el('span', {}, ['micro-kernel'])]),
    el('span', { class: 'pill' }, [el('span', { class: `dot ${dotClass}` }), label]),
  ]);

  if (status) {
    bar.append(
      el('span', { class: 'pill' }, [
        el('span', { class: `dot ${status.fl_watching ? 'live' : ''}` }),
        status.fl_watching ? `watching · ${status.fl_exports_seen} renders` : 'not watching',
      ]),
      el('span', { class: 'pill' }, [
        el('span', { class: `dot ${status.ollama_reachable ? 'live' : 'bad'}` }),
        status.ollama_reachable ? status.ollama_model : 'model offline',
      ]),
    );
  }

  bar.append(el('div', { class: 'spacer' }));

  if (state.connection.status === 'connected') {
    const check = el('button', {}, ['Check']);
    on(check, 'click', () => void runProgram(false));
    const run = el('button', { class: 'primary' }, ['Run']);
    on(run, 'click', () => void runProgram(true));
    const save = el(
      'button',
      state.dirty ? {} : { disabled: 'true' },
      [state.dirty ? 'Save*' : 'Save'],
    );
    on(save, 'click', () => void saveFile());
    if (state.busy) {
      check.setAttribute('disabled', 'true');
      run.setAttribute('disabled', 'true');
    }
    bar.append(save, check, run);
  }
  return bar;
}

function renderPairScreen(): HTMLElement {
  const tokenInput = el('input', {
    type: 'password',
    placeholder: 'hk_...',
    autocomplete: 'off',
  }) as HTMLInputElement;
  const endpointInput = el('input', {
    type: 'text',
    value: daemon.endpoint,
  }) as HTMLInputElement;

  const submit = el('button', { class: 'primary' }, ['Pair']);
  const attempt = async () => {
    state.pairError = null;
    submit.setAttribute('disabled', 'true');
    try {
      await daemon.pair(tokenInput.value, endpointInput.value);
      // pair() may have found the daemon on a neighbouring port; show where
      // it actually landed rather than the address that failed
      endpointInput.value = daemon.endpoint;
      await loadEverything();
    } catch (error) {
      state.pairError = error instanceof Error ? error.message : String(error);
      state.connection = { status: 'unpaired' };
      render();
    }
  };
  on(submit, 'click', () => void attempt());
  on(tokenInput, 'keydown', (ev) => {
    if ((ev as KeyboardEvent).key === 'Enter') void attempt();
  });

  return el('div', { class: 'pair-screen' }, [
    el('div', { class: 'pair-card' }, [
      el('h2', {}, ['Connect to your machine']),
      el('p', {}, [
        'This page has no compute of its own. Start the daemon and paste the ',
        'token it prints; the runtime graph, the analysis and the model all ',
        'run locally, and nothing leaves your machine.',
      ]),
      el('pre', {}, ['heihachi serve --watch "D:\\Renders"']),
      el('label', {}, ['Daemon address']),
      endpointInput,
      el('p', { class: 'pair-hint' }, [
        'Must match the address on the daemon banner "listening" line. It differs ',
        'from the default whenever you passed --addr, which is what you do ',
        'when the usual port is already taken.',
      ]),
      el('label', {}, ['Pairing token']),
      tokenInput,
      state.pairError ? el('div', { class: 'pair-error' }, [state.pairError]) : null,
      submit,
    ]),
  ]);
}

function renderColumns(): HTMLElement {
  return el('div', { class: 'columns' }, [
    renderFilesColumn(),
    renderEditorColumn(),
    renderOutputColumn(),
  ]);
}

function renderFilesColumn(): HTMLElement {
  const collapsed = state.collapsed.files;
  const toggle = el('button', {}, [collapsed ? '›' : '‹']);
  on(toggle, 'click', () => {
    state.collapsed.files = !collapsed;
    render();
  });

  const column = el('div', { class: `column files${collapsed ? ' collapsed' : ''}` }, [
    el('div', { class: 'col-head' }, [
      toggle,
      el('span', { class: 'col-head-label' }, ['files']),
    ]),
  ]);
  if (collapsed) return column;

  const body = el('div', { class: 'col-body' });
  for (const group of ['mishima', 'sangoma'] as const) {
    const files = state.files.filter((f) => f.language === group);
    const items = files.map((f) => {
      const item = el(
        'button',
        { class: `tree-item${state.openPath === f.path ? ' active' : ''}` },
        [f.name.replace(/\.(mma|sgn)$/, ''), el('span', { class: 'ext' }, [
          f.name.endsWith('.sgn') ? '.sgn' : '.mma',
        ])],
      );
      on(item, 'click', () => void openFile(f.path));
      return item;
    });
    body.append(
      el('div', { class: 'tree-group' }, [
        el('div', { class: 'tree-group-label' }, [group]),
        ...(items.length > 0
          ? items
          : [el('div', { class: 'tree-item hint' }, ['(none yet)'])]),
      ]),
    );
  }
  column.append(body);
  return column;
}

function renderEditorColumn(): HTMLElement {
  const column = el('div', { class: 'column editor' }, [
    el('div', { class: 'col-head' }, [
      el('span', { class: 'col-head-label' }, [
        state.openPath ?? 'no file open',
        state.dirty ? ' *' : '',
      ]),
    ]),
  ]);

  const gutter = el('div', { class: 'gutter' });
  const lines = state.source.split('\n').length;
  const check =
    languageOf(state.openPath) === 'sangoma'
      ? state.output.sangomaCheck
      : state.output.mishimaCheck;
  const marked = new Map<number, string>();
  for (const d of check?.diagnostics ?? []) {
    if (d.severity === 'error') marked.set(d.line, 'has-error');
    else if (!marked.has(d.line)) marked.set(d.line, 'has-warn');
  }
  for (let i = 1; i <= Math.max(lines, 1); i += 1) {
    gutter.append(el('div', { class: marked.get(i) ?? '' }, [String(i)]));
  }

  const area = el('textarea', {
    class: 'code',
    spellcheck: 'false',
    placeholder: 'Open a file, or ask below for one to be written.',
  }) as HTMLTextAreaElement;
  area.value = state.source;
  on(area, 'input', () => {
    state.source = area.value;
    state.dirty = true;
    // redraw the gutter only, so typing does not lose the caret
    const g = column.querySelector('.gutter') as HTMLElement | null;
    if (g) {
      clear(g);
      const n = state.source.split('\n').length;
      for (let i = 1; i <= Math.max(n, 1); i += 1) {
        g.append(el('div', { class: marked.get(i) ?? '' }, [String(i)]));
      }
    }
    const head = column.querySelector('.col-head-label');
    if (head && !head.textContent?.endsWith('*')) {
      head.textContent = `${state.openPath ?? 'untitled'} *`;
    }
  });
  on(area, 'scroll', () => {
    const g = column.querySelector('.gutter') as HTMLElement | null;
    if (g) g.scrollTop = area.scrollTop;
  });

  column.append(el('div', { class: 'editor-wrap' }, [gutter, area]));
  column.append(renderComposeBar());
  return column;
}

/**
 * Asking for a program in English.
 *
 * The model writes source; it never answers. What comes back lands in the
 * editor for review and is not run, because a program nobody read is a
 * program nobody authored.
 */
function renderComposeBar(): HTMLElement {
  const input = el('input', {
    type: 'text',
    placeholder: 'describe what you want, and a program is written for you to review…',
  }) as HTMLInputElement;
  const button = el('button', {}, ['Compose']);

  const go = async () => {
    const request = input.value.trim();
    if (!request) return;
    button.setAttribute('disabled', 'true');
    try {
      const result = await daemon.compose(languageOf(state.openPath), request);
      state.source = result.source;
      state.dirty = true;
      render();
    } catch (error) {
      state.output.status = state.output.status;
      window.alert(error instanceof Error ? error.message : String(error));
    } finally {
      button.removeAttribute('disabled');
    }
  };
  on(button, 'click', () => void go());
  on(input, 'keydown', (ev) => {
    if ((ev as KeyboardEvent).key === 'Enter') void go();
  });

  return el('div', { class: 'compose-bar' }, [input, button]);
}

function renderOutputColumn(): HTMLElement {
  const collapsed = state.collapsed.output;
  const toggle = el('button', {}, [collapsed ? '‹' : '›']);
  on(toggle, 'click', () => {
    state.collapsed.output = !collapsed;
    render();
  });

  const column = el('div', { class: `column output${collapsed ? ' collapsed' : ''}` }, [
    el('div', { class: 'col-head' }, [
      toggle,
      el('span', { class: 'col-head-label' }, ['output']),
    ]),
  ]);
  if (collapsed) return column;

  const tabs: [TabId, string][] = [
    ['mishima', 'mishima'],
    ['sangoma', 'sangoma'],
    ['graph', 'runtime graph'],
    ['studio', 'studio'],
  ];
  const tabBar = el('div', { class: 'tabs' });
  for (const [id, label] of tabs) {
    const tab = el('button', {
      class: `tab${state.output.tab === id ? ' active' : ''}`,
    }, [label]);
    on(tab, 'click', () => {
      state.output.tab = id;
      if (id === 'graph') void refreshGraph();
      render();
    });
    tabBar.append(tab);
  }

  column.append(tabBar);
  column.append(el('div', { class: 'col-body' }, [renderOutput(state.output)]));
  return column;
}

// ── actions ────────────────────────────────────────────────────────

async function openFile(path: string): Promise<void> {
  try {
    const { source } = await daemon.readFile(path);
    state.openPath = path;
    state.source = source;
    state.dirty = false;
    state.output.tab = languageOf(path);
    render();
  } catch (error) {
    window.alert(error instanceof Error ? error.message : String(error));
  }
}

async function saveFile(): Promise<void> {
  if (!state.openPath) return;
  await daemon.writeFile(state.openPath, state.source);
  state.dirty = false;
  render();
}

async function runProgram(execute: boolean): Promise<void> {
  const language = languageOf(state.openPath);
  state.busy = true;
  render();
  try {
    if (execute) {
      const result: RunResult & { check?: CheckResult } = await daemon.run(
        language,
        state.source,
      );
      if (language === 'sangoma') {
        state.output.sangoma = result;
        state.output.sangomaCheck = result.check ?? null;
      } else {
        state.output.mishima = result;
        state.output.mishimaCheck = result.check ?? null;
      }
      await refreshGraph();
    } else {
      const check = await daemon.check(language, state.source);
      if (language === 'sangoma') state.output.sangomaCheck = check;
      else state.output.mishimaCheck = check;
    }
    state.output.tab = language;
  } catch (error) {
    window.alert(error instanceof Error ? error.message : String(error));
  } finally {
    state.busy = false;
    render();
  }
}

async function refreshGraph(): Promise<void> {
  try {
    state.output.graph = (await daemon.graph()) as GraphSnapshot;
  } catch {
    /* the graph view will say it is empty */
  }
}

async function loadEverything(): Promise<void> {
  try {
    const [files, status, exports] = await Promise.all([
      daemon.files(),
      daemon.status(),
      daemon.exports(),
    ]);
    state.files = files;
    state.output.status = status as StudioStatus;
    state.output.exports = exports as ExportEvent[];
    await refreshGraph();
    if (!state.openPath && files.length > 0) {
      await openFile(files[0].path);
      return;
    }
  } catch (error) {
    state.pairError = error instanceof Error ? error.message : String(error);
  }
  render();
}

// ── wiring ─────────────────────────────────────────────────────────

daemon.onState((connection) => {
  state.connection = connection;
  render();
});

daemon.onMessage((message: ServerMessage) => {
  switch (message.type) {
    case 'status':
      state.output.status = message.status;
      render();
      break;
    case 'export':
      state.output.exports = [...state.output.exports, message.event];
      // a render appearing is the reason to look at the studio
      void refreshGraph().then(render);
      break;
    case 'delta':
      void refreshGraph().then(() => {
        if (state.output.tab === 'graph') render();
      });
      break;
    case 'error':
      console.warn('daemon:', message.message);
      break;
    default:
      break;
  }
});

if (daemon.token) {
  daemon.connect();
  void loadEverything();
} else {
  render();
}
