import { writable, derived } from 'svelte/store';
import { apiUrl } from '$lib/api';

export interface HealthState {
  healthy: boolean;
  reachable: boolean;
  backend: string;
  base_url: string;
  models: string[];
  chat_model_found: boolean;
  embed_model_found: boolean;
  error?: string;
  last_checked: number | null;
  checking: boolean;
}

export interface AgentsState {
  agents: string[];
  loaded: boolean;
}

const initialHealth: HealthState = {
  healthy: false,
  reachable: false,
  backend: '',
  base_url: '',
  models: [],
  chat_model_found: false,
  embed_model_found: false,
  last_checked: null,
  checking: false
};

export const health = writable<HealthState>(initialHealth);
export const agents = writable<AgentsState>({ agents: [], loaded: false });

// Derived: simple connectivity label
export const connectivityLabel = derived(health, ($h) => {
  if ($h.checking) return 'checking';
  if ($h.healthy)  return 'online';
  if ($h.reachable) return 'degraded';
  return 'offline';
});

export async function checkHealth(): Promise<void> {
  health.update((s) => ({ ...s, checking: true }));
  try {
    const res = await fetch(apiUrl('/api/health'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    health.set({
      ...data,
      last_checked: Date.now(),
      checking: false
    });
  } catch (err) {
    health.update((s) => ({
      ...s,
      healthy: false,
      reachable: false,
      error: String(err),
      last_checked: Date.now(),
      checking: false
    }));
  }
}

export async function loadAgents(): Promise<void> {
  try {
    const res = await fetch(apiUrl('/api/agents'));
    if (!res.ok) return;
    const data = await res.json();
    agents.set({ agents: data.agents ?? [], loaded: true });
  } catch {
    // silently ignore — agents panel just shows empty
  }
}

// Poll health every 15 seconds (called once from root layout)
let _pollTimer: ReturnType<typeof setInterval> | null = null;

export function startHealthPolling(): void {
  if (_pollTimer) return;
  checkHealth();
  loadAgents();
  _pollTimer = setInterval(() => {
    checkHealth();
    loadAgents();
  }, 15_000);
}

export function stopHealthPolling(): void {
  if (_pollTimer) {
    clearInterval(_pollTimer);
    _pollTimer = null;
  }
}
