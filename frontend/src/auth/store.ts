import { useSyncExternalStore } from "react";
import { decodeToken, type User } from "./user";

/*
 * The auth state, outside React so the Apollo links can read the token per
 * request. The JWT is kept in sessionStorage under the same key the lucuma-ts
 * apps use, so a reload keeps the session without a round trip to SSO.
 */

const STORAGE_KEY = "lucuma-refresh-token";

export type SessionStatus = "checking" | "signed-out" | "signed-in";

export interface AuthState {
  readonly token: string | null;
  readonly user: User | null;
  readonly exp: Date | null;
  readonly status: SessionStatus;
}

function readStoredToken(): string | null {
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as string) : null;
  } catch {
    return null;
  }
}

function writeStoredToken(token: string | null): void {
  try {
    if (token === null) window.sessionStorage.removeItem(STORAGE_KEY);
    else window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(token));
  } catch {
    // Storage can be denied (private mode); the session just won't survive a reload.
  }
}

function build(token: string | null, checked: boolean): AuthState {
  const decoded = token ? decodeToken(token) : null;
  const live = decoded !== null && decoded.exp.getTime() > Date.now();
  if (live)
    return { token, user: decoded.user, exp: decoded.exp, status: "signed-in" };
  return {
    token: null,
    user: null,
    exp: null,
    status: checked ? "signed-out" : "checking",
  };
}

// A live stored token opens the app at once; otherwise wait for the first SSO answer.
let state: AuthState = build(readStoredToken(), false);
const listeners = new Set<() => void>();

function emit(): void {
  listeners.forEach((l) => l());
}

export const getAuthState = (): AuthState => state;

export function subscribeAuth(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Store a new token (or null to sign out locally). An undecodable or expired
 *  token counts as signed out. */
export function setToken(token: string | null): void {
  const next = build(token, true);
  writeStoredToken(next.token);
  if (next.token === state.token && next.status === state.status) return;
  state = next;
  emit();
}

/** The first SSO answer settled with no token: leave `checking`. */
export function markChecked(): void {
  if (state.status !== "checking") return;
  state = build(state.token, true);
  emit();
}

export const useAuth = (): AuthState =>
  useSyncExternalStore(subscribeAuth, getAuthState);
