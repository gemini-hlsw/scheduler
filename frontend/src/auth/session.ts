import { REQUIRED_ACCESS, SSO_URI } from "./environment";
import * as sso from "./sso";
import { getAuthState, markChecked, setToken, subscribeAuth } from "./store";
import { accessAtLeast, eligibleOtherRole } from "./user";

/*
 * Keeps the SSO session alive, following lucuma-ts resource-ui
 * (src/auth/session.ts) and admin-ui (src/auth/AuthSession.tsx):
 *
 * - a silent cookie refresh on start, so a returning user is signed in
 *   without clicking;
 * - a refresh shortly before each token expires, and the token dropped when
 *   it does expire;
 * - SSO unreachable keeps the current token and retries with backoff; SSO
 *   refusing the cookie signs out;
 * - a refresh when the tab becomes visible again, if one is due;
 * - signing out in one tab signs out the others (BroadcastChannel);
 * - after sign-in, a one-time switch to the user's staff/admin role when the
 *   active role is too low for the scheduler.
 */

const REFRESH_AHEAD_MS = 30_000;
const MIN_INTERVAL_MS = 30_000;
const BACKOFF_CAP_MS = 16 * 60_000;
const REFRESH_TIMEOUT_MS = 10_000;
// A longer setTimeout delay overflows and fires at once.
const MAX_TIMER_MS = 2_147_483_647;

const CHANNEL = "scheduler-session";
const SIGNED_OUT = "signed-out";

interface Keeper {
  stop(): void;
  endLocally(): void;
  channel: BroadcastChannel;
}

let current: Keeper | null = null;

const clampDelay = (ms: number) => Math.min(MAX_TIMER_MS, Math.max(0, ms));

function createKeeper(): Keeper {
  let refreshTimer: ReturnType<typeof setTimeout> | undefined;
  let expiryTimer: ReturnType<typeof setTimeout> | undefined;
  let inFlight: AbortController | null = null;
  let lastAttemptAt = 0;
  let backoffMs = 0;
  // After a sign-out the SSO cookie may outlive a failed logout, so nothing
  // renews from it until a new token arrives by sign-in.
  let ended = false;
  let roleSwitchTriedFor: string | null = null;

  const clearTimers = () => {
    clearTimeout(refreshTimer);
    clearTimeout(expiryTimer);
    refreshTimer = expiryTimer = undefined;
  };

  /** When the next refresh may run, or null when none is needed. */
  const nextRefreshAt = (): number | null => {
    if (ended) return null;
    const { exp } = getAuthState();
    let at: number | null = exp ? exp.getTime() - REFRESH_AHEAD_MS : null;
    if (backoffMs > 0) at = Math.max(at ?? 0, lastAttemptAt + backoffMs);
    if (at === null) return null;
    return Math.max(at, lastAttemptAt + MIN_INTERVAL_MS);
  };

  const arm = () => {
    clearTimers();
    const { exp } = getAuthState();
    if (exp) {
      expiryTimer = setTimeout(
        () => {
          if ((getAuthState().exp?.getTime() ?? 0) <= Date.now())
            setToken(null);
          else arm();
        },
        clampDelay(exp.getTime() - Date.now()),
      );
    }
    const at = nextRefreshAt();
    if (at !== null && inFlight === null) {
      refreshTimer = setTimeout(
        () => void refresh(),
        clampDelay(at - Date.now()),
      );
    }
  };

  const refresh = async () => {
    if (inFlight !== null || ended) return;
    const controller = new AbortController();
    inFlight = controller;
    lastAttemptAt = Date.now();
    const result = await sso.refreshSession(
      SSO_URI,
      controller.signal,
      REFRESH_TIMEOUT_MS,
    );
    if (inFlight !== controller) return; // stopped, signed out, or superseded
    inFlight = null;
    switch (result.kind) {
      case "token":
        backoffMs = 0;
        setToken(result.token);
        break;
      case "rejected":
        backoffMs = 0;
        setToken(null);
        break;
      case "unreachable":
        backoffMs =
          backoffMs === 0
            ? MIN_INTERVAL_MS
            : Math.min(backoffMs * 2, BACKOFF_CAP_MS);
        console.warn(`SSO unreachable; retrying in ${backoffMs / 1000} s.`);
        break;
    }
    markChecked();
    arm();
  };

  const maybeSwitchRole = () => {
    const { token, user } = getAuthState();
    if (!token || accessAtLeast(user, REQUIRED_ACCESS)) return;
    if (roleSwitchTriedFor === token) return;
    const role = eligibleOtherRole(user, REQUIRED_ACCESS);
    if (!role) return;
    roleSwitchTriedFor = token;
    sso
      .setRole(SSO_URI, role.id, REFRESH_TIMEOUT_MS)
      .then((fresh) => {
        if (!ended) setToken(fresh);
      })
      .catch((err: unknown) => console.error("Role switch failed", err));
  };

  const onAuthChange = () => {
    if (getAuthState().token !== null) {
      // A token from elsewhere (role switch) is newer than any answer in flight.
      ended = false;
      backoffMs = 0;
      inFlight?.abort();
      inFlight = null;
      maybeSwitchRole();
    }
    arm();
  };

  const onVisible = () => {
    if (document.visibilityState !== "visible") return;
    const now = Date.now();
    if (ended || now - lastAttemptAt < MIN_INTERVAL_MS) return;
    if (backoffMs > 0 && now < lastAttemptAt + backoffMs) return;
    const { exp } = getAuthState();
    if (exp === null || now >= exp.getTime() - REFRESH_AHEAD_MS) void refresh();
  };

  const endLocally = () => {
    ended = true;
    inFlight?.abort();
    inFlight = null;
    setToken(null);
    arm();
  };

  const channel = new BroadcastChannel(CHANNEL);
  channel.onmessage = (e: MessageEvent) => {
    if (e.data === SIGNED_OUT) endLocally();
  };
  const unsubscribe = subscribeAuth(onAuthChange);
  document.addEventListener("visibilitychange", onVisible);

  if (getAuthState().status === "checking") void refresh();
  maybeSwitchRole();
  arm();

  return {
    channel,
    endLocally,
    stop() {
      clearTimers();
      inFlight?.abort();
      inFlight = null;
      unsubscribe();
      document.removeEventListener("visibilitychange", onVisible);
      channel.close();
    },
  };
}

/** Start keeping the session; returns the stop function (an effect cleanup). */
export function startSession(): () => void {
  current?.stop();
  const keeper = createKeeper();
  current = keeper;
  return () => {
    if (current === keeper) {
      keeper.stop();
      current = null;
    }
  };
}

/** Navigate to SSO's ORCID login, coming back to the current page. */
export const signIn = (): void => {
  window.location.assign(sso.signInUrl(SSO_URI, window.location.href));
};

/** Drop the local token first (that alone signs this app out), tell the other
 *  tabs, then end the SSO session. Resolves false when SSO was not reached, in
 *  which case the cookie lives on until it expires. */
export async function signOut(): Promise<boolean> {
  current?.channel.postMessage(SIGNED_OUT);
  if (current) current.endLocally();
  else setToken(null);
  return sso.logout(SSO_URI).then(
    () => true,
    (err: unknown) => {
      console.error(err);
      return false;
    },
  );
}
