/*
 * SSO endpoint transport, ported from lucuma-ts (packages/common-ui/src/sso.ts
 * and packages/resource-ui/src/auth/ssoClient.ts) so the scheduler signs in the
 * same way as the other GPP apps. Raw HTTP calls against the SSO origin, with
 * no UI concerns.
 *
 *   POST /api/v1/refresh-token   → fresh JWT (text body) from the session cookie
 *   GET  /auth/v1/stage1?state=  → ORCID login redirect
 *   GET  /auth/v1/set-role?role= → switch active role (then refresh)
 *   POST /api/v1/logout          → clear the session
 *
 * All are cookie-based, so every call sends credentials. The cookie belongs to
 * the SSO domain and is SameSite=Strict, so these flows only complete from an
 * https *.gemini.edu origin, never from bare localhost (see the README).
 */

export type RefreshResult =
  | { readonly kind: "token"; readonly token: string }
  | { readonly kind: "rejected" }
  | { readonly kind: "unreachable" };

const UNREACHABLE: RefreshResult = { kind: "unreachable" };

/** Exchange the SSO session cookie for a fresh JWT. A 401/403 or an empty
 *  body means there is no session (`rejected`); anything else that fails is
 *  `unreachable`, so the caller can keep its token and retry later. */
export async function refreshSession(
  ssoUri: string,
  signal: AbortSignal,
  timeoutMs: number,
): Promise<RefreshResult> {
  const deadline = new AbortController();
  const followCaller = () => deadline.abort(signal.reason);
  if (signal.aborted) followCaller();
  else signal.addEventListener("abort", followCaller, { once: true });
  const timer = setTimeout(() => deadline.abort(), timeoutMs);
  try {
    const res = await fetch(new URL("/api/v1/refresh-token", ssoUri), {
      method: "POST",
      credentials: "include",
      signal: deadline.signal,
    });
    if (res.status === 401 || res.status === 403) return { kind: "rejected" };
    if (!res.ok) return UNREACHABLE;
    const token = (await res.text()).trim();
    return token === "" ? { kind: "rejected" } : { kind: "token", token };
  } catch {
    return UNREACHABLE;
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", followCaller);
  }
}

/** The ORCID sign-in URL; navigating here begins the login redirect. SSO
 *  returns to `returnTo` only when it is on this origin, else to the app root,
 *  so sign-in can never send the user to another site. */
export function signInUrl(
  ssoUri: string,
  returnTo: string,
  origin: string = window.location.origin,
): string {
  const root = new URL("/", origin);
  const url = new URL("/auth/v1/stage1", ssoUri);
  url.searchParams.set("state", sameOriginOr(root, returnTo).toString());
  return url.toString();
}

function sameOriginOr(root: URL, returnTo: string): URL {
  try {
    const target = new URL(returnTo, root);
    return target.origin === root.origin ? target : root;
  } catch {
    return root;
  }
}

/** SSO puts the human-readable reason for a failure in the body. */
async function failureDetail(res: Response): Promise<string> {
  const body = (await res.text().catch(() => "")).trim();
  return body ? `${res.status}: ${body}` : String(res.status);
}

/** Switch the session's active role and return the JWT that reflects it.
 *  Throws when SSO rejects the switch or returns no fresh token. */
export async function setRole(
  ssoUri: string,
  roleId: string,
  timeoutMs: number,
): Promise<string> {
  const url = new URL("/auth/v1/set-role", ssoUri);
  url.searchParams.set("role", roleId);
  const res = await fetch(url, { method: "GET", credentials: "include" });
  if (!res.ok)
    throw new Error(
      `SSO rejected the role switch (${await failureDetail(res)}).`,
    );
  const fresh = await refreshSession(
    ssoUri,
    new AbortController().signal,
    timeoutMs,
  );
  if (fresh.kind !== "token")
    throw new Error(
      "SSO accepted the role switch but returned no fresh token.",
    );
  return fresh.token;
}

/** End the SSO session. Throws when SSO rejects the call or is unreachable. */
export async function logout(ssoUri: string): Promise<void> {
  const res = await fetch(new URL("/api/v1/logout", ssoUri), {
    method: "POST",
    credentials: "include",
  });
  if (!res.ok)
    throw new Error(`SSO logout failed (${await failureDetail(res)}).`);
}
